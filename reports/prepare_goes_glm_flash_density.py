"""Prepare ABI clouds, GLM flash counts, event energies, and a light curve.

Pixel outlines are reconstructed from local event-center spacing, rather than
claiming to reproduce NOAA's operational 2-km gridded product.
"""
from pathlib import Path
import json
import hashlib

import cartopy.io.shapereader as shp
import h5py
import numpy as np
from PIL import Image
from pyproj import Proj
from scipy.ndimage import map_coordinates
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "products/goes19_glm_20260530"
BOUNDS = [-73.2, -68.3, 40.3, 44.2]
TRAJECTORY = ROOT / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json"


def clusters(values):
    ordered = np.sort(values)
    centers = [np.mean(c) for c in np.split(ordered, np.where(np.diff(ordered) > 2000)[0]+1)]
    centers = np.asarray(centers)
    edges = np.r_[centers[0]-(centers[1]-centers[0])/2,
                  (centers[:-1]+centers[1:])/2,
                  centers[-1]+(centers[-1]-centers[-2])/2]
    return centers, edges


def main():
    west, east, south, north = BOUNDS
    abi = xr.open_dataset(next(DATA.glob("OR_ABI*.nc")))
    meta = abi.goes_imager_projection.attrs
    h = meta["perspective_point_height"]
    projection = Proj(proj="geos", h=h, lon_0=meta["longitude_of_projection_origin"],
                      a=meta["semi_major_axis"], b=meta["semi_minor_axis"],
                      sweep=meta["sweep_angle_axis"])
    lons, lats = np.meshgrid(np.linspace(west, east, 1000), np.linspace(north, south, 1050))
    px, py = projection(lons, lats)
    ix = (px/h-float(abi.x[0]))/float(abi.x[1]-abi.x[0])
    iy = (py/h-float(abi.y[0]))/float(abi.y[1]-abi.y[0])
    x0, x1 = int(np.floor(ix.min())), int(np.ceil(ix.max()))+1
    y0, y1 = int(np.floor(iy.min())), int(np.ceil(iy.max()))+1
    region = abi.isel(x=slice(x0,x1+1), y=slice(y0,y1+1))
    source = np.where(region.DQF.values <= 1, region.CMI.values, np.nan)
    cloud = map_coordinates(source, [iy-y0, ix-x0], order=0, mode="constant", cval=np.nan)
    gray = (np.nan_to_num(np.clip(cloud, 0, 1))*255).astype("uint8")
    Image.fromarray(gray).convert("RGBA").save(DATA / "cloud_background.png")
    lat, lon, flash, event_energy, event_seconds, event_ids = [], [], [], [], [], []
    sources = sorted(DATA.glob("OR_GLM*s20261501806*.nc"))
    if len(sources) != 3:
        raise ValueError("Need all three 20-second granules for the full minute")
    for source_path in sources:
        with xr.open_dataset(source_path) as ds:
            keep = ((ds.event_lat.values > 41.5) & (ds.event_lat.values < 43.5)
                    & (ds.event_lon.values > -71.5) & (ds.event_lon.values < -69.5)
                    & (ds.event_time_offset.values >= np.datetime64("2026-05-30T18:06:00"))
                    & (ds.event_time_offset.values < np.datetime64("2026-05-30T18:07:00")))
            parents = dict(zip(ds.group_id.values, ds.group_parent_flash_id.values))
            starts = dict(zip(ds.flash_id.values,
                              ds.flash_frame_time_offset_of_first_event.values.astype("int64")))
            lat.extend(ds.event_lat.values[keep])
            lon.extend(ds.event_lon.values[keep])
            flash.extend(starts[parents[g]] for g in ds.event_parent_group_id.values[keep])
            event_energy.extend(ds.event_energy.values[keep])
            event_seconds.extend((ds.event_time_offset.values[keep]
                                 - np.datetime64("2026-05-30T18:06:20"))/np.timedelta64(1,"s"))
            event_ids.extend(ds.event_id.values[keep])
    ex, ey = projection(np.asarray(lon), np.asarray(lat))
    xc, xe = clusters(ex)
    yc, ye = clusters(ey)
    col = np.abs(ex[:,None]-xc).argmin(axis=1)
    row = np.abs(ey[:,None]-yc).argmin(axis=1)
    # Count distinct parent flashes once per native GLM pixel for a 60-s interval.
    unique = np.unique(np.column_stack([row, col, flash]), axis=0)
    counts = np.zeros((len(yc),len(xc)), dtype="int32")
    for r,c,_ in unique:
        counts[int(r),int(c)] += 1
    event_energy = np.asarray(event_energy, dtype="float64")
    event_seconds = np.asarray(event_seconds, dtype="float64")
    bolide = ((event_seconds >= 0) & (event_seconds < 4)
              & np.isfinite(event_energy) & (event_energy >= 0))
    if np.unique(np.asarray(event_ids)[bolide]).size != bolide.sum():
        raise ValueError("Duplicate event IDs in bolide selection")
    energy_map = np.zeros_like(counts, dtype="float64")
    np.add.at(energy_map, (row[bolide], col[bolide]), event_energy[bolide])
    time_edges = np.linspace(0,4,401)
    light_curve, _ = np.histogram(event_seconds[bolide],bins=time_edges,weights=event_energy[bolide])
    total_energy = event_energy[bolide].sum()
    np.testing.assert_allclose(energy_map.sum(),total_energy,rtol=1e-12,atol=0)
    np.testing.assert_allclose(light_curve.sum(),total_energy,rtol=1e-12,atol=0)
    trajectory = json.loads(TRAJECTORY.read_text())
    epoch = np.datetime64("2026-05-30T18:06:20", "ns").astype("int64")/1e9
    model_seconds = np.asarray(trajectory["model"]["time_unix"])-epoch
    model_altitude = np.asarray(trajectory["model"]["alt_m"])/1000
    if not (np.all(np.diff(model_seconds)>0) and np.all(np.diff(model_altitude)<0)):
        raise ValueError("Trajectory must be monotonic in time and altitude")
    centers = (time_edges[:-1]+time_edges[1:])/2
    curve_altitude = np.interp(centers, model_seconds, model_altitude, left=np.nan, right=np.nan)
    altitude_ticks = []
    for altitude in [60,50,40,30,20]:
        tick_time = float(np.interp(altitude, model_altitude[::-1], model_seconds[::-1]))
        if 1.5 <= tick_time <= 3.4:
            np.testing.assert_allclose(np.interp(tick_time,model_seconds,model_altitude),altitude,atol=1e-8)
            altitude_ticks.append([tick_time,altitude])
    polygons, values, energy_polygons, energy_values = [], [], [], []
    for r,c in zip(*np.where(counts > 0)):
        xx = [xe[c],xe[c+1],xe[c+1],xe[c]]
        yy = [ye[r],ye[r],ye[r+1],ye[r+1]]
        llon,llat = projection(xx,yy,inverse=True)
        polygons.append(np.column_stack([llon,llat]).tolist())
        values.append(int(counts[r,c]))
        if energy_map[r,c] > 0:
            energy_polygons.append(polygons[-1])
            energy_values.append(float(energy_map[r,c]*1e15))
    coast = []
    reader = shp.Reader(shp.natural_earth("10m","physical","coastline"))
    from shapely.geometry import box
    window = box(west,south,east,north)
    for record in reader.geometries():
        if record.intersects(window):
            clipped = record.intersection(window)
            parts = list(clipped.geoms) if hasattr(clipped,"geoms") else [clipped]
            coast.extend(np.asarray(g.coords).tolist() for g in parts if g.geom_type == "LineString")
    with h5py.File(DATA / "glm_flash_density_180600_180700.h5","w") as out:
        out.attrs["quantity"] = "Distinct flashes per native GLM pixel in 60 seconds"
        out.attrs["window_utc"] = "2026-05-30T18:06:00Z/2026-05-30T18:07:00Z"
        out.attrs["pixel_geometry"] = "Reconstructed from local event-center spacing; no altitude parallax correction"
        out.attrs["sources"] = json.dumps([p.name for p in sources])
        out.create_dataset("flash_count",data=counts)
        out.create_dataset("pixel_x_edges_m",data=xe)
        out.create_dataset("pixel_y_edges_m",data=ye)
        out.create_dataset("cloud_reflectance",data=cloud,compression="gzip")
    with h5py.File(DATA / "glm_bolide_event_energy.h5","w") as out:
        out.attrs["quantity"] = "GLM Level-2 event radiant energy at the instrument, in the GLM passband"
        out.attrs["selection_window_utc"] = "2026-05-30T18:06:20Z/2026-05-30T18:06:24Z"
        out.attrs["spatial_selection"] = "41.5 < latitude < 43.5; -71.5 < longitude < -69.5"
        out.attrs["pixel_geometry"] = "Reconstructed from local event-center spacing; no altitude parallax correction"
        out.attrs["sources"] = json.dumps([p.name for p in sources])
        out.attrs["total_received_energy_J"] = total_energy
        out.attrs["light_curve_bin_width_s"] = 0.01
        out.attrs["zero_bin_meaning"] = "No selected Level-2 events in bin; not proof of zero physical emission"
        out.attrs["trajectory_source"] = str(TRAJECTORY.relative_to(ROOT))
        out.attrs["trajectory_sha256"] = hashlib.sha256(TRAJECTORY.read_bytes()).hexdigest()
        out.attrs["trajectory_fit_version"] = str(trajectory["trajectory_fit_version"])
        out.attrs["altitude_interpretation"] = "Continued fitted trajectory; below nominal 33.61 km radar terminus not a measured bolide altitude"
        for key,val,unit in [
            ("event_energy",event_energy[bolide],"J"),
            ("event_seconds_after_180620",event_seconds[bolide],"s"),
            ("event_id",np.asarray(event_ids)[bolide],"1"),
            ("event_latitude",np.asarray(lat)[bolide],"degrees_north"),
            ("event_longitude",np.asarray(lon)[bolide],"degrees_east"),
            ("pixel_energy",energy_map,"J"),
            ("pixel_x_edges",xe,"m"),("pixel_y_edges",ye,"m"),
            ("time_bin_edges",time_edges,"s after 2026-05-30T18:06:20 UTC"),
            ("time_bin_modeled_altitude",curve_altitude,"km"),
            ("time_bin_energy",light_curve,"J per 10 ms bin")]:
            out.create_dataset(key,data=val).attrs["units"] = unit
    scene = {"bounds": BOUNDS, "polygons": polygons, "values": values, "coast": coast,
             "maximum_flashes_per_minute": int(counts.max()), "source_files": [p.name for p in sources],
             "energy_polygons": energy_polygons,"energy_fj": energy_values,
             "altitude_ticks": altitude_ticks,
             "light_curve": np.column_stack([(time_edges[:-1]+time_edges[1:])/2,light_curve*1e15]).tolist()}
    (DATA / "rustplotlib_scene.json").write_text(json.dumps(scene))
    print("Maximum flashes/min:",counts.max(),"illuminated pixels:",len(values))
    print("Energy selection:",bolide.sum(),"events; total:",total_energy,"J")
    print("Pixel energy range:",min(energy_values),max(energy_values),"fJ")
    print("Maximum 10-ms bin:",light_curve.max()*1e15,"fJ")


if __name__ == "__main__":
    main()
