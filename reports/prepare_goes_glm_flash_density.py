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
from pyproj import Proj, Transformer
from scipy.ndimage import map_coordinates
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "products/goes19_glm_20260530"
BOUNDS = [-73.2, -68.3, 40.3, 44.2]
TRAJECTORY = ROOT / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json"


def apparent_trajectory(model, seconds, sat_lon, sat_height):
    """Project the 3-D trajectory onto GLM's assumed lightning ellipsoid."""
    from compare_glm_trajectory import A, B, reported_point, satellite_position, angles
    xyz = np.column_stack(Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        model["lon_deg"], model["lat_deg"], model["alt_m"]))
    satellite = satellite_position(sat_lon, sat_height)
    ray = xyz-satellite
    q = 1/np.array([(A+14000.)**2, (A+14000.)**2, 6362755.**2])
    aa = np.sum(ray*ray*q, axis=1)
    bb = 2*np.sum(satellite*ray*q, axis=1)
    cc = np.sum(satellite*satellite*q)-1
    distance = (-bb-np.sqrt(bb*bb-4*aa*cc))/(2*aa)
    point = satellite+distance[:,None]*ray
    lon = np.degrees(np.arctan2(point[:,1], point[:,0]))
    gc = np.arctan2(point[:,2], np.hypot(point[:,0], point[:,1]))
    lat = np.degrees(np.arctan(np.tan(gc)/(B/A)**2))
    np.testing.assert_allclose(angles(reported_point(lon,lat),sat_lon,sat_height),
                               angles(xyz,sat_lon,sat_height),atol=1e-12,rtol=0)
    return np.column_stack([lon,lat,seconds])


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
    abi = xr.open_dataset(next(DATA.glob("OR_ABI-L2-CMIPC*.nc")))
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
    with xr.open_dataset(sources[1]) as glm:
        sat_lon = float(glm.nominal_satellite_subpoint_lon)
        sat_height = float(glm.nominal_satellite_height)*1000
    apparent = apparent_trajectory(trajectory["model"],model_seconds,sat_lon,sat_height)
    terminus_time = float(np.interp(33.61,model_altitude[::-1],model_seconds[::-1]))
    cutoff_time = float(np.interp(30.,model_altitude[::-1],model_seconds[::-1]))
    event_altitude = np.interp(event_seconds,model_seconds,model_altitude,left=np.nan,right=np.nan)
    paper_selection = bolide & (event_altitude >= 30.)
    paper_energy_map = np.zeros_like(energy_map)
    np.add.at(paper_energy_map,(row[paper_selection],col[paper_selection]),event_energy[paper_selection])
    np.testing.assert_allclose(paper_energy_map.sum(),event_energy[paper_selection].sum(),rtol=1e-12,atol=0)
    assert np.all(event_seconds[paper_selection] <= cutoff_time)
    assert np.all(paper_energy_map <= energy_map)
    apparent_terminus = [float(np.interp(terminus_time,model_seconds,apparent[:,i])) for i in [0,1]]
    ground = np.column_stack([trajectory["model"]["lon_deg"],trajectory["model"]["lat_deg"]])
    ground_terminus = [float(np.interp(terminus_time,model_seconds,ground[:,i])) for i in [0,1]]
    track_selection = (model_seconds>=0)&(model_seconds<terminus_time)
    mapping_heights = np.array([90.,50.,33.61])
    mapping_times = np.interp(mapping_heights,model_altitude[::-1],model_seconds[::-1])
    mapping_model = {key:np.interp(mapping_times,model_seconds,trajectory["model"][key])
                     for key in ["lon_deg","lat_deg","alt_m"]}
    mapping_ground = np.column_stack([mapping_model["lon_deg"],mapping_model["lat_deg"]])
    mapping_expected = apparent_trajectory(mapping_model,mapping_times,sat_lon,sat_height)[:,:2]
    centers = (time_edges[:-1]+time_edges[1:])/2
    curve_altitude = np.interp(centers, model_seconds, model_altitude, left=np.nan, right=np.nan)
    altitude_ticks = []
    for altitude in [60,50,40,30,20]:
        tick_time = float(np.interp(altitude, model_altitude[::-1], model_seconds[::-1]))
        if 1.5 <= tick_time <= 3.4:
            np.testing.assert_allclose(np.interp(tick_time,model_seconds,model_altitude),altitude,atol=1e-8)
            altitude_ticks.append([tick_time,altitude])
    polygons, values, energy_polygons, energy_values = [], [], [], []
    energy_pixel_indices = []
    for r,c in zip(*np.where(counts > 0)):
        xx = [xe[c],xe[c+1],xe[c+1],xe[c]]
        yy = [ye[r],ye[r],ye[r+1],ye[r+1]]
        llon,llat = projection(xx,yy,inverse=True)
        polygons.append(np.column_stack([llon,llat]).tolist())
        values.append(int(counts[r,c]))
        if energy_map[r,c] > 0:
            energy_polygons.append(polygons[-1])
            energy_values.append(float(energy_map[r,c]*1e15))
            energy_pixel_indices.append((int(r),int(c)))
    animation_dt = 0.002
    animation_start = 1.5
    animation_count = 950
    animation_edges = animation_start + np.arange(animation_count+1)*animation_dt
    animation_energy = np.zeros((animation_count,len(energy_polygons)),dtype="float64")
    pixel_lookup = {rc:i for i,rc in enumerate(energy_pixel_indices)}
    for t,e,r,c in zip(event_seconds[bolide],event_energy[bolide],row[bolide],col[bolide]):
        frame_index = int(np.searchsorted(animation_edges,t,side="right")-1)
        if not 0 <= frame_index < animation_count:
            raise ValueError("Selected event outside animation window")
        animation_energy[frame_index,pixel_lookup[(int(r),int(c))]] += e
    np.testing.assert_allclose(animation_energy.sum(),total_energy,rtol=1e-12,atol=0)
    coast = []
    reader = shp.Reader(shp.natural_earth("10m","physical","coastline"))
    from shapely.geometry import box
    window = box(west,south,east,north)
    for record in reader.geometries():
        if record.intersects(window):
            clipped = record.intersection(window)
            parts = list(clipped.geoms) if hasattr(clipped,"geoms") else [clipped]
            coast.extend(np.asarray(g.coords).tolist() for g in parts if g.geom_type == "LineString")
    borders = []
    reader = shp.Reader(shp.natural_earth("10m","cultural","admin_1_states_provinces_lines"))
    for geom in reader.geometries():
        if geom.intersects(window):
            clipped = geom.intersection(window)
            parts = list(clipped.geoms) if hasattr(clipped,"geoms") else [clipped]
            borders.extend(np.asarray(g.coords).tolist() for g in parts if g.geom_type == "LineString")
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
        out.attrs["apparent_trajectory_projection"] = "Satellite sightline intersected with 14/6-km GLM lightning ellipsoid; nominal satellite position"
        out.attrs["radar_terminus_seconds_after_180620"] = terminus_time
        out.attrs["figure1_minimum_modeled_altitude_km"] = 30.
        out.attrs["figure1_cutoff_seconds_after_180620"] = cutoff_time
        out.create_dataset("expected_glm_longitude_latitude_time",data=apparent)
        out.create_dataset("expected_glm_terminus_longitude_latitude",data=apparent_terminus)
        out.create_dataset("mapping_heights_km",data=mapping_heights)
        out.create_dataset("mapping_ground_longitude_latitude",data=mapping_ground)
        out.create_dataset("mapping_expected_glm_longitude_latitude",data=mapping_expected)
        for key,val,unit in [
            ("event_energy",event_energy[bolide],"J"),
            ("event_seconds_after_180620",event_seconds[bolide],"s"),
            ("event_id",np.asarray(event_ids)[bolide],"1"),
            ("event_latitude",np.asarray(lat)[bolide],"degrees_north"),
            ("event_longitude",np.asarray(lon)[bolide],"degrees_east"),
            ("pixel_energy",energy_map,"J"),
            ("figure1_event_selection",paper_selection[bolide],"1"),
            ("figure1_pixel_energy",paper_energy_map,"J"),
            ("pixel_x_edges",xe,"m"),("pixel_y_edges",ye,"m"),
            ("time_bin_edges",time_edges,"s after 2026-05-30T18:06:20 UTC"),
            ("time_bin_modeled_altitude",curve_altitude,"km"),
            ("animation_time_bin_edges",animation_edges,"s after 2026-05-30T18:06:20 UTC"),
            ("animation_pixel_energy",animation_energy,"J per pixel per 2 ms bin"),
            ("animation_pixel_row_col",np.asarray(energy_pixel_indices),"index"),
            ("time_bin_energy",light_curve,"J per 10 ms bin")]:
            out.create_dataset(key,data=val).attrs["units"] = unit
    scene = {"bounds": BOUNDS, "polygons": polygons, "values": values, "coast": coast,
             "borders": borders, "apparent_trajectory": apparent[track_selection,:2].tolist()+[apparent_terminus],
             "ground_trajectory": ground[track_selection].tolist()+[ground_terminus],
             "mapping_heights_km": mapping_heights.tolist(),
             "mapping_ground": mapping_ground.tolist(),"mapping_expected": mapping_expected.tolist(),
             "apparent_terminus": apparent_terminus,
             "terminus_time": terminus_time,
             "maximum_flashes_per_minute": int(counts.max()), "source_files": [p.name for p in sources],
             "energy_polygons": energy_polygons,"energy_fj": energy_values,
             "figure1_energy_fj": [float(paper_energy_map[r,c]*1e15) for r,c in energy_pixel_indices],
             "altitude_ticks": altitude_ticks,
             "animation_times": ((animation_edges[:-1]+animation_edges[1:])/2).tolist(),
             "animation_energy_pj": (animation_energy*1e12).tolist(),
             "light_curve": np.column_stack([(time_edges[:-1]+time_edges[1:])/2,light_curve*1e15]).tolist()}
    (DATA / "rustplotlib_scene.json").write_text(json.dumps(scene))
    print("Maximum flashes/min:",counts.max(),"illuminated pixels:",len(values))
    print("Energy selection:",bolide.sum(),"events; total:",total_energy,"J")
    print("Pixel energy range:",min(energy_values),max(energy_values),"fJ")
    print("Maximum 10-ms bin:",light_curve.max()*1e15,"fJ")
    print("Maximum animated pixel energy:",animation_energy.max()*1e12,"pJ per 2 ms")
    print("Figure 1 >=30 km:",paper_selection.sum(),"events; cutoff:",cutoff_time,
          "s after 18:06:20; energy:",paper_energy_map.sum()*1e12,"pJ")


if __name__ == "__main__":
    main()
