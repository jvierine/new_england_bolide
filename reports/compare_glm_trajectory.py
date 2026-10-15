"""Compare native GLM viewing directions with the time-matched radar trajectory.

Lightning-ellipsoid conversion follows glmtools.io.lightning_ellipse and
Bezooijen et al., doi:10.1117/12.2242141. Nominal satellite location and a
post-2018 14/6-km lightning ellipsoid are assumptions, not fitted quantities.
"""
from pathlib import Path
import json
import hashlib

import h5py
import numpy as np
from pyproj import Transformer, Proj, Geod
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "products/goes19_glm_20260530"
FIT = ROOT / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json"
A, B = 6378137., 6356752.31414
ECEF = Transformer.from_crs("EPSG:4979","EPSG:4978",always_xy=True)
GEODETIC = Transformer.from_crs("EPSG:4978","EPSG:4979",always_xy=True)


def reported_point(lon,lat,equatorial_height=14000.):
    # GLM reports latitude referenced to GRS80 geocentric latitude, with
    # distance along that radial direction set by the lightning ellipsoid.
    longitude, latitude = np.radians(lon),np.radians(lat)
    gc = np.arctan((B/A)**2*np.tan(latitude))
    ae,be = A+equatorial_height,6362755.
    radius = ae*be/np.sqrt((be*np.cos(gc))**2+(ae*np.sin(gc))**2)
    return np.column_stack([radius*np.cos(gc)*np.cos(longitude),
                            radius*np.cos(gc)*np.sin(longitude),radius*np.sin(gc)])


def angles(points,sat_lon,sat_height):
    longitude = np.radians(sat_lon)
    x = points[:,0]*np.cos(longitude)+points[:,1]*np.sin(longitude)
    y = -points[:,0]*np.sin(longitude)+points[:,1]*np.cos(longitude)
    z = points[:,2]
    distance = np.sqrt((A+sat_height-x)**2+y*y+z*z)
    return np.arcsin(y/distance),np.arctan2(z,A+sat_height-x)


def satellite_position(sat_lon,sat_height):
    longitude = np.radians(sat_lon)
    return np.array([(A+sat_height)*np.cos(longitude),(A+sat_height)*np.sin(longitude),0.])


def intersect_model_height(observed,sat_lon,sat_height,target_height):
    satellite = satellite_position(sat_lon,sat_height)
    ray = observed-satellite
    distance = np.linalg.norm(ray,axis=1)
    ray /= distance[:,None]
    for _ in range(7):
        xyz = satellite+distance[:,None]*ray
        lon,lat,height = GEODETIC.transform(*xyz.T)
        nearby = satellite+(distance+10.)[:,None]*ray
        _,_,height_nearby = GEODETIC.transform(*nearby.T)
        derivative = (height_nearby-height)/10.
        distance -= (height-target_height)/derivative
    xyz = satellite+distance[:,None]*ray
    lon,lat,height = GEODETIC.transform(*xyz.T)
    np.testing.assert_allclose(height,target_height,atol=0.01,rtol=0)
    return np.asarray(lon),np.asarray(lat),np.asarray(height)


def cell_edges(values):
    ordered = np.sort(values)
    centers = np.array([np.mean(v) for v in np.split(ordered,np.where(np.diff(ordered)>40e-6)[0]+1)])
    return centers,np.r_[centers[0]-(centers[1]-centers[0])/2,
                         (centers[:-1]+centers[1:])/2,centers[-1]+(centers[-1]-centers[-2])/2]


def cloud_top_samples(observed,sat_lon,sat_height):
    files = list(DATA.glob("OR_ABI-L2-ACHAC*.nc"))
    if not files:
        return np.full(len(observed),np.nan)
    with xr.open_dataset(files[0]) as cloud:
        meta = cloud.goes_imager_projection.attrs
        projection = Proj(proj="geos",h=meta["perspective_point_height"],
                          lon_0=meta["longitude_of_projection_origin"],
                          a=meta["semi_major_axis"],b=meta["semi_minor_axis"],sweep="x")
        longitude = np.radians(sat_lon)
        satellite = np.array([(A+sat_height)*np.cos(longitude),
                              (A+sat_height)*np.sin(longitude),0.])
        ray = observed-satellite
        ray /= np.linalg.norm(ray,axis=1)[:,None]
        height = np.zeros(len(observed))
        quality = cloud.DQF.values
        ht = cloud.HT.values
        for _ in range(5):
            q = 1/np.column_stack([(A+height)**2,(A+height)**2,(B+height)**2])
            aa = np.sum(ray*ray*q,axis=1)
            bb = 2*np.sum(satellite*ray*q,axis=1)
            cc = np.sum(satellite*satellite*q,axis=1)-1
            distance = (-bb-np.sqrt(bb*bb-4*aa*cc))/(2*aa)
            point = satellite+distance[:,None]*ray
            lon,lat,_ = GEODETIC.transform(*point.T)
            gx,gy = projection(lon,lat)
            gx,gy = gx/meta["perspective_point_height"],gy/meta["perspective_point_height"]
            ix = np.abs(gx[:,None]-cloud.x.values).argmin(axis=1)
            iy = np.abs(gy[:,None]-cloud.y.values).argmin(axis=1)
            sampled = ht[iy,ix]
            valid = np.isfinite(sampled)&np.isin(quality[iy,ix],[0,1])
            height = np.where(valid,sampled,0.)
        return np.where(valid,height,np.nan)


def main():
    fit = json.loads(FIT.read_text())
    epoch = np.datetime64("2026-05-30T18:06:20","ns").astype("int64")/1e9
    model = fit["model"]
    mt = np.asarray(model["time_unix"])-epoch
    mh = np.asarray(model["alt_m"])
    terminus = float(np.interp(33610.,mh[::-1],mt[::-1]))
    with h5py.File(DATA/"glm_bolide_event_energy.h5") as h:
        t,e,lat,lon = (h[name][()] for name in ["event_seconds_after_180620","event_energy","event_latitude","event_longitude"])
    with xr.open_dataset(next(DATA.glob("OR_GLM*s20261501806200*.nc"))) as glm:
        sat_lon = float(glm.nominal_satellite_subpoint_lon)
        sat_height = float(glm.nominal_satellite_height)*1000
    observed = reported_point(lon,lat)
    cloud_height = cloud_top_samples(observed,sat_lon,sat_height)
    ox,oy = angles(observed,sat_lon,sat_height)
    mlon,mlat,malt = [np.interp(t,mt,model[k]) for k in ["lon_deg","lat_deg","alt_m"]]
    position = np.column_stack(ECEF.transform(mlon,mlat,malt))
    mapped_lon,mapped_lat,mapped_height = intersect_model_height(observed,sat_lon,sat_height,malt)
    check_angles = angles(np.column_stack(ECEF.transform(mapped_lon,mapped_lat,mapped_height)),sat_lon,sat_height)
    # Allow centimetre-scale geodetic/ECEF round-trip error, far below GLM resolution.
    np.testing.assert_allclose(check_angles[0],ox,atol=1e-9,rtol=0)
    np.testing.assert_allclose(check_angles[1],oy,atol=1e-9,rtol=0)
    _,_,separation = Geod(ellps="GRS80").inv(mapped_lon,mapped_lat,mlon,mlat)
    tx,ty = angles(position,sat_lon,sat_height)
    dx,dy = (ox-tx)*1e6,(oy-ty)*1e6
    xc,xe = cell_edges(ox)
    yc,ye = cell_edges(oy)
    col = np.abs(ox[:,None]-xc).argmin(axis=1)
    row = np.abs(oy[:,None]-yc).argmin(axis=1)
    inside = (tx>=xe[col])&(tx<=xe[col+1])&(ty>=ye[row])&(ty<=ye[row+1])
    phase_bounds = [("Compact early emission",1.65,2.05),
                    ("Bright emission before radar terminus",2.15,terminus),
                    ("Emission after radar terminus",terminus,3.1)]
    summaries = []
    term_position = np.column_stack(ECEF.transform(
        [np.interp(terminus,mt,model["lon_deg"])],
        [np.interp(terminus,mt,model["lat_deg"])],[33610.]))
    xref,yref = angles(term_position,sat_lon,sat_height)
    panels = []
    for name,start,end in phase_bounds:
        select = (t>=start)&(t<end)
        w = e[select]
        energy = np.zeros((len(yc),len(xc)))
        np.add.at(energy,(row[select],col[select]),w)
        summary = {"phase":name,"time_bounds_s":[start,end],"events":int(select.sum()),
                   "energy_pj":float(w.sum()*1e12),
                   "centroid_offset_urad":[float(np.average(dx[select],weights=w)),float(np.average(dy[select],weights=w))],
                   "energy_in_model_pixel_fraction":float(w[inside[select]].sum()/w.sum()),
                   "illuminated_pixels":int((energy>0).sum()),
                   "energy_weighted_rms_offset_urad":float(np.sqrt(np.average(dx[select]**2+dy[select]**2,weights=w)))}
        summary["height_corrected_centroid_lonlat"] = [float(np.average(mapped_lon[select],weights=w)),float(np.average(mapped_lat[select],weights=w))]
        summary["height_corrected_weighted_rms_distance_to_model_km"] = float(np.sqrt(np.average(separation[select]**2,weights=w))/1000)
        valid_cloud = select&np.isfinite(cloud_height)
        summary["cloud_top_valid_energy_fraction"] = float(e[valid_cloud].sum()/w.sum())
        summary["energy_weighted_cloud_top_height_km"] = (
            float(np.average(cloud_height[valid_cloud],weights=e[valid_cloud])/1000)
            if valid_cloud.any() else None)
        summaries.append(summary)
        polys,values = [],[]
        for r,c in zip(*np.where(energy>0)):
            xx = (np.array([xe[c],xe[c+1],xe[c+1],xe[c]])-xref[0])*1e6
            yy = (np.array([ye[r],ye[r],ye[r+1],ye[r+1]])-yref[0])*1e6
            polys.append(np.column_stack([xx,yy]).tolist())
            values.append(float(energy[r,c]*1e12))
        track_t = np.linspace(start,end,100)
        track_xyz = np.column_stack(ECEF.transform(*[np.interp(track_t,mt,model[k]) for k in ["lon_deg","lat_deg","alt_m"]]))
        track_x,track_y = angles(track_xyz,sat_lon,sat_height)
        panels.append({"polygons":polys,"values":values,
                       "track":np.column_stack([(track_x-xref[0])*1e6,(track_y-yref[0])*1e6]).tolist()})
    time_edges = np.linspace(1.5,3.4,191)
    binned=[]
    for lo,hi in zip(time_edges[:-1],time_edges[1:]):
        sel = (t>=lo)&(t<hi)
        if sel.any():
            binned.append([(lo+hi)/2,float(np.average(dx[sel],weights=e[sel])),
                           float(np.average(dy[sel],weights=e[sel])),float(e[sel&inside].sum()/e[sel].sum())])
    with h5py.File(DATA/"glm_trajectory_consistency.h5","w") as h:
        h.attrs["trajectory_sha256"] = hashlib.sha256(FIT.read_bytes()).hexdigest()
        h.attrs["trajectory_source"] = str(FIT.relative_to(ROOT))
        h.attrs["trajectory_version"] = str(fit["trajectory_fit_version"])
        h.attrs["glm_source_files"] = json.dumps([p.name for p in sorted(DATA.glob("OR_GLM*s20261501806*.nc"))])
        h.attrs["cloud_top_source_files"] = json.dumps([p.name for p in DATA.glob("OR_ABI-L2-ACHAC*.nc")])
        h.attrs["height_mapping_assumption"] = "Every event originates at the time-interpolated trajectory geodetic altitude; invalid for cloud-scattered light"
        h.attrs["nominal_satellite_longitude_deg"] = sat_lon
        h.attrs["nominal_satellite_height_m"] = sat_height
        h.attrs["lightning_ellipsoid_axes_m"] = [A+14000.,6362755.]
        h.attrs["pixel_geometry"] = "Approximate local native grid inferred from event centers"
        h.attrs["radar_terminus_seconds_after_180620"] = terminus
        for key,value in {"event_seconds_after_180620":t,"event_energy_J":e,
                          "observed_beta_rad":ox,"observed_alpha_rad":oy,
                          "modeled_beta_rad":tx,"modeled_alpha_rad":ty,
                          "within_model_pixel":inside,"pixel_beta_edges_rad":xe,
                          "cloud_top_height_m":cloud_height,
                          "height_corrected_longitude_deg":mapped_lon,
                          "height_corrected_latitude_deg":mapped_lat,
                          "assumed_source_height_m":mapped_height,
                          "height_corrected_distance_to_model_m":separation,
                          "pixel_alpha_edges_rad":ye,"binned_residuals":binned}.items():
            h.create_dataset(key,data=value)
    summary = {"radar_terminus_seconds_after_180620":terminus,
               "native_pixel_pitch_urad":[float(np.median(np.diff(xc))*1e6),float(np.median(np.diff(yc))*1e6)],
               "phases":summaries,
               "assumptions":"Nominal satellite location; 14/6-km lightning ellipsoid; inferred pixel outlines; no fit-covariance or GLM navigation uncertainty propagation"}
    with h5py.File(DATA/"glm_bolide_event_energy.h5") as h:
        bin_edges = h["time_bin_edges"][()]
        bin_energy = h["time_bin_energy"][()]
    peak = int(np.argmax(bin_energy))
    peak_time = float((bin_edges[peak]+bin_edges[peak+1])/2)
    summary["brightest_10ms_bin_center_s"] = peak_time
    summary["brightest_bin_after_radar_terminus_s"] = peak_time-terminus
    (DATA/"glm_trajectory_consistency.json").write_text(json.dumps(summary,indent=2)+"\n")
    lightcurve = np.column_stack([(bin_edges[:-1]+bin_edges[1:])/2,bin_energy*1e12]).tolist()
    (DATA/"glm_trajectory_comparison_scene.json").write_text(json.dumps({"panels":panels,"binned":binned,"terminus":terminus,"lightcurve":lightcurve}))
    selection = t<=2.05
    track_sel = (mt>=1.5)&(mt<=3.1)
    footprint_polys=[]
    for r,c in set(zip(row[selection],col[selection])):
        cell_sel = selection&(row==r)&(col==c)
        at = float(np.average(t[cell_sel],weights=e[cell_sel]))
        hgt = np.interp(at,mt,mh)
        beta=np.array([xe[c],xe[c+1],xe[c+1],xe[c]])
        alpha=np.array([ye[r],ye[r],ye[r+1],ye[r+1]])
        local=np.column_stack([-np.cos(alpha)*np.cos(beta),np.sin(beta),np.sin(alpha)*np.cos(beta)])
        ll=np.radians(sat_lon)
        rays=np.column_stack([local[:,0]*np.cos(ll)-local[:,1]*np.sin(ll),
                              local[:,0]*np.sin(ll)+local[:,1]*np.cos(ll),local[:,2]])
        fake_points=satellite_position(sat_lon,sat_height)+rays*37000000.
        flo,fla,_=intersect_model_height(fake_points,sat_lon,sat_height,np.full(4,hgt))
        footprint_polys.append(np.column_stack([flo,fla]).tolist())
    map_scene={"points":np.column_stack([mapped_lon,mapped_lat,t]).tolist(),
               "early_footprints":footprint_polys,
               "track":np.column_stack([np.asarray(model["lon_deg"])[track_sel],np.asarray(model["lat_deg"])[track_sel]]).tolist(),
               "terminus_lonlat":[float(np.interp(terminus,mt,model["lon_deg"])),float(np.interp(terminus,mt,model["lat_deg"]))]}
    map_scene["coast"] = json.loads((DATA/"rustplotlib_scene.json").read_text())["coast"]
    (DATA/"glm_height_corrected_map_scene.json").write_text(json.dumps(map_scene))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
