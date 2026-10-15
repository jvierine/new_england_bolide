"""Numerical checks and diagram coordinates for the GLM geometry memo."""
import hashlib
import json

import h5py
import numpy as np
import xarray as xr
from pyproj import Geod
from scipy.optimize import brentq

from compare_glm_trajectory import A, B, ECEF, GEODETIC, angles, reported_point, satellite_position
from prepare_goes_glm_flash_density import DATA, TRAJECTORY, apparent_trajectory


def main():
    fit = json.loads(TRAJECTORY.read_text())
    model = fit["model"]
    epoch = np.datetime64("2026-05-30T18:06:20", "ns").astype("int64")/1e9
    mt = np.asarray(model["time_unix"])-epoch
    mh = np.asarray(model["alt_m"])
    terminus = float(np.interp(33610.,mh[::-1],mt[::-1]))
    cutoff = float(np.interp(30000.,mh[::-1],mt[::-1]))
    times = np.array([0.26,1.7,2.,2.28,terminus,cutoff])
    samples = {k:np.interp(times,mt,model[k]) for k in ["lon_deg","lat_deg","alt_m"]}
    with xr.open_dataset(next(DATA.glob("OR_GLM*s20261501806200*.nc"))) as glm:
        sat_lon = float(glm.nominal_satellite_subpoint_lon)
        sat_height = float(glm.nominal_satellite_height)*1000
    app = apparent_trajectory(samples,times,sat_lon,sat_height)
    xyz = np.column_stack(ECEF.transform(samples["lon_deg"],samples["lat_deg"],samples["alt_m"]))
    projected = reported_point(app[:,0],app[:,1])
    np.testing.assert_allclose(angles(xyz,sat_lon,sat_height),angles(projected,sat_lon,sat_height),atol=1e-12,rtol=0)
    bearing,_,distance = Geod(ellps="GRS80").inv(samples["lon_deg"],samples["lat_deg"],app[:,0],app[:,1])
    satellite = satellite_position(sat_lon,sat_height)
    lon,lat = np.radians(samples["lon_deg"]),np.radians(samples["lat_deg"])
    east = np.column_stack([-np.sin(lon),np.cos(lon),np.zeros(len(lon))])
    north = np.column_stack([-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)])
    up = np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    sightline = satellite-xyz
    enu = np.column_stack([np.sum(sightline*axis,axis=1) for axis in [east,north,up]])
    azimuth = np.degrees(np.arctan2(enu[:,0],enu[:,1]))%360
    elevation = np.degrees(np.arctan2(enu[:,2],np.hypot(enu[:,0],enu[:,1])))
    rows = np.column_stack([times,samples["alt_m"]/1000,samples["lat_deg"],samples["lon_deg"],app[:,1],app[:,0],distance/1000,azimuth,elevation])
    # Local, equal-scale section through satellite, reference source and origin.
    i=3
    ground=np.asarray(ECEF.transform(samples["lon_deg"][i],samples["lat_deg"][i],0.))
    radial=ground/np.linalg.norm(ground)
    toward_sat=satellite-ground
    toward_sat-=radial*np.dot(toward_sat,radial)
    horizontal=-toward_sat/np.linalg.norm(toward_sat)
    def section(points):
        return np.column_stack([(points-ground)@horizontal,(points-ground)@radial])/1000
    x=np.linspace(-30.,80.,111)*1000
    direction=ground[None,:]+x[:,None]*horizontal
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    earth_radius=1/np.sqrt(np.sum(direction**2/np.array([A*A,A*A,B*B]),axis=1))
    le_radius=1/np.sqrt(np.sum(direction**2/np.array([(A+14000.)**2,(A+14000.)**2,6362755.**2]),axis=1))
    earth=section(direction*earth_radius[:,None])
    ellipse=section(direction*le_radius[:,None])
    p,q=section(xyz[[i]])[0],section(projected[[i]])[0]
    ray=np.vstack([p+(q-p)*(p[1]-65.)/(p[1]-q[1]),q])
    with h5py.File(DATA/"glm_geolocation_geometry.h5","w") as out:
        out.attrs["columns"]="seconds_after_180620, altitude_km, true_lat_deg, true_lon_deg, apparent_lat_deg, apparent_lon_deg, map_displacement_km, source_to_satellite_azimuth_deg, source_to_satellite_elevation_deg"
        out.attrs["trajectory_sha256"]=hashlib.sha256(TRAJECTORY.read_bytes()).hexdigest()
        out.attrs["satellite_longitude_deg"]=sat_lon
        out.attrs["satellite_height_m"]=sat_height
        out.attrs["lightning_ellipsoid_axes_m"]=[A+14000.,6362755.]
        out.create_dataset("samples",data=rows)
        for name,value in {"earth_section_km":earth,"lightning_ellipsoid_section_km":ellipse,
                           "reference_source_section_km":p,"reference_apparent_section_km":q,
                           "reference_ray_section_km":ray}.items():
            out.create_dataset(name,data=value)
    print("Satellite longitude / altitude:",sat_lon,sat_height/1000)
    print("Columns: time, height, true lat/lon, apparent lat/lon, displacement, satellite az/el")
    for row in rows: print(" ".join(f"{value:.6f}" for value in row))
    print("Local source / apparent / ray:",p,q,ray)
    print("Ground and lightning-ellipsoid cross sections (km):")
    for a,b in zip(earth[::10],ellipse[::10]): print(a,b)
    _,_,le_height=GEODETIC.transform(*projected[i])
    print("Reference lightning ellipsoid geodetic height:",le_height/1000)

    heights = np.array([90000.,50000.,33610.])
    sampled = {k:np.interp(heights,mh[::-1],np.asarray(model[k])[::-1])
               for k in ["lon_deg","lat_deg","alt_m"]}
    apparent = apparent_trajectory(sampled,np.zeros(len(heights)),sat_lon,sat_height)[:,:2]
    source = np.column_stack(ECEF.transform(sampled["lon_deg"],sampled["lat_deg"],heights))
    # Independent scalar root solving checks the analytic quadratic projection.
    axes = np.array([A+14000.,A+14000.,6362755.])
    for p,expected in zip(source,apparent):
        ray = (p-satellite)/np.linalg.norm(p-satellite)
        travel = brentq(lambda u: np.sum(((p+u*ray)/axes)**2)-1,0.,500000.,xtol=1e-6)
        q = p+travel*ray
        observed = reported_point(expected[None,0],expected[None,1])[0]
        np.testing.assert_allclose(q,observed,atol=1e-5,rtol=0)
    track_heights = np.linspace(90000.,33610.,250)
    track = {k:np.interp(track_heights,mh[::-1],np.asarray(model[k])[::-1])
             for k in ["lon_deg","lat_deg","alt_m"]}
    expected_track = apparent_trajectory(track,np.zeros(250),sat_lon,sat_height)[:,:2]
    pre=np.asarray(fit["fragment_impact"]["nominal"]["pre_break_llh"])
    full=np.vstack([pre[pre[:,2]>mh.max()],np.column_stack([model["lat_deg"],model["lon_deg"],mh])])
    full=full[np.argsort(full[:,2])]
    high=np.linspace(300000.,90000.,150)
    extension=np.column_stack([np.interp(high,full[:,2],full[:,1]),np.interp(high,full[:,2],full[:,0])])
    true_track=np.column_stack([track["lon_deg"],track["lat_deg"]])
    true_points=np.column_stack([sampled["lon_deg"],sampled["lat_deg"]])
    with h5py.File(DATA/"glm_geolocation_geometry.h5","a") as out:
        out.create_dataset("paired_height_km",data=heights/1000)
        out.create_dataset("paired_ground_lonlat_deg",data=true_points)
        out.create_dataset("paired_glm_lonlat_deg",data=apparent)
        out.attrs["independent_brentq_position_tolerance_m"]=1e-5
    scene={"ground":true_track.tolist(),"expected":expected_track.tolist(),
           "extension":extension.tolist(),"ground_points":true_points.tolist(),
           "expected_points":apparent.tolist(),"heights_km":(heights/1000).tolist(),
           "coast":json.loads((DATA/"rustplotlib_scene.json").read_text())["coast"]}
    (DATA/"glm_projection_explanation_scene.json").write_text(json.dumps(scene))
    print("Independent ray--ellipsoid root check passed (10 micrometres).")


if __name__=="__main__":
    main()
