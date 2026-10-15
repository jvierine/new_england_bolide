"""Prepare Figure 2 in HDF5 and render via reports/rust_geometry (rustplotlib).

Run: conda run -n base python reports/plot_article_observing_geometry.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from cartopy.io import shapereader
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json"
DARK_FLIGHT = ROOT / "data/marc_fries/cape_cod_bay_dark_flight_20260625.json"
STATIONS = {
    "Beverly": (42.551324, -70.901134), "Clinton": (42.419328, -71.677843),
    "Pomfret": (43.717156, -72.434666), "Siena": (42.719476, -73.752195),
    "Amherst": (42.828043, -71.573002), "TX": (42.611495, -71.484666),
    "Mechanicville": (42.9029, -73.6873), "KBOX": (41.9558, -71.1369),
    "KENX": (42.5864, -74.0641), "KOKX": (40.8655, -72.8639),
    "TBOS": (42.1581, -70.9333),
}

def main():
    fit = json.loads(FIT.read_text())
    terminus = fit["fragment_impact"]["display_trail_min_alt_km"]
    model = fit["model"]
    pre = np.asarray(fit["fragment_impact"]["nominal"]["pre_break_llh"])
    llh = np.column_stack([model["lat_deg"], model["lon_deg"], model["alt_m"]])
    path = np.vstack([pre[pre[:, 2] > max(model["alt_m"])], llh])
    path = path[np.argsort(path[:, 2])]
    assert path[0, 2] <= terminus * 1000 < 300000 <= path[-1, 2]
    def segment(upper):
        alt = np.linspace(upper * 1000, terminus * 1000, 320)
        return np.column_stack([np.interp(alt, path[:, 2], path[:, 1]),
                                np.interp(alt, path[:, 2], path[:, 0])])
    rows = json.loads(DARK_FLIGHT.read_text())["Masses"]
    detections = [[r.get("obs_blip_lon", r.get("blip_lon")),
                   r.get("obs_blip_lat", r.get("blip_lat"))] for r in rows
                  if r.get("obs_blip_alt_m") is not None]
    assert len(detections) == 6 and np.isfinite(detections).all()
    output = ROOT / "products/observing_geometry"
    output.mkdir(parents=True, exist_ok=True)
    data = output / "observing_geometry.h5"
    bounds = [-74.35, -69.95, 40.55, 43.9]
    clip = box(bounds[0], bounds[2], bounds[1], bounds[3])
    with h5py.File(data, "w") as h:
        h.attrs["trajectory_source"] = str(FIT.relative_to(ROOT))
        h.attrs["weather_measurement_source"] = str(DARK_FLIGHT.relative_to(ROOT))
        h.attrs["trajectory_fit_version"] = fit["trajectory_fit_version"]
        h.attrs["observed_segment_start_altitude_km"] = 90.
        h.attrs["terminus_altitude_km"] = terminus
        h.attrs["camera_coordinate"] = "Mechanicville town coordinate; not surveyed"
        h["bounds"] = bounds
        h["trajectory"] = segment(300.)
        h["observed_trajectory"] = segment(90.)
        h["weather_detections"] = np.asarray(detections)
        for name, (lat, lon) in STATIONS.items():
            h[f"stations/{name}"] = [lon, lat]
        for group, category, name in [("land", "physical", "land"),
                ("coast", "physical", "coastline"),
                ("states", "cultural", "admin_1_states_provinces_lines")]:
            g = h.create_group(group)
            shapes = shapereader.Reader(shapereader.natural_earth("10m", category, name))
            def save(geom):
                if geom.is_empty:
                    return
                if hasattr(geom, "geoms"):
                    for part in geom.geoms:
                        save(part)
                elif geom.geom_type == "Polygon":
                    g[str(len(g))] = np.asarray(geom.exterior.coords)[:, :2]
                elif geom.geom_type in ("LineString", "LinearRing"):
                    g[str(len(g))] = np.asarray(geom.coords)[:, :2]
            for geom in shapes.geometries():
                if geom.intersects(clip):
                    save(geom.intersection(clip))
    env = os.environ.copy()
    if (Path(sys.prefix) / "include/hdf5.h").exists():
        env.setdefault("HDF5_DIR", sys.prefix)
        env["DYLD_FALLBACK_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    subprocess.run(["cargo", "run", "--release", "--manifest-path",
        str(ROOT / "reports/rust_geometry/Cargo.toml"), "--", str(data),
        str(ROOT / "reports/figures/appendix_correlated_fit/article_observing_geometry")], check=True, env=env)
    print(f"Rendered archived trajectory and {len(detections)} weather-radar centroids")

if __name__ == "__main__":
    main()
