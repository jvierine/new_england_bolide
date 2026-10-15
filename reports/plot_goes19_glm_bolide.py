"""Download NOAA GOES-19 GLM data and plot the Cape Cod Bay optical transient."""
from pathlib import Path
import hashlib
import json

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "products/goes19_glm_20260530"
FIG = ROOT / "reports/figures/goes19_glm_bolide"
KEY = "GLM-L2-LCFA/2026/150/18/OR_GLM-L2-LCFA_G19_s20261501806200_e20261501806400_c20261501806422.nc"
URL = "https://noaa-goes19.s3.amazonaws.com/" + KEY


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    source = DATA / Path(KEY).name
    if not source.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        source.write_bytes(response.content)
    with xr.open_dataset(source) as ds:
        lat, lon = ds.event_lat.values, ds.event_lon.values
        time = ds.event_time_offset.values
        keep = ((lat > 41.5) & (lat < 43.5) & (lon > -71.5) & (lon < -69.5)
                & (time >= np.datetime64("2026-05-30T18:06:20"))
                & (time < np.datetime64("2026-05-30T18:06:24")))
        lat, lon, time = lat[keep], lon[keep], time[keep]
        energy = ds.event_energy.values[keep]
    seconds = (time - np.datetime64("2026-05-30T18:06:20")) / np.timedelta64(1, "s")
    with h5py.File(DATA / "selected_events.h5", "w") as out:
        out.attrs["source_url"] = URL
        out.attrs["geolocation"] = "Reported GLM Level 2 coordinates; no bolide-altitude parallax correction"
        for name, value in {"latitude_deg": lat, "longitude_deg": lon,
                            "seconds_after_180620_utc": seconds, "event_energy_J": energy}.items():
            out.create_dataset(name, data=value)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11})
    fig = plt.figure(figsize=(6, 5.2), layout="constrained")
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent([-71.35, -69.65, 41.4, 42.65])
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#e7f1f6")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f4f0")
    ax.coastlines("10m", linewidth=0.7)
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.4, edgecolor="0.5")
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.5)
    grid.top_labels = grid.right_labels = False
    # Draw later events last so the color follows the observed progression.
    order = np.argsort(seconds)
    points = ax.scatter(lon[order], lat[order], c=seconds[order], s=18,
                        cmap="viridis", vmin=1.6, vmax=3.2, edgecolors="none",
                        transform=ccrs.PlateCarree(), zorder=4, rasterized=True)
    ax.text(-70.4, 41.88, "Cape Cod Bay", ha="center", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2}, zorder=5)
    ax.text(-71.1, 42.4, "Massachusetts", fontsize=10)
    fig.colorbar(points, ax=ax, orientation="horizontal", pad=0.04,
                 label="Seconds after 18:06:20 UTC on 30 May 2026", shrink=0.94)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix(".pdf"), dpi=300)
    fig.savefig(FIG.with_suffix(".png"), dpi=300)
    summary = {"source_url": URL, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "event_count": len(time), "start_utc": str(time.min()), "end_utc": str(time.max()),
               "selection": "41.5 < lat < 43.5; -71.5 < lon < -69.5; 18:06:20 <= UTC < 18:06:24",
               "geolocation": "GLM Level 2 apparent coordinates; no bolide-altitude parallax correction"}
    (DATA / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
