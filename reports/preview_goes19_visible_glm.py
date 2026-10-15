"""Preview the native GOES-19 visible image with raw GLM event locations."""
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "products/goes19_glm_20260530"


def main():
    abi = xr.open_dataset(next(DATA.glob("OR_ABI*.nc")))
    glm = xr.open_dataset(next(DATA.glob("OR_GLM*.nc")))
    meta = abi.goes_imager_projection.attrs
    h = meta["perspective_point_height"]
    globe = ccrs.Globe(semimajor_axis=meta["semi_major_axis"],
                       semiminor_axis=meta["semi_minor_axis"])
    projection = ccrs.Geostationary(central_longitude=meta["longitude_of_projection_origin"],
                                  satellite_height=h, sweep_axis=meta["sweep_angle_axis"], globe=globe)
    west, east, south, north = -73.2, -68.3, 40.3, 44.2
    corners = projection.transform_points(ccrs.PlateCarree(),
                np.array([west, west, east, east]), np.array([south, north, south, north]))
    x, y = abi.x.values * h, abi.y.values * h
    xi = np.where((x >= corners[:, 0].min()) & (x <= corners[:, 0].max()))[0]
    yi = np.where((y >= corners[:, 1].min()) & (y <= corners[:, 1].max()))[0]
    region = abi.isel(x=slice(xi.min(), xi.max()+1), y=slice(yi.min(), yi.max()+1))
    reflectance = region.CMI.values
    quality = region.DQF.values
    reflectance = np.where(quality <= 1, reflectance, np.nan)
    extent = [x[xi.min()] - (x[1]-x[0])/2, x[xi.max()] + (x[1]-x[0])/2,
              y[yi.max()] + (y[1]-y[0])/2, y[yi.min()] - (y[1]-y[0])/2]
    lat, lon = glm.event_lat.values, glm.event_lon.values
    t = glm.event_time_offset.values
    keep = ((lat > 41.5) & (lat < 43.5) & (lon > -71.5) & (lon < -69.5)
            & (t >= np.datetime64("2026-05-30T18:06:20"))
            & (t < np.datetime64("2026-05-30T18:06:24")))
    plt.rcParams.update({"font.size": 11})
    for overlay, name in [(False, "goes19_visible_clouds"), (True, "goes19_visible_clouds_glm")]:
        fig = plt.figure(figsize=(9, 7.5), layout="constrained")
        ax = fig.add_subplot(projection=projection)
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        image = ax.imshow(reflectance, extent=extent, origin="upper", transform=projection,
                          cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.coastlines("10m", color="#53c9f0", linewidth=0.55)
        grid = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, color="white")
        grid.top_labels = grid.right_labels = False
        if overlay:
            ax.scatter(lon[keep], lat[keep], s=25, facecolors="none", edgecolors="#ff492f",
                       linewidths=0.65, transform=ccrs.PlateCarree(), zorder=5)
        ax.set_title("GOES-19 ABI 0.64 µm · 30 May 2026 · 18:06 UTC" +
                     ("\nGLM event pixels, 18:06:21.70–23.09 UTC" if overlay else ""))
        fig.colorbar(image, ax=ax, shrink=0.8, label="ABI reflectance factor")
        fig.savefig(DATA / (name + ".png"), dpi=220)
        fig.savefig(DATA / (name + ".pdf"), dpi=220)
        plt.close(fig)
    print("ABI coverage:", abi.attrs.get("time_coverage_start"), abi.attrs.get("time_coverage_end"))
    print("Native regional image shape:", reflectance.shape)
    print(DATA)


if __name__ == "__main__":
    main()
