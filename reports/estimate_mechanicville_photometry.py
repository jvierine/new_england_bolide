#!/usr/bin/env python3
"""Estimate Mechanicville bolide photometry from a Rayleigh-sky reference."""

import json
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
from PIL import Image
from scipy.ndimage import binary_dilation


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data/security_camera/mechanicville_20260530_140622.png"
ANNOTATION = ROOT / "annotations/mechanicville_bolide.h5"
OUTPUT = ROOT / "reports/figures/mechanicville_photometry_regions.png"
TRAJECTORY = (
    ROOT
    / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh"
    / "ballistic_head_trail_correlated_fit.json"
)
H5_OUTPUT = ROOT / "products/mechanicville_photometry/mechanicville_photometry.h5"

EVENT_UNIX = 1780164382.0
OBSERVER_LAT_DEG = 42.0
OBSERVER_LON_DEG = -73.0
OBSERVER_HEIGHT_M = 89.0
ASSUMED_HORIZONTAL_FOV_DEG = 120.0
SOLAR_V_MAG = -26.76
CHANNEL_NAMES = ("red", "green", "blue")
EFFECTIVE_WAVELENGTH_UM = np.array([0.610, 0.540, 0.460])


def add_regions(
    ax, meteor_vertices: np.ndarray, sky_vertices: np.ndarray, *, labels: bool
) -> None:
    ax.add_patch(
        Polygon(
            meteor_vertices, closed=True, fill=False, edgecolor="#00ffff",
            linewidth=2.2, label="bolide",
        )
    )
    ax.add_patch(
        Polygon(
            sky_vertices, closed=True, fill=False, edgecolor="#0066ff",
            linewidth=2.5, label="blue-sky reference",
        )
    )
    if labels:
        ax.legend(loc="best")


def inverse_srgb(encoded: np.ndarray) -> np.ndarray:
    """Approximate scene-linear values from the displayed 8-bit screenshot."""
    encoded = np.asarray(encoded, dtype=float)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def polygon_mask(vertices_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    centers = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    return MplPath(vertices_xy).contains_points(centers).reshape(shape)


def robust_local_background(
    channel: np.ndarray, meteor_mask: np.ndarray, vertices_xy: np.ndarray,
    padding_px: int = 45,
) -> tuple[np.ndarray, float]:
    """Fit a clipped quadratic background around the meteor aperture."""
    height, width = channel.shape
    x0 = max(0, int(np.floor(vertices_xy[:, 0].min())) - padding_px)
    x1 = min(width, int(np.ceil(vertices_xy[:, 0].max())) + padding_px + 1)
    y0 = max(0, int(np.floor(vertices_xy[:, 1].min())) - padding_px)
    y1 = min(height, int(np.ceil(vertices_xy[:, 1].max())) + padding_px + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xn = (xx - 0.5 * (x0 + x1)) / (0.5 * (x1 - x0))
    yn = (yy - 0.5 * (y0 + y1)) / (0.5 * (y1 - y0))
    design = np.stack(
        [
            np.ones(xx.size), xn.ravel(), yn.ravel(), (xn * xn).ravel(),
            (xn * yn).ravel(), (yn * yn).ravel(),
        ],
        axis=1,
    )
    values = channel[y0:y1, x0:x1].ravel()
    excluded = binary_dilation(meteor_mask, iterations=3)[y0:y1, x0:x1].ravel()
    good = ~excluded
    for _ in range(8):
        coeff = np.linalg.lstsq(design[good], values[good], rcond=None)[0]
        residual = values - design @ coeff
        center = np.median(residual[good])
        sigma = 1.4826 * np.median(np.abs(residual[good] - center))
        good = (~excluded) & (np.abs(residual - center) < 2.5 * max(sigma, 1e-8))
    model = np.full(channel.shape, np.nan, dtype=float)
    model[y0:y1, x0:x1] = (design @ coeff).reshape(y1 - y0, x1 - x0)
    return model, float(np.std(residual[good]))


def rayleigh_optical_depth(wavelength_um: float) -> float:
    """Bucholtz (1995) sea-level vertical Rayleigh optical depth."""
    inverse2 = wavelength_um ** -2
    return 0.008569 * wavelength_um ** -4 * (
        1.0 + 0.0113 * inverse2 + 0.00013 * inverse2**2
    )


def rayleigh_sky_ratio(
    wavelength_um: float,
    meteor_elevation_deg: float,
    solar_elevation_deg: float,
    scattering_angle_deg: float,
    meteor_height_m: float,
) -> tuple[float, float]:
    """Return L_sky/F_sun [sr^-1] and meteor extinction [mag]."""
    tau0 = rayleigh_optical_depth(wavelength_um)
    scale_height_m = 8000.0
    elevation = np.deg2rad(meteor_elevation_deg)
    solar_elevation = np.deg2rad(solar_elevation_deg)
    scattering_angle = np.deg2rad(scattering_angle_deg)
    z = np.linspace(OBSERVER_HEIGHT_M, 120_000.0, 100_000)
    density_ratio = np.exp(-z / scale_height_m)
    observer_density_ratio = np.exp(-OBSERVER_HEIGHT_M / scale_height_m)
    beta = tau0 * density_ratio / scale_height_m
    tau_view = tau0 * (observer_density_ratio - density_ratio) / np.sin(elevation)
    tau_sun = tau0 * density_ratio / np.sin(solar_elevation)
    phase = 0.75 * (1.0 + np.cos(scattering_angle) ** 2)
    integrand = (
        beta * phase / (4.0 * np.pi)
        * np.exp(-tau_view - tau_sun) / np.sin(elevation)
    )
    radiance_ratio = float(np.trapezoid(integrand, z))
    meteor_density_ratio = np.exp(-meteor_height_m / scale_height_m)
    meteor_tau = tau0 * (
        observer_density_ratio - meteor_density_ratio
    ) / np.sin(elevation)
    extinction_mag = float(2.5 / np.log(10.0) * meteor_tau)
    return radiance_ratio, extinction_mag


def event_geometry() -> dict[str, float]:
    trajectory = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    model = trajectory["model"]
    times = np.asarray(model["time_unix"], dtype=float)
    meteor_lat = float(np.interp(EVENT_UNIX, times, model["lat_deg"]))
    meteor_lon = float(np.interp(EVENT_UNIX, times, model["lon_deg"]))
    meteor_height = float(np.interp(EVENT_UNIX, times, model["alt_m"]))
    observer = EarthLocation.from_geodetic(
        OBSERVER_LON_DEG * u.deg,
        OBSERVER_LAT_DEG * u.deg,
        OBSERVER_HEIGHT_M * u.m,
    )
    meteor = EarthLocation.from_geodetic(
        meteor_lon * u.deg, meteor_lat * u.deg, meteor_height * u.m
    )
    event_time = Time(EVENT_UNIX, format="unix", scale="utc")
    observer_itrs = observer.get_itrs(obstime=event_time)
    meteor_itrs = meteor.get_itrs(obstime=event_time)
    delta = meteor_itrs.cartesian - observer_itrs.cartesian
    # A topocentric ITRS vector and a direct ENU rotation give the same result;
    # use the latter to avoid interpreting the vector as a geocentric point.
    dx, dy, dz = [value.to_value(u.m) for value in delta.xyz]
    lat = np.deg2rad(OBSERVER_LAT_DEG)
    lon = np.deg2rad(OBSERVER_LON_DEG)
    east = -np.sin(lon) * dx + np.cos(lon) * dy
    north = (
        -np.sin(lat) * np.cos(lon) * dx
        - np.sin(lat) * np.sin(lon) * dy
        + np.cos(lat) * dz
    )
    up = (
        np.cos(lat) * np.cos(lon) * dx
        + np.cos(lat) * np.sin(lon) * dy
        + np.sin(lat) * dz
    )
    distance = float(np.sqrt(east**2 + north**2 + up**2))
    meteor_az = float(np.rad2deg(np.arctan2(east, north)) % 360.0)
    meteor_el = float(np.rad2deg(np.arcsin(up / distance)))

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    sun = get_sun(event_time).transform_to(
        AltAz(obstime=event_time, location=observer, pressure=0 * u.hPa)
    )
    sun_az = float(sun.az.deg)
    sun_el = float(sun.alt.deg)
    separation_cos = (
        np.sin(np.deg2rad(meteor_el)) * np.sin(np.deg2rad(sun_el))
        + np.cos(np.deg2rad(meteor_el)) * np.cos(np.deg2rad(sun_el))
        * np.cos(np.deg2rad(meteor_az - sun_az))
    )
    return {
        "meteor_lat_deg": meteor_lat,
        "meteor_lon_deg": meteor_lon,
        "meteor_height_m": meteor_height,
        "range_m": distance,
        "meteor_az_deg": meteor_az,
        "meteor_el_deg": meteor_el,
        "sun_az_deg": sun_az,
        "sun_el_deg": sun_el,
        "scattering_angle_deg": float(np.rad2deg(np.arccos(separation_cos))),
    }


def pixel_solid_angle(image_shape: tuple[int, int], x: float, y: float) -> float:
    """Rectilinear-pinhole solid angle under the stated provisional FOV."""
    height, width = image_shape
    focal_px = 0.5 * width / np.tan(0.5 * np.deg2rad(ASSUMED_HORIZONTAL_FOV_DEG))
    xn = (x - 0.5 * (width - 1)) / focal_px
    yn = (y - 0.5 * (height - 1)) / focal_px
    return float(1.0 / (focal_px**2 * (1.0 + xn**2 + yn**2) ** 1.5))


def estimate_photometry(
    image_rgb: np.ndarray, meteor_vertices: np.ndarray, sky_vertices: np.ndarray
) -> dict[str, np.ndarray | float | dict]:
    encoded = image_rgb.astype(float) / 255.0
    linear = inverse_srgb(encoded)
    meteor = polygon_mask(meteor_vertices, encoded.shape[:2])
    sky = polygon_mask(sky_vertices, encoded.shape[:2])
    geometry = event_geometry()
    meteor_x, meteor_y = np.mean(meteor_vertices, axis=0)
    omega = pixel_solid_angle(encoded.shape[:2], meteor_x, meteor_y)

    signal = np.empty(3)
    sky_level = np.empty(3)
    q_sky_pixels = np.empty(3)
    saturation_count = np.empty(3, dtype=int)
    sky_saturation_count = np.empty(3, dtype=int)
    background_rms = np.empty(3)
    radiance_ratio = np.empty(3)
    extinction_mag = np.empty(3)
    apparent_mag = np.empty(3)
    absolute_mag = np.empty(3)
    for channel in range(3):
        background, background_rms[channel] = robust_local_background(
            linear[..., channel], meteor, meteor_vertices
        )
        signal[channel] = np.nansum(
            (linear[..., channel] - background)[meteor]
        )
        sky_level[channel] = np.median(linear[..., channel][sky])
        q_sky_pixels[channel] = signal[channel] / sky_level[channel]
        saturation_count[channel] = int(np.sum(encoded[..., channel][meteor] >= 1.0))
        sky_saturation_count[channel] = int(np.sum(encoded[..., channel][sky] >= 1.0))
        radiance_ratio[channel], extinction_mag[channel] = rayleigh_sky_ratio(
            EFFECTIVE_WAVELENGTH_UM[channel],
            geometry["meteor_el_deg"],
            geometry["sun_el_deg"],
            geometry["scattering_angle_deg"],
            geometry["meteor_height_m"],
        )
        flux_ratio = q_sky_pixels[channel] * omega * radiance_ratio[channel]
        apparent_mag[channel] = SOLAR_V_MAG - 2.5 * np.log10(flux_ratio) if flux_ratio > 0 else np.nan
        absolute_mag[channel] = (
            apparent_mag[channel]
            - extinction_mag[channel]
            - 5.0 * np.log10(geometry["range_m"] / 100_000.0)
        )
    return {
        "geometry": geometry,
        "meteor_mask": meteor,
        "sky_mask": sky,
        "linear_signal": signal,
        "sky_level": sky_level,
        "q_sky_pixels": q_sky_pixels,
        "saturation_count": saturation_count,
        "sky_saturation_count": sky_saturation_count,
        "background_rms": background_rms,
        "radiance_ratio_sr_inv": radiance_ratio,
        "extinction_mag": extinction_mag,
        "apparent_mag": apparent_mag,
        "absolute_mag_100km": absolute_mag,
        "pixel_solid_angle_sr": omega,
    }


def write_hdf5(result: dict[str, np.ndarray | float | dict]) -> None:
    H5_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_OUTPUT, "w") as h5:
        h5.attrs["schema"] = "new_england_bolide.mechanicville_photometry.v1"
        h5.attrs["status"] = "manual masks; screenshot-calibrated; saturation-limited"
        h5.attrs["source_image"] = str(IMAGE.relative_to(ROOT))
        h5.attrs["annotation_source"] = str(ANNOTATION.relative_to(ROOT))
        h5.attrs["trajectory_source"] = str(TRAJECTORY.relative_to(ROOT))
        h5.attrs["event_unix"] = EVENT_UNIX
        h5.attrs["assumed_horizontal_fov_deg"] = ASSUMED_HORIZONTAL_FOV_DEG
        h5.attrs["solar_v_magnitude"] = SOLAR_V_MAG
        h5.attrs["magnitude_bound_convention"] = (
            "for any clipped channel, true magnitude <= stored magnitude; "
            "equivalently true flux >= measured flux"
        )
        h5.create_dataset("channel_name", data=np.asarray(CHANNEL_NAMES, dtype="S"))
        h5.create_dataset("effective_wavelength_um", data=EFFECTIVE_WAVELENGTH_UM)
        for key in (
            "linear_signal", "sky_level", "q_sky_pixels", "saturation_count",
            "sky_saturation_count", "background_rms",
            "radiance_ratio_sr_inv", "extinction_mag", "apparent_mag",
            "absolute_mag_100km",
        ):
            h5.create_dataset(key, data=result[key])
        h5.create_dataset("meteor_mask", data=result["meteor_mask"], compression="gzip")
        h5.create_dataset("sky_mask", data=result["sky_mask"], compression="gzip")
        h5.create_dataset("pixel_solid_angle_sr", data=result["pixel_solid_angle_sr"])
        geometry_group = h5.create_group("geometry")
        for key, value in result["geometry"].items():
            geometry_group.create_dataset(key, data=value)


def main() -> None:
    with h5py.File(ANNOTATION, "r") as annotation:
        meteor_vertices = annotation["bolide_polygon_xy"][...]
        sky_vertices = annotation["blue_sky_polygon_xy"][...]
    image = Image.open(IMAGE).convert("RGB")
    image_array = np.asarray(image)
    result = estimate_photometry(image_array, meteor_vertices, sky_vertices)
    write_hdf5(result)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    axes[0].imshow(image)
    add_regions(axes[0], meteor_vertices, sky_vertices, labels=False)
    axes[0].set_title("Full frame: region location")
    axes[0].set_xlabel("pixel x")
    axes[0].set_ylabel("pixel y")

    axes[1].imshow(image)
    add_regions(axes[1], meteor_vertices, sky_vertices, labels=True)
    xmin = min(meteor_vertices[:, 0].min(), sky_vertices[:, 0].min()) - 30
    xmax = max(meteor_vertices[:, 0].max(), sky_vertices[:, 0].max()) + 30
    ymin = min(meteor_vertices[:, 1].min(), sky_vertices[:, 1].min()) - 30
    ymax = max(meteor_vertices[:, 1].max(), sky_vertices[:, 1].max()) + 30
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymax, ymin)
    axes[1].set_title("Photometry-region zoom")
    axes[1].set_xlabel("pixel x")
    axes[1].set_ylabel("pixel y")
    axes[1].grid(color="white", alpha=0.28, linewidth=0.6)

    fig.suptitle(
        "Mechanicville WYZE frame, 2026-05-30 14:06:22 EDT\n"
        "Cyan: manual bolide aperture; blue: manual sky reference",
        fontsize=13,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)
    print(H5_OUTPUT)
    for index, name in enumerate(CHANNEL_NAMES):
        print(
            name,
            f"Q={result['q_sky_pixels'][index]:.3f}",
            f"saturated={result['saturation_count'][index]}/{result['meteor_mask'].sum()}",
            f"M100_upper_bound={result['absolute_mag_100km'][index]:.3f}",
        )


if __name__ == "__main__":
    main()
