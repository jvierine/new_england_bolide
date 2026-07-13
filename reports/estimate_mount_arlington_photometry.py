#!/usr/bin/env python3
"""Estimate Mount Arlington bolide magnitude from manual sky/bolide masks."""

from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers
from PIL import Image
from scipy.ndimage import binary_dilation

from estimate_mechanicville_photometry import (
    EFFECTIVE_WAVELENGTH_UM,
    SOLAR_V_MAG,
    inverse_srgb,
    rayleigh_optical_depth,
)


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "annotations/mount_arlington_bolide.h5"
TRAJECTORY = (
    ROOT
    / "products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh"
    / "ballistic_head_trail_correlated_fit.json"
)
OUTPUT = ROOT / "products/mount_arlington_photometry/mount_arlington_photometry.h5"
FIGURE = ROOT / "reports/figures/mount_arlington_magnitude_vs_fov.png"
EVENT_UNIX = 1780164382.0
ASSUMED_OBSERVER_HEIGHT_M = 0.0
REFERENCE_HORIZONTAL_FOV_DEG = 120.0
FOV_GRID_DEG = np.linspace(80.0, 140.0, 241)
CHANNEL_NAMES = ("red", "green", "blue")


def polygon_mask(vertices_xy: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    yy, xx = np.mgrid[:height, :width]
    pixel_centers = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    return MplPath(vertices_xy).contains_points(pixel_centers).reshape(height, width)


def local_background(
    channel: np.ndarray,
    bolide_mask: np.ndarray,
    vertices_xy: np.ndarray,
    padding_px: int = 35,
) -> tuple[np.ndarray, float]:
    height, width = channel.shape
    x0 = max(0, int(np.floor(vertices_xy[:, 0].min())) - padding_px)
    x1 = min(width, int(np.ceil(vertices_xy[:, 0].max())) + padding_px + 1)
    y0 = max(0, int(np.floor(vertices_xy[:, 1].min())) - padding_px)
    y1 = min(height, int(np.ceil(vertices_xy[:, 1].max())) + padding_px + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xn = (xx - 0.5 * (x0 + x1)) / max(1.0, 0.5 * (x1 - x0))
    yn = (yy - 0.5 * (y0 + y1)) / max(1.0, 0.5 * (y1 - y0))
    design = np.stack(
        [
            np.ones(xx.size), xn.ravel(), yn.ravel(), (xn * xn).ravel(),
            (xn * yn).ravel(), (yn * yn).ravel(),
        ],
        axis=1,
    )
    values = channel[y0:y1, x0:x1].ravel()
    excluded = binary_dilation(bolide_mask, iterations=3)[y0:y1, x0:x1].ravel()
    good = ~excluded
    for _ in range(8):
        coefficients = np.linalg.lstsq(design[good], values[good], rcond=None)[0]
        residual = values - design @ coefficients
        center = np.median(residual[good])
        sigma = 1.4826 * np.median(np.abs(residual[good] - center))
        good = (~excluded) & (np.abs(residual - center) < 2.5 * max(sigma, 1e-8))
    model = np.full(channel.shape, np.nan, dtype=float)
    model[y0:y1, x0:x1] = (design @ coefficients).reshape(y1 - y0, x1 - x0)
    return model, float(np.std(residual[good]))


def event_geometry(latitude_deg: float, longitude_deg: float) -> dict[str, float]:
    trajectory = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    model = trajectory["model"]
    times = np.asarray(model["time_unix"], dtype=float)
    meteor_lat = float(np.interp(EVENT_UNIX, times, model["lat_deg"]))
    meteor_lon = float(np.interp(EVENT_UNIX, times, model["lon_deg"]))
    meteor_height = float(np.interp(EVENT_UNIX, times, model["alt_m"]))
    observer = EarthLocation.from_geodetic(
        longitude_deg * u.deg, latitude_deg * u.deg, ASSUMED_OBSERVER_HEIGHT_M * u.m
    )
    meteor = EarthLocation.from_geodetic(
        meteor_lon * u.deg, meteor_lat * u.deg, meteor_height * u.m
    )
    event_time = Time(EVENT_UNIX, format="unix", scale="utc")
    delta = meteor.get_itrs(obstime=event_time).cartesian - observer.get_itrs(
        obstime=event_time
    ).cartesian
    dx, dy, dz = [component.to_value(u.m) for component in delta.xyz]
    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
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
    separation_cosine = (
        np.sin(np.deg2rad(meteor_el)) * np.sin(np.deg2rad(sun.alt.deg))
        + np.cos(np.deg2rad(meteor_el)) * np.cos(np.deg2rad(sun.alt.deg))
        * np.cos(np.deg2rad(meteor_az - sun.az.deg))
    )
    return {
        "meteor_lat_deg": meteor_lat,
        "meteor_lon_deg": meteor_lon,
        "meteor_height_m": meteor_height,
        "range_m": distance,
        "meteor_az_deg": meteor_az,
        "meteor_el_deg": meteor_el,
        "sun_az_deg": float(sun.az.deg),
        "sun_el_deg": float(sun.alt.deg),
        "scattering_angle_deg": float(np.rad2deg(np.arccos(separation_cosine))),
    }


def rayleigh_sky_ratio(
    wavelength_um: float,
    meteor_elevation_deg: float,
    solar_elevation_deg: float,
    scattering_angle_deg: float,
    meteor_height_m: float,
) -> tuple[float, float]:
    tau0 = rayleigh_optical_depth(wavelength_um)
    scale_height_m = 8000.0
    elevation = np.deg2rad(meteor_elevation_deg)
    solar_elevation = np.deg2rad(solar_elevation_deg)
    scattering_angle = np.deg2rad(scattering_angle_deg)
    z = np.linspace(ASSUMED_OBSERVER_HEIGHT_M, 120_000.0, 100_000)
    density_ratio = np.exp(-z / scale_height_m)
    observer_ratio = np.exp(-ASSUMED_OBSERVER_HEIGHT_M / scale_height_m)
    beta = tau0 * density_ratio / scale_height_m
    tau_view = tau0 * (observer_ratio - density_ratio) / np.sin(elevation)
    tau_sun = tau0 * density_ratio / np.sin(solar_elevation)
    phase = 0.75 * (1.0 + np.cos(scattering_angle) ** 2)
    radiance_ratio = float(
        np.trapezoid(
            beta * phase / (4.0 * np.pi)
            * np.exp(-tau_view - tau_sun) / np.sin(elevation),
            z,
        )
    )
    meteor_ratio = np.exp(-meteor_height_m / scale_height_m)
    meteor_tau = tau0 * (observer_ratio - meteor_ratio) / np.sin(elevation)
    return radiance_ratio, float(2.5 / np.log(10.0) * meteor_tau)


def pixel_solid_angle(
    image_shape: tuple[int, int], x: float, y: float, horizontal_fov_deg: float
) -> float:
    height, width = image_shape
    focal_px = 0.5 * width / np.tan(0.5 * np.deg2rad(horizontal_fov_deg))
    xn = (x - 0.5 * (width - 1)) / focal_px
    yn = (y - 0.5 * (height - 1)) / focal_px
    return float(1.0 / (focal_px**2 * (1.0 + xn**2 + yn**2) ** 1.5))


def main() -> None:
    with h5py.File(ANNOTATION, "r") as annotation:
        image_path = ROOT / annotation.attrs["source_image"]
        latitude_deg = float(annotation.attrs["observer_latitude_deg"])
        longitude_deg = float(annotation.attrs["observer_longitude_deg"])
        bolide_vertices = annotation["bolide_polygon_xy"][...]
        sky_vertices = annotation["blue_sky_polygon_xy"][...]
    image = np.asarray(Image.open(image_path).convert("RGB"))
    encoded = image.astype(float) / 255.0
    linear = inverse_srgb(encoded)
    bolide_mask = polygon_mask(bolide_vertices, image.shape[:2])
    sky_mask = polygon_mask(sky_vertices, image.shape[:2])
    geometry = event_geometry(latitude_deg, longitude_deg)

    signal = np.empty(3)
    sky_level = np.empty(3)
    q_sky_pixels = np.empty(3)
    background_rms = np.empty(3)
    bolide_saturated = np.empty(3, dtype=int)
    sky_saturated = np.empty(3, dtype=int)
    radiance_ratio = np.empty(3)
    extinction_mag = np.empty(3)
    for channel in range(3):
        background, background_rms[channel] = local_background(
            linear[..., channel], bolide_mask, bolide_vertices
        )
        signal[channel] = np.nansum((linear[..., channel] - background)[bolide_mask])
        sky_level[channel] = np.median(linear[..., channel][sky_mask])
        q_sky_pixels[channel] = signal[channel] / sky_level[channel]
        bolide_saturated[channel] = int(np.sum(image[..., channel][bolide_mask] == 255))
        sky_saturated[channel] = int(np.sum(image[..., channel][sky_mask] == 255))
        radiance_ratio[channel], extinction_mag[channel] = rayleigh_sky_ratio(
            EFFECTIVE_WAVELENGTH_UM[channel],
            geometry["meteor_el_deg"],
            geometry["sun_el_deg"],
            geometry["scattering_angle_deg"],
            geometry["meteor_height_m"],
        )

    centroid = np.mean(bolide_vertices, axis=0)
    solid_angle = np.array(
        [
            pixel_solid_angle(image.shape[:2], centroid[0], centroid[1], fov)
            for fov in FOV_GRID_DEG
        ]
    )
    magnitude = np.full((len(FOV_GRID_DEG), 3), np.nan)
    for channel in range(3):
        flux_ratio = q_sky_pixels[channel] * solid_angle * radiance_ratio[channel]
        valid = flux_ratio > 0
        apparent = SOLAR_V_MAG - 2.5 * np.log10(flux_ratio[valid])
        magnitude[valid, channel] = (
            apparent
            - extinction_mag[channel]
            - 5.0 * np.log10(geometry["range_m"] / 100_000.0)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT, "w") as h5:
        h5.attrs["schema"] = "new_england_bolide.mount_arlington_photometry.v1"
        h5.attrs["status"] = "provisional; unknown camera field of view"
        h5.attrs["annotation_source"] = str(ANNOTATION.relative_to(ROOT))
        h5.attrs["trajectory_source"] = str(TRAJECTORY.relative_to(ROOT))
        h5.attrs["reference_horizontal_fov_deg"] = REFERENCE_HORIZONTAL_FOV_DEG
        h5.attrs["observer_height_status"] = "unknown; assumed 0 m"
        h5.attrs["magnitude_bound_convention"] = (
            "for any clipped channel, true magnitude <= stored magnitude; "
            "equivalently true flux >= measured flux"
        )
        h5.create_dataset("channel_name", data=np.asarray(CHANNEL_NAMES, dtype="S"))
        h5.create_dataset("effective_wavelength_um", data=EFFECTIVE_WAVELENGTH_UM)
        h5.create_dataset("bolide_mask", data=bolide_mask, compression="gzip")
        h5.create_dataset("blue_sky_mask", data=sky_mask, compression="gzip")
        h5.create_dataset("linear_signal", data=signal)
        h5.create_dataset("sky_level", data=sky_level)
        h5.create_dataset("q_sky_pixels", data=q_sky_pixels)
        h5.create_dataset("background_rms", data=background_rms)
        h5.create_dataset("bolide_saturated_pixel_count", data=bolide_saturated)
        h5.create_dataset("blue_sky_saturated_pixel_count", data=sky_saturated)
        h5.create_dataset("rayleigh_radiance_ratio_sr_inv", data=radiance_ratio)
        h5.create_dataset("extinction_mag", data=extinction_mag)
        h5.create_dataset("horizontal_fov_deg", data=FOV_GRID_DEG)
        h5.create_dataset("pixel_solid_angle_sr", data=solid_angle)
        h5.create_dataset("absolute_magnitude_100km", data=magnitude)
        group = h5.create_group("geometry")
        for key, value in geometry.items():
            group.create_dataset(key, data=value)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = ("C3", "C2", "C0")
    reference_values = []
    for channel, (name, color) in enumerate(zip(CHANNEL_NAMES, colors, strict=True)):
        reference_value = float(
            np.interp(
                REFERENCE_HORIZONTAL_FOV_DEG,
                FOV_GRID_DEG,
                magnitude[:, channel],
            )
        )
        reference_values.append(reference_value)
        ax.plot(
            FOV_GRID_DEG,
            magnitude[:, channel],
            color=color,
            linewidth=2,
            label=(
                f"{name}: $M_{{true}}\\leq M_{{measured}}$ "
                f"({bolide_saturated[channel]} clipped)"
            ),
        )
        ax.scatter(
            [REFERENCE_HORIZONTAL_FOV_DEG], [reference_value],
            color=color, zorder=3,
        )
        ax.annotate(
            f"$\\leq {reference_value:.2f}$",
            (REFERENCE_HORIZONTAL_FOV_DEG, reference_value),
            xytext=(7, 2), textcoords="offset points", color=color,
        )
    ax.axvline(REFERENCE_HORIZONTAL_FOV_DEG, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("assumed horizontal field of view (deg)")
    ax.set_ylabel("measured 100-km magnitude upper bound")
    ax.set_title("Mount Arlington independent RGB saturation bounds")
    ax.grid(alpha=0.25)
    ax.legend()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    print(f"Saved: {OUTPUT}")
    print(f"Saved: {FIGURE}")
    print(
        f"geometry: range={geometry['range_m']/1000:.3f} km, "
        f"az={geometry['meteor_az_deg']:.3f} deg, el={geometry['meteor_el_deg']:.3f} deg"
    )
    for index, name in enumerate(CHANNEL_NAMES):
        value = float(np.interp(REFERENCE_HORIZONTAL_FOV_DEG, FOV_GRID_DEG, magnitude[:, index]))
        print(
            f"{name}: Q={q_sky_pixels[index]:.3f}, "
            f"saturated={bolide_saturated[index]}/{bolide_mask.sum()}, "
            f"M100_true<={value:.3f} at {REFERENCE_HORIZONTAL_FOV_DEG:.0f} deg"
        )


if __name__ == "__main__":
    main()
