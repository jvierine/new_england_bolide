#!/usr/bin/env python3
"""Plot the modeled Mount Arlington Rayleigh-sky photometric reference."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from estimate_mechanicville_photometry import (
    EFFECTIVE_WAVELENGTH_UM,
    rayleigh_optical_depth,
)
from estimate_mount_arlington_photometry import (
    ANNOTATION,
    ASSUMED_OBSERVER_HEIGHT_M,
    CHANNEL_NAMES,
    REFERENCE_HORIZONTAL_FOV_DEG,
    event_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "reports/figures/mount_arlington_rayleigh_reference.png"
OUTPUT = (
    ROOT
    / "products/mount_arlington_photometry"
    / "mount_arlington_rayleigh_reference.h5"
)
FOV_GRID_DEG = np.linspace(80.0, 140.0, 121)
MAP_STEP_PX = 3
MAP_WAVELENGTH_UM = 0.540
SCALE_HEIGHT_M = 8000.0


def azel_unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return an east-north-up unit vector."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.asarray(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    )


def camera_ray(
    x: np.ndarray | float,
    y: np.ndarray | float,
    image_shape: tuple[int, int],
    horizontal_fov_deg: float,
) -> np.ndarray:
    """Rectilinear camera ray; image x is right and image y is down."""
    height, width = image_shape
    focal_px = 0.5 * width / np.tan(0.5 * np.deg2rad(horizontal_fov_deg))
    xn = (np.asarray(x) - 0.5 * (width - 1)) / focal_px
    yn = -(np.asarray(y) - 0.5 * (height - 1)) / focal_px
    ray = np.stack(np.broadcast_arrays(xn, yn, np.ones_like(xn)), axis=-1)
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def tangent_basis(ray: np.ndarray, nominal_up: np.ndarray) -> np.ndarray:
    """Right-up-ray basis with up as close as possible to nominal_up."""
    up = nominal_up - np.dot(nominal_up, ray) * ray
    up /= np.linalg.norm(up)
    right = np.cross(up, ray)
    right /= np.linalg.norm(right)
    return np.column_stack((right, up, ray))


def camera_to_enu_rotation(
    image_shape: tuple[int, int],
    meteor_xy: np.ndarray,
    horizontal_fov_deg: float,
    meteor_azimuth_deg: float,
    meteor_elevation_deg: float,
) -> np.ndarray:
    """Anchor the meteor pixel to its trajectory direction with zero local roll."""
    camera_meteor = camera_ray(
        meteor_xy[0], meteor_xy[1], image_shape, horizontal_fov_deg
    )
    world_meteor = azel_unit(meteor_azimuth_deg, meteor_elevation_deg)
    camera_basis = tangent_basis(camera_meteor, np.asarray([0.0, 1.0, 0.0]))
    world_basis = tangent_basis(world_meteor, np.asarray([0.0, 0.0, 1.0]))
    return world_basis @ camera_basis.T


def rays_to_azel(rays_enu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east = rays_enu[..., 0]
    north = rays_enu[..., 1]
    up = np.clip(rays_enu[..., 2], -1.0, 1.0)
    azimuth = np.rad2deg(np.arctan2(east, north)) % 360.0
    elevation = np.rad2deg(np.arcsin(up))
    return azimuth, elevation


def rayleigh_elevation_kernel(
    wavelength_um: float,
    elevations_deg: np.ndarray,
    solar_elevation_deg: float,
) -> np.ndarray:
    """Rayleigh radiance without the phase factor, tabulated versus elevation."""
    tau0 = rayleigh_optical_depth(wavelength_um)
    z = np.linspace(ASSUMED_OBSERVER_HEIGHT_M, 120_000.0, 5000)
    density = np.exp(-z / SCALE_HEIGHT_M)
    observer_density = np.exp(-ASSUMED_OBSERVER_HEIGHT_M / SCALE_HEIGHT_M)
    beta = tau0 * density / SCALE_HEIGHT_M
    tau_sun = tau0 * density / np.sin(np.deg2rad(solar_elevation_deg))
    result = np.full_like(elevations_deg, np.nan, dtype=float)
    positive = elevations_deg > 0.05
    sine = np.sin(np.deg2rad(elevations_deg[positive]))
    tau_view = (
        tau0
        * (observer_density - density[:, None])
        / sine[None, :]
    )
    integrand = (
        beta[:, None]
        * np.exp(-tau_view - tau_sun[:, None])
        / (4.0 * np.pi * sine[None, :])
    )
    result[positive] = np.trapezoid(integrand, z, axis=0)
    return result


def rayleigh_ratio(
    wavelength_um: float,
    rays_enu: np.ndarray,
    sun_enu: np.ndarray,
    solar_elevation_deg: float,
) -> np.ndarray:
    """Single-scattered L_sky/F_sun in sr^-1 for arbitrary ENU rays."""
    _, elevation = rays_to_azel(rays_enu)
    elevation_grid = np.linspace(0.05, 90.0, 1800)
    kernel = rayleigh_elevation_kernel(
        wavelength_um, elevation_grid, solar_elevation_deg
    )
    interpolated = np.interp(
        elevation, elevation_grid, kernel, left=np.nan, right=kernel[-1]
    )
    cosine = np.clip(np.sum(rays_enu * sun_enu, axis=-1), -1.0, 1.0)
    phase = 0.75 * (1.0 + cosine**2)
    return interpolated * phase


def main() -> None:
    with h5py.File(ANNOTATION, "r") as annotation:
        image_path = ROOT / annotation.attrs["source_image"]
        latitude_deg = float(annotation.attrs["observer_latitude_deg"])
        longitude_deg = float(annotation.attrs["observer_longitude_deg"])
        bolide_vertices = annotation["bolide_polygon_xy"][...]
        sky_vertices = annotation["blue_sky_polygon_xy"][...]
    image = np.asarray(Image.open(image_path).convert("RGB"))
    image_shape = image.shape[:2]
    meteor_xy = np.mean(bolide_vertices, axis=0)
    sky_xy = np.mean(sky_vertices, axis=0)
    geometry = event_geometry(latitude_deg, longitude_deg)
    sun_enu = azel_unit(geometry["sun_az_deg"], geometry["sun_el_deg"])

    meteor_ray = azel_unit(
        geometry["meteor_az_deg"], geometry["meteor_el_deg"]
    )
    meteor_ratios = np.asarray(
        [
            rayleigh_ratio(
                wavelength,
                meteor_ray[None, :],
                sun_enu,
                geometry["sun_el_deg"],
            )[0]
            for wavelength in EFFECTIVE_WAVELENGTH_UM
        ]
    )

    sky_ratios = np.empty((len(FOV_GRID_DEG), 3))
    sky_azimuth = np.empty(len(FOV_GRID_DEG))
    sky_elevation = np.empty(len(FOV_GRID_DEG))
    for fov_index, fov in enumerate(FOV_GRID_DEG):
        rotation = camera_to_enu_rotation(
            image_shape,
            meteor_xy,
            fov,
            geometry["meteor_az_deg"],
            geometry["meteor_el_deg"],
        )
        sky_ray_camera = camera_ray(sky_xy[0], sky_xy[1], image_shape, fov)
        sky_ray_enu = rotation @ sky_ray_camera
        sky_azimuth[fov_index], sky_elevation[fov_index] = rays_to_azel(
            sky_ray_enu
        )
        for channel, wavelength in enumerate(EFFECTIVE_WAVELENGTH_UM):
            sky_ratios[fov_index, channel] = rayleigh_ratio(
                wavelength,
                sky_ray_enu[None, :],
                sun_enu,
                geometry["sun_el_deg"],
            )[0]

    map_fov = REFERENCE_HORIZONTAL_FOV_DEG
    map_x = np.arange(0, image_shape[1], MAP_STEP_PX)
    map_y = np.arange(0, image_shape[0], MAP_STEP_PX)
    xx, yy = np.meshgrid(map_x, map_y)
    map_rotation = camera_to_enu_rotation(
        image_shape,
        meteor_xy,
        map_fov,
        geometry["meteor_az_deg"],
        geometry["meteor_el_deg"],
    )
    map_camera_rays = camera_ray(xx, yy, image_shape, map_fov)
    map_enu_rays = np.einsum("ij,...j->...i", map_rotation, map_camera_rays)
    map_azimuth, map_elevation = rays_to_azel(map_enu_rays)
    map_ratio = rayleigh_ratio(
        MAP_WAVELENGTH_UM,
        map_enu_rays,
        sun_enu,
        geometry["sun_el_deg"],
    )

    reference_index = int(np.argmin(np.abs(FOV_GRID_DEG - map_fov)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT, "w") as h5:
        h5.attrs["schema"] = "new_england_bolide.rayleigh_reference_map.v1"
        h5.attrs["projection"] = "rectilinear"
        h5.attrs["roll_constraint"] = "image vertical aligned with local zenith at meteor"
        h5.attrs["map_horizontal_fov_deg"] = map_fov
        h5.attrs["map_wavelength_um"] = MAP_WAVELENGTH_UM
        h5.create_dataset("horizontal_fov_deg", data=FOV_GRID_DEG)
        h5.create_dataset("effective_wavelength_um", data=EFFECTIVE_WAVELENGTH_UM)
        h5.create_dataset("meteor_rayleigh_ratio_sr_inv", data=meteor_ratios)
        h5.create_dataset("sky_reference_rayleigh_ratio_sr_inv", data=sky_ratios)
        h5.create_dataset("sky_reference_azimuth_deg", data=sky_azimuth)
        h5.create_dataset("sky_reference_elevation_deg", data=sky_elevation)
        h5.create_dataset("map_x_px", data=map_x)
        h5.create_dataset("map_y_px", data=map_y)
        h5.create_dataset("map_azimuth_deg", data=map_azimuth, compression="gzip")
        h5.create_dataset("map_elevation_deg", data=map_elevation, compression="gzip")
        h5.create_dataset(
            "map_rayleigh_ratio_sr_inv", data=map_ratio, compression="gzip"
        )

    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    image_ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1])
    curve_ax = fig.add_subplot(grid[1, :])

    image_ax.imshow(image)
    image_ax.add_patch(
        Polygon(bolide_vertices, closed=True, fill=False, color="cyan", linewidth=2)
    )
    image_ax.add_patch(
        Polygon(sky_vertices, closed=True, fill=False, color="blue", linewidth=2)
    )
    image_ax.set_title("Mount Arlington source frame and manual regions")
    image_ax.set_xlabel("pixel x")
    image_ax.set_ylabel("pixel y")

    map_image = map_ax.pcolormesh(
        map_x,
        map_y,
        1e3 * map_ratio,
        shading="nearest",
        cmap="viridis",
    )
    map_ax.add_patch(
        Polygon(bolide_vertices, closed=True, fill=False, color="cyan", linewidth=2)
    )
    map_ax.add_patch(
        Polygon(sky_vertices, closed=True, fill=False, color="white", linewidth=2)
    )
    map_ax.set_ylim(image_shape[0], 0)
    map_ax.set_xlim(0, image_shape[1])
    map_ax.set_title(
        r"Modeled 540-nm Rayleigh sky, $120^\circ$ rectilinear FOV"
    )
    map_ax.set_xlabel("pixel x")
    map_ax.set_ylabel("pixel y")
    colorbar = fig.colorbar(map_image, ax=map_ax, pad=0.02)
    colorbar.set_label(
        r"$10^3\,L_{\rm sky}(540\,{\rm nm})/F_\odot$ (sr$^{-1}$)"
    )

    colors = ("C3", "C2", "C0")
    for channel, (name, color) in enumerate(
        zip(CHANNEL_NAMES, colors, strict=True)
    ):
        curve_ax.plot(
            FOV_GRID_DEG,
            1e3 * sky_ratios[:, channel],
            color=color,
            linewidth=2,
            label=f"selected sky region — {name}",
        )
        curve_ax.axhline(
            1e3 * meteor_ratios[channel],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        curve_ax.scatter(
            [map_fov],
            [1e3 * sky_ratios[reference_index, channel]],
            color=color,
            zorder=3,
        )
    curve_ax.set_xlabel("assumed horizontal field of view (deg)")
    curve_ax.set_ylabel(
        r"$10^3\,L_{\rm sky}/F_\odot$ (sr$^{-1}$)"
    )
    curve_ax.set_title(
        "Rayleigh brightness of selected sky patch; dashed lines are the "
        "meteor-direction values used by the current estimator"
    )
    curve_ax.grid(alpha=0.25)
    handles, labels = curve_ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.2)
    )
    labels.append("meteor direction used previously")
    curve_ax.legend(handles, labels, ncol=2, loc="lower right")
    ratios_at_reference = sky_ratios[reference_index] / meteor_ratios
    curve_ax.text(
        0.015,
        0.97,
        (
            r"At $120^\circ$: selected/current = "
            f"{ratios_at_reference[0]:.2f} R, "
            f"{ratios_at_reference[1]:.2f} G, "
            f"{ratios_at_reference[2]:.2f} B"
        ),
        transform=curve_ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.suptitle(
        "Mount Arlington Rayleigh-sky reference sensitivity to camera FOV"
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    print(f"Saved: {FIGURE}")
    print(f"Saved: {OUTPUT}")
    print(
        f"At {map_fov:.0f} deg FOV, selected sky az/el = "
        f"{sky_azimuth[reference_index]:.3f}/"
        f"{sky_elevation[reference_index]:.3f} deg"
    )
    for channel, name in enumerate(CHANNEL_NAMES):
        print(
            f"{name}: selected={sky_ratios[reference_index, channel]:.6f} sr^-1, "
            f"meteor_direction={meteor_ratios[channel]:.6f} sr^-1, "
            f"ratio={sky_ratios[reference_index, channel]/meteor_ratios[channel]:.4f}"
        )


if __name__ == "__main__":
    main()
