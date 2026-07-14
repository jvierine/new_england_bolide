#!/usr/bin/env python3
"""Replace the closest-NEA overlay with known meteorite-fall tracks.

This helper preserves a published fit-specific raster when its Monte Carlo HDF5
product is unavailable locally. The axes bounds and pixel rectangle must match
the source figure; the defaults describe the committed v0.3.1 Figure 6 raster.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from plot_orbit_xy import kepler_to_xy, read_meteorite_fall_orbits


DEFAULT_CATALOG = Path("reports/meteorite_orbits_info_orbits.csv")
DEFAULT_OUT = Path("reports/figures/orbit_xy_bolide.png")


def data_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    axes_pixels: tuple[float, float, float, float],
    limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = axes_pixels
    lo, hi = limits
    px = left + (x - lo) * (right - left) / (hi - lo)
    py = bottom - (y - lo) * (bottom - top) / (hi - lo)
    return px, py


def remove_closest_nea(base: Image.Image) -> Image.Image:
    """Remove the green closest-NEA curve, marker, and two legend entries."""
    rgb = np.asarray(base.convert("RGB")).copy()
    red = rgb[..., 0].astype(float)
    green = rgb[..., 1].astype(float)
    blue = rgb[..., 2].astype(float)
    mask = (green > 70) & (green > 1.08 * red) & (green > 1.02 * blue)
    mask = ndimage.binary_dilation(mask, iterations=6)
    _distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)
    rgb[mask] = rgb[tuple(axis[mask] for axis in indices)]

    # The original legend is opaque white.  Clear the two former NEA rows so
    # that a meteorite-fall key can be drawn in their place.
    rgb[70:205, 760:1668] = 255
    return Image.fromarray(rgb).convert("RGBA")


def overlay(args: argparse.Namespace) -> int:
    base = remove_closest_nea(Image.open(args.base))
    width, height = base.size
    tracks = read_meteorite_fall_orbits(args.catalog)
    if not tracks:
        raise RuntimeError(f"No complete meteorite-fall orbits found in {args.catalog}")

    fig = plt.figure(figsize=(width / args.dpi, height / args.dpi), dpi=args.dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.axis("off")
    for kep in tracks:
        x, y = kepler_to_xy(kep, n_points=720)
        px, py = data_to_pixels(x, y, tuple(args.axes_pixels), tuple(args.limits))
        ax.plot(px, py, color="#3f6f8a", alpha=0.34, lw=0.95, solid_capstyle="round")
    ax.plot(
        [810, 875],
        [105, 105],
        color="#3f6f8a",
        alpha=0.95,
        lw=3.0,
        solid_capstyle="round",
    )
    ax.text(
        905,
        112,
        f"Known meteorite falls ({len(tracks)})",
        color="black",
        fontsize=35,
        ha="left",
        va="center",
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=args.dpi, transparent=True, pad_inches=0)
    buffer.seek(0)
    rendered = np.asarray(Image.open(buffer).convert("RGBA")).copy()
    plt.close(fig)
    overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    copy_height = min(height, rendered.shape[0])
    copy_width = min(width, rendered.shape[1])
    overlay_rgba[:copy_height, :copy_width] = rendered[:copy_height, :copy_width]

    # Keep the Keplerian-element inset intact.  Meteorite tracks and their
    # legend key are allowed in the published legend region.
    yy, xx = np.mgrid[:height, :width]
    inset_x_fade = np.clip((xx - 700.0) / 200.0, 0.0, 1.0)
    inset_y_fade = np.clip((1150.0 - yy) / 200.0, 0.0, 1.0)
    inset_keep = np.maximum(inset_x_fade, inset_y_fade)
    overlay_rgba[..., 3] = np.rint(overlay_rgba[..., 3] * inset_keep).astype(np.uint8)

    composed = Image.alpha_composite(base, Image.fromarray(overlay_rgba))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(args.out, dpi=(240, 240))
    print(f"Wrote {args.out} with {len(tracks)} complete meteorite-fall tracks")
    return len(tracks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Unmodified fit-specific orbit-XY raster.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--axes-pixels", type=float, nargs=4, default=(177.0, 40.0, 1906.0, 1769.0))
    parser.add_argument("--limits", type=float, nargs=2, default=(-8.0, 8.0))
    parser.add_argument("--dpi", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    overlay(parse_args())
