#!/usr/bin/env python3
"""Overlay known meteorite-fall tracks on an existing orbit-XY figure.

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


def overlay(args: argparse.Namespace) -> int:
    base = Image.open(args.base).convert("RGBA")
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
        ax.plot(px, py, color="#4f86a6", alpha=0.22, lw=0.9, solid_capstyle="round")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=args.dpi, transparent=True, pad_inches=0)
    buffer.seek(0)
    rendered = np.asarray(Image.open(buffer).convert("RGBA")).copy()
    plt.close(fig)
    overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    copy_height = min(height, rendered.shape[0])
    copy_width = min(width, rendered.shape[1])
    overlay_rgba[:copy_height, :copy_width] = rendered[:copy_height, :copy_width]

    # Keep the published legend and Keplerian-element inset completely intact.
    overlay_rgba[45:395, 390:1680, 3] = 0
    yy, xx = np.mgrid[:height, :width]
    inset_x_fade = np.clip((xx - 700.0) / 200.0, 0.0, 1.0)
    inset_y_fade = np.clip((1150.0 - yy) / 200.0, 0.0, 1.0)
    inset_keep = np.maximum(inset_x_fade, inset_y_fade)
    overlay_rgba[..., 3] = np.rint(overlay_rgba[..., 3] * inset_keep).astype(np.uint8)

    composed = Image.alpha_composite(base, Image.fromarray(overlay_rgba, mode="RGBA"))
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
