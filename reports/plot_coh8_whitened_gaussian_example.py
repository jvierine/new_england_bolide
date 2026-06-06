#!/usr/bin/env python3
"""Plot the Clinton eight-code covariance-whitened Gaussian example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import (  # noqa: E402
    correlated_ellipse_points,
    fit_gaussian_patch,
    gaussian2d,
    smooth_power_db,
)

DEFAULT_CACHE = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "cache"
    / "coh8_whitened_match_clinton_t040_shrink002.h5"
)
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "figures"
    / "coh8_whitened_gaussian_example_clinton_t040.png"
)


def db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-30))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=8.0)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    args = p.parse_args()

    with h5py.File(args.cache, "r") as h:
        station = str(h.attrs["station"])
        time_rel = float(h.attrs["time_rel"])
        seed_range_km = float(h.attrs["seed_range_m"]) / 1000.0
        seed_rel_hz = float(h.attrs["seed_doppler_hz"]) - float(h.attrs["model_doppler_hz"])
        model_doppler_hz = float(h.attrs["model_doppler_hz"])
        range_km = h["range_m"][()] / 1000.0
        rel_hz = h["rel_freqs"][()]
        raw_db = db(h["raw_power"][()])
        whitened_db = db(h["whitened_power"][()])

    whitened_smoothed_db = smooth_power_db(
        whitened_db,
        args.smooth_range_pixels,
        args.smooth_frequency_pixels,
    )
    fit = fit_gaussian_patch(
        range_km,
        rel_hz,
        whitened_smoothed_db,
        seed_range_km,
        seed_rel_hz,
        args.search_range_half_width_km,
        args.search_doppler_half_width_hz,
        args.fit_range_half_width_km,
        args.fit_doppler_half_width_hz,
    )

    fit_range_km = float(fit["range_km"])
    fit_rel_hz = float(fit["rel_doppler_hz"])
    sig_r = float(fit["sigma_range_km"])
    sig_f = float(fit["sigma_doppler_hz"])
    rho = float(fit["range_doppler_correlation"])
    rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
    fit_model = gaussian2d(
        np.array(
            [
                float(fit["offset_db"]),
                float(fit["amplitude_db"]),
                fit_range_km,
                fit_rel_hz,
                sig_r,
                sig_f,
                rho,
            ]
        ),
        rr,
        ff,
    )
    ell_x, ell_y = correlated_ellipse_points(fit_range_km, fit_rel_hz, sig_r, sig_f, rho)

    x_pad = 650.0
    y_pad = 14.0
    x_min = min(seed_rel_hz, fit_rel_hz) - x_pad
    x_max = max(seed_rel_hz, fit_rel_hz) + x_pad
    y_min = min(seed_range_km, fit_range_km) - y_pad
    y_max = max(seed_range_km, fit_range_km) + y_pad

    images = [
        ("Raw matched filter", raw_db),
        ("Covariance-whitened MF", whitened_db),
        ("Whitened + smoothed", whitened_smoothed_db),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, img) in zip(axes, images, strict=True):
        show = (
            (range_km[:, None] >= y_min)
            & (range_km[:, None] <= y_max)
            & (rel_hz[None, :] >= x_min)
            & (rel_hz[None, :] <= x_max)
        )
        finite = img[show & np.isfinite(img)]
        vmin = float(np.nanpercentile(finite, 3.0))
        vmax = float(np.nanpercentile(finite, 99.6))
        mappable = ax.pcolormesh(rel_hz, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot([seed_rel_hz, fit_rel_hz], [seed_range_km, fit_range_km], color="white", lw=0.9, alpha=0.9)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=9, mew=1.4, color="#9be7ff", linestyle="none")
        ax.plot(fit_rel_hz, fit_range_km, marker="x", ms=8, mew=1.4, color="#42ff9f", linestyle="none")
        ax.plot(ell_x, ell_y, color="#42ff9f", lw=1.2)
        ax.contour(
            rel_hz,
            range_km,
            fit_model,
            levels=[
                float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-2.0),
                float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-0.5),
            ],
            colors=["#42ff9f", "#d8ffe8"],
            linewidths=[0.8, 1.0],
        )
        ax.set_title(title)
        ax.set_xlabel(r"$f-f_{D,model}$ (Hz)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(color="white", alpha=0.12, lw=0.5)
        cb = fig.colorbar(mappable, ax=ax, pad=0.012)
        cb.set_label("dB")
    axes[0].set_ylabel("Bistatic range (km)")

    dr_m = (fit_range_km - seed_range_km) * 1000.0
    df_hz = fit_rel_hz - seed_rel_hz
    abs_doppler = model_doppler_hz + fit_rel_hz
    fig.suptitle(
        f"{station} at t={time_rel:.3f} s: covariance-whitened Gaussian correction "
        f"{dr_m:+.0f} m, {df_hz:+.1f} Hz; refined Doppler {abs_doppler:.1f} Hz",
        y=1.03,
        fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    print(
        f"seed range={seed_range_km:.3f} km seed residual={seed_rel_hz:.1f} Hz "
        f"fit range={fit_range_km:.3f} km fit residual={fit_rel_hz:.1f} Hz"
    )


if __name__ == "__main__":
    main()
