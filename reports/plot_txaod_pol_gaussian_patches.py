#!/usr/bin/env python3
"""Fit and plot AoD-steered, polarization-optimized head-echo RD patches."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import (  # noqa: E402
    correlated_ellipse_points,
    fit_gaussian_patch,
    gaussian2d,
    gaussian_plot_window,
    smooth_power_db,
)
from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0  # noqa: E402

DEFAULT_CACHE = (
    ROOT
    / "products"
    / "head_echo_gaussian_patches"
    / "head_echo_gaussian_patches_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_polgrid_txaod5_retained_ext0p5.h5"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "reports"
    / "figures"
    / "head_echo_gaussian_fits_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_polgrid_txaod5_retained_ext0p5"
)
DEFAULT_SUMMARY = (
    ROOT
    / "reports"
    / "head_echo_gaussian_fit_summary_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_polgrid_txaod5_retained_ext0p5.csv"
)

FIELDNAMES = [
    "station",
    "idx",
    "time_unix",
    "time_rel_s",
    "patch_time_unix",
    "patch_time_rel_s",
    "seed_range_m",
    "fit_range_m",
    "range_correction_m",
    "seed_doppler_hz",
    "fit_doppler_hz",
    "doppler_correction_hz",
    "model_doppler_hz",
    "fit_rel_doppler_hz",
    "sigma_range_m",
    "sigma_doppler_hz",
    "range_doppler_correlation",
    "amplitude_db",
    "cost",
    "success",
    "error",
    "n_coh",
    "n_incoh",
    "step_codes",
    "notch_half_width",
    "smooth_range_pixels",
    "smooth_frequency_pixels",
    "code_period_s",
    "figure",
]


def signed_time_label(time_rel_s: float) -> str:
    prefix = "tp" if time_rel_s >= 0.0 else "tm"
    return f"{prefix}{abs(time_rel_s):06.3f}s"


def root_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def plot_patch(out: Path, station: str, attrs: dict, range_km: np.ndarray, rel_hz: np.ndarray, raw_db: np.ndarray, smoothed_db: np.ndarray, fit: dict) -> None:
    seed_range_km = float(attrs["seed_range_m"]) / 1000.0
    model_doppler_hz = float(attrs["model_doppler_hz"])
    seed_rel_hz = float(attrs["picked_doppler_hz"]) - model_doppler_hz
    fit_range_km = float(fit["range_km"])
    fit_rel_hz = float(fit["rel_doppler_hz"])
    sig_r = float(fit["sigma_range_km"])
    sig_f = float(fit["sigma_doppler_hz"])
    rho = float(fit.get("range_doppler_correlation", 0.0))
    rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
    model_db = gaussian2d(
        np.array(
            [
                float(fit["offset_db"]),
                float(fit["amplitude_db"]),
                fit_range_km,
                fit_rel_hz,
                sig_r,
                sig_f,
                rho,
            ],
            dtype=float,
        ),
        rr,
        ff,
    )
    x_min, x_max, y_min, y_max = gaussian_plot_window(seed_range_km, seed_rel_hz, fit_range_km, fit_rel_hz)
    show = (
        (range_km[:, None] >= y_min)
        & (range_km[:, None] <= y_max)
        & (rel_hz[None, :] >= x_min)
        & (rel_hz[None, :] <= x_max)
    )
    ell_x, ell_y = correlated_ellipse_points(fit_range_km, fit_rel_hz, sig_r, sig_f, rho)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("AoD + pol matched filter", raw_db, "inferno", "dB"),
        ("Range-smoothed Gaussian fit", smoothed_db, "inferno", "dB"),
    ]
    for ax, (title, img, cmap, label) in zip(axes, panels, strict=True):
        finite = img[show & np.isfinite(img)]
        if finite.size:
            vmin = float(np.nanpercentile(finite, 3.0))
            vmax = float(np.nanpercentile(finite, 99.6))
        else:
            vmin, vmax = 0.0, 1.0
        mesh = ax.pcolormesh(rel_hz, range_km, img, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot([seed_rel_hz, fit_rel_hz], [seed_range_km, fit_range_km], color="white", lw=0.9, alpha=0.9)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=9, mew=1.4, color="#9be7ff", linestyle="none")
        ax.plot(fit_rel_hz, fit_range_km, marker="x", ms=8, mew=1.4, color="#42ff9f", linestyle="none")
        ax.plot(ell_x, ell_y, color="#42ff9f", lw=1.2)
        ax.contour(
            rel_hz,
            range_km,
            model_db,
            levels=[
                float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-2.0),
                float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-0.5),
            ],
            colors=["#42ff9f", "#d8ffe8"],
            linewidths=[0.8, 1.0],
        )
        ax.set_title(title)
        ax.set_xlabel("$f-f_{D,model}$ (Hz)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(color="white", alpha=0.12, lw=0.5)
        cb = fig.colorbar(mesh, ax=ax, pad=0.012)
        cb.set_label(label)
    axes[0].set_ylabel("Bistatic range (km)")
    dr_m = (fit_range_km - seed_range_km) * 1000.0
    df_hz = fit_rel_hz - seed_rel_hz
    abs_doppler = model_doppler_hz + fit_rel_hz
    gamma_text = ""
    if "polarization_gamma" in attrs:
        gamma = attrs["polarization_gamma"]
        gamma_text = f"; gamma=[{gamma[0]:.2f}, {gamma[1]:.2f}]"
    fig.suptitle(
        f"{STATION_LABELS.get(station, station)} at t={float(attrs['time_rel_s']):.3f} s: "
        f"Gaussian correction {dr_m:+.0f} m, {df_hz:+.1f} Hz; refined Doppler {abs_doppler:.1f} Hz"
        f"{gamma_text}",
        y=1.03,
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def row_from_fit(station: str, patch_name: str, attrs: dict, fit: dict, fig_path: Path, args: argparse.Namespace) -> dict:
    idx = int(attrs["idx"])
    time_unix = float(attrs["time_unix"])
    time_rel_s = float(attrs["time_rel_s"])
    seed_range_m = float(attrs["seed_range_m"])
    seed_doppler_hz = float(attrs["picked_doppler_hz"])
    model_doppler_hz = float(attrs["model_doppler_hz"])
    fit_range_m = float(fit["range_km"]) * 1000.0
    fit_doppler_hz = model_doppler_hz + float(fit["rel_doppler_hz"])
    return {
        "station": station,
        "idx": idx,
        "time_unix": time_unix,
        "time_rel_s": time_rel_s,
        "patch_time_unix": time_unix,
        "patch_time_rel_s": time_rel_s,
        "seed_range_m": seed_range_m,
        "fit_range_m": fit_range_m,
        "range_correction_m": fit_range_m - seed_range_m,
        "seed_doppler_hz": seed_doppler_hz,
        "fit_doppler_hz": fit_doppler_hz,
        "doppler_correction_hz": fit_doppler_hz - seed_doppler_hz,
        "model_doppler_hz": model_doppler_hz,
        "fit_rel_doppler_hz": float(fit["rel_doppler_hz"]),
        "sigma_range_m": float(fit["sigma_range_km"]) * 1000.0,
        "sigma_doppler_hz": float(fit["sigma_doppler_hz"]),
        "range_doppler_correlation": float(fit.get("range_doppler_correlation", 0.0)),
        "amplitude_db": float(fit["amplitude_db"]),
        "cost": float(fit["cost"]),
        "success": bool(fit["success"]),
        "error": "",
        "n_coh": int(attrs.get("n_coh", -1)),
        "n_incoh": int(attrs.get("n_incoh", -1)),
        "step_codes": int(attrs.get("step_codes", -1)),
        "notch_half_width": 0,
        "smooth_range_pixels": float(args.smooth_range_pixels),
        "smooth_frequency_pixels": float(args.smooth_frequency_pixels),
        "code_period_s": 0.01,
        "figure": root_relative(fig_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--stations", nargs="+", default=["beverly", "clinton", "pomfret"])
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=16.0)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    rows: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as h:
        for station in args.stations:
            g = h[station]
            patch_names = sorted(g.keys())
            if args.limit is not None:
                patch_names = patch_names[: args.limit]
            station_dir = args.out_dir / station
            station_dir.mkdir(parents=True, exist_ok=True)
            for n_done, patch_name in enumerate(patch_names, start=1):
                pgrp = g[patch_name]
                attrs = dict(pgrp.attrs.items())
                idx = int(attrs["idx"])
                time_rel_s = float(attrs["time_rel_s"])
                fig_path = station_dir / f"{station}_{idx:03d}_{signed_time_label(time_rel_s)}.png"
                try:
                    if not bool(attrs.get("valid", False)):
                        raise ValueError("invalid patch")
                    range_km = pgrp["range_m"][()].astype(float) / 1000.0
                    rel_hz = pgrp["rel_freqs"][()].astype(float)
                    raw_db = pgrp["raw_db"][()].astype(float)
                    smoothed_db = smooth_power_db(raw_db, args.smooth_range_pixels, args.smooth_frequency_pixels)
                    seed_range_km = float(attrs["seed_range_m"]) / 1000.0
                    seed_rel_hz = float(attrs["picked_doppler_hz"]) - float(attrs["model_doppler_hz"])
                    fit = fit_gaussian_patch(
                        range_km,
                        rel_hz,
                        smoothed_db,
                        seed_range_km,
                        seed_rel_hz,
                        args.search_range_half_width_km,
                        args.search_doppler_half_width_hz,
                        args.fit_range_half_width_km,
                        args.fit_doppler_half_width_hz,
                    )
                    if "polarization_gamma" in pgrp:
                        attrs["polarization_gamma"] = pgrp["polarization_gamma"][()]
                    plot_patch(fig_path, station, attrs, range_km, rel_hz, raw_db, smoothed_db, fit)
                    row = row_from_fit(station, patch_name, attrs, fit, fig_path, args)
                except Exception as exc:  # keep the CSV aligned with the existing workflow
                    row = {
                        "station": station,
                        "idx": idx,
                        "time_unix": float(attrs.get("time_unix", np.nan)),
                        "time_rel_s": float(attrs.get("time_rel_s", np.nan)),
                        "patch_time_unix": float(attrs.get("time_unix", np.nan)),
                        "patch_time_rel_s": float(attrs.get("time_rel_s", np.nan)),
                        "seed_range_m": float(attrs.get("seed_range_m", np.nan)),
                        "fit_range_m": np.nan,
                        "range_correction_m": np.nan,
                        "seed_doppler_hz": float(attrs.get("picked_doppler_hz", np.nan)),
                        "fit_doppler_hz": np.nan,
                        "doppler_correction_hz": np.nan,
                        "model_doppler_hz": float(attrs.get("model_doppler_hz", np.nan)),
                        "fit_rel_doppler_hz": np.nan,
                        "sigma_range_m": np.nan,
                        "sigma_doppler_hz": np.nan,
                        "range_doppler_correlation": np.nan,
                        "amplitude_db": np.nan,
                        "cost": np.nan,
                        "success": False,
                        "error": str(exc),
                        "n_coh": int(attrs.get("n_coh", -1)),
                        "n_incoh": int(attrs.get("n_incoh", -1)),
                        "step_codes": int(attrs.get("step_codes", -1)),
                        "notch_half_width": 0,
                        "smooth_range_pixels": float(args.smooth_range_pixels),
                        "smooth_frequency_pixels": float(args.smooth_frequency_pixels),
                        "code_period_s": 0.01,
                        "figure": root_relative(fig_path),
                    }
                rows.append(row)
                if n_done == 1 or n_done == len(patch_names) or n_done % 25 == 0:
                    print(f"{station}: {n_done}/{len(patch_names)}", flush=True)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(args.summary, flush=True)
    print(args.out_dir, flush=True)


if __name__ == "__main__":
    main()
