#!/usr/bin/env python3
"""Generate a Gaussian-corrected head-echo micro-Doppler appendix figure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_batch import smooth_power_db  # noqa: E402
from reports.generate_head_microdoppler import (  # noqa: E402
    FIG_DIR,
    STATIONS,
    STATION_LABELS,
)
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, notch_2d_doppler_repetition  # noqa: E402

DEFAULT_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code_common_grid.h5"
DEFAULT_FIT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_retained_fig3seed.csv"
OUT_FIG = FIG_DIR / "head_echo_microdoppler_corrected.png"
OUT_SUMMARY = ROOT / "reports" / "head_echo_microdoppler_corrected_summary.csv"


def interpolate_range_row(range_m: np.ndarray, image: np.ndarray, target_range_m: float) -> np.ndarray:
    if target_range_m <= range_m[0]:
        return image[0, :]
    if target_range_m >= range_m[-1]:
        return image[-1, :]
    hi = int(np.searchsorted(range_m, target_range_m))
    lo = hi - 1
    w = (target_range_m - range_m[lo]) / (range_m[hi] - range_m[lo])
    return (1.0 - w) * image[lo, :] + w * image[hi, :]


def load_fit_rows(path: Path, stations: list[str]) -> dict[str, list[dict]]:
    wanted = set(stations)
    rows = {station: [] for station in stations}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            station = row["station"]
            if station not in wanted:
                continue
            if row.get("success", "").lower() != "true":
                continue
            rows[station].append(row)
    for station in rows:
        rows[station].sort(key=lambda r: (float(r["time_unix"]), int(r["idx"])))
    return rows


def corrected_station(station: str, fit_rows: list[dict], group: h5py.Group, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    rows = []
    spectra = []
    times = []
    ranges_km = []
    fit_doppler_hz = []
    seed_doppler_hz = []
    model_doppler_hz = []
    fit_rel_hz = []
    rel_axis = None

    for fit_row in fit_rows:
        patch_key = f"patch_{int(fit_row['idx']):04d}"
        if patch_key not in group:
            rows.append(
                {
                    "station": station,
                    "time_unix": float(fit_row["time_unix"]),
                    "time_rel_s": float(fit_row["time_rel_s"]),
                    "success": False,
                    "error": f"missing cache group {patch_key}",
                }
            )
            continue
        patch_group = group[patch_key]
        if not bool(patch_group.attrs.get("valid", False)):
            continue
        patch_time = float(patch_group.attrs["time_unix"])
        fit_time = float(fit_row["time_unix"])
        fit_rel_doppler_hz = float(fit_row["fit_rel_doppler_hz"])
        fit_range_m = float(fit_row["fit_range_m"])

        range_m = patch_group["range_m"][()].astype(float)
        rel_freqs = patch_group["rel_freqs"][()].astype(float)
        raw_db = patch_group["raw_db"][()].astype(float)
        filtered_unsmoothed_db = notch_2d_doppler_repetition(raw_db, args.n_coh, args.notch_half_width)
        filtered_db = smooth_power_db(
            filtered_unsmoothed_db,
            args.smooth_range_pixels,
            args.smooth_frequency_pixels,
        )
        model = float(patch_group.attrs["model_doppler_hz"])
        seed_range_m = float(patch_group.attrs["seed_range_m"])
        seed_doppler_value_hz = float(patch_group.attrs["picked_doppler_hz"])

        row = interpolate_range_row(range_m, filtered_db, fit_range_m)
        spectra.append(np.interp(rel_freqs, rel_freqs - fit_rel_doppler_hz, row, left=np.nan, right=np.nan))
        times.append(fit_time)
        ranges_km.append(fit_range_m / 1000.0)
        fit_doppler_hz.append(float(fit_row["fit_doppler_hz"]))
        seed_doppler_hz.append(seed_doppler_value_hz)
        model_doppler_hz.append(model)
        fit_rel_hz.append(fit_rel_doppler_hz)
        rel_axis = rel_freqs.copy()
        rows.append(
            {
                "station": station,
                "time_unix": fit_time,
                "time_rel_s": fit_time - EVENT_T0,
                "patch_time_unix": patch_time,
                "patch_time_rel_s": patch_time - EVENT_T0,
                "patch_time_offset_s": patch_time - float(fit_row.get("patch_time_unix", patch_time)),
                "smooth_range_pixels": float(args.smooth_range_pixels),
                "smooth_frequency_pixels": float(args.smooth_frequency_pixels),
                "code_period_s": float(args.code_period_s),
                "seed_range_m": seed_range_m,
                "fit_range_m": fit_range_m,
                "range_correction_m": fit_range_m - seed_range_m,
                "model_doppler_hz": model,
                "seed_doppler_hz": seed_doppler_value_hz,
                "fit_doppler_hz": float(fit_row["fit_doppler_hz"]),
                "doppler_correction_hz": float(fit_row["doppler_correction_hz"]),
                "fit_rel_doppler_hz": fit_rel_doppler_hz,
                "sigma_range_m": float(fit_row["sigma_range_m"]),
                "sigma_doppler_hz": float(fit_row["sigma_doppler_hz"]),
                "range_doppler_correlation": float(fit_row.get("range_doppler_correlation", 0.0)),
                "amplitude_db": float(fit_row["amplitude_db"]),
                "cost": float(fit_row["cost"]),
                "success": True,
                "error": "",
            }
        )

    if not spectra:
        raise ValueError(f"No corrected spectra for {station}")
    spec_db = np.asarray(spectra, dtype=float)
    spec_db -= np.nanmedian(spec_db)
    return (
        {
            "station": station,
            "times": np.asarray(times),
            "time_rel": np.asarray(times) - EVENT_T0,
            "ranges_km": np.asarray(ranges_km),
            "fit_dopplers_hz": np.asarray(fit_doppler_hz),
            "seed_dopplers_hz": np.asarray(seed_doppler_hz),
            "model_dopplers_hz": np.asarray(model_doppler_hz),
            "fit_rel_hz": np.asarray(fit_rel_hz),
            "rel_freqs": rel_axis,
            "spec_db": spec_db,
        },
        rows,
    )


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "station",
        "time_unix",
        "time_rel_s",
        "patch_time_unix",
        "patch_time_rel_s",
        "patch_time_offset_s",
        "smooth_range_pixels",
        "smooth_frequency_pixels",
        "code_period_s",
        "seed_range_m",
        "fit_range_m",
        "range_correction_m",
        "model_doppler_hz",
        "seed_doppler_hz",
        "fit_doppler_hz",
        "doppler_correction_hz",
        "fit_rel_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "range_doppler_correlation",
        "amplitude_db",
        "cost",
        "success",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_corrected(results: list[dict], out_fig: Path, vmin_db: float, vmax_db: float) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    mappable = None
    for ax, res in zip(axes, results):
        ax.set_facecolor("black")
        t = res["time_rel"]
        f = res["rel_freqs"]
        z = res["spec_db"].T
        mappable = ax.pcolormesh(t, f, z, shading="auto", cmap="inferno", vmin=vmin_db, vmax=vmax_db)
        ax.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.65)
        ax.set_ylabel("$f-f_{D,Gaussian}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo")
        ax.text(
            0.01,
            0.93,
            f"corrected range {np.nanmedian(res['ranges_km']):.0f} km, fitted Doppler {np.nanmedian(res['fit_dopplers_hz']):.0f} Hz",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, pad=0.012)
        cb.set_label("Relative matched-filter power (dB)")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    p.add_argument("--summary", type=Path, default=OUT_SUMMARY)
    p.add_argument("--out-fig", type=Path, default=OUT_FIG)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=16.0)
    p.add_argument("--code-period-s", type=float, default=0.01)
    p.add_argument("--vmin-db", type=float, default=0.0)
    p.add_argument("--vmax-db", type=float, default=6.0)
    args = p.parse_args()

    fit_rows = load_fit_rows(args.fit_summary, args.stations)
    results = []
    rows = []
    with h5py.File(args.cache, "r") as h:
        for station in args.stations:
            print(f"{station}: corrected micro-Doppler from {len(fit_rows[station])} validated fits", flush=True)
            result, station_rows = corrected_station(station, fit_rows[station], h[station], args)
            results.append(result)
            rows.extend(station_rows)
    write_summary(args.summary, rows)

    out = plot_corrected(results, args.out_fig, args.vmin_db, args.vmax_db)
    print(out)
    print(args.summary)


if __name__ == "__main__":
    main()
