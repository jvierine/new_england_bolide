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

from reports.generate_head_echo_gaussian_example import fit_gaussian_patch  # noqa: E402
from reports.generate_head_microdoppler import (  # noqa: E402
    FIG_DIR,
    STATIONS,
    STATION_LABELS,
    load_doppler_model,
    load_station_observations,
)
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, notch_2d_doppler_repetition  # noqa: E402

DEFAULT_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code_common_grid.h5"
OUT_FIG = FIG_DIR / "head_echo_microdoppler_corrected.png"
OUT_SUMMARY = ROOT / "reports" / "head_echo_microdoppler_corrected_summary.csv"


def patch_keys(group: h5py.Group) -> list[str]:
    return sorted(key for key in group.keys() if key.startswith("patch_"))


def nearest_patch(group: h5py.Group, unix_time: float) -> h5py.Group:
    keys = patch_keys(group)
    times = np.asarray([float(group[key].attrs["time_unix"]) for key in keys])
    return group[keys[int(np.argmin(np.abs(times - unix_time)))]]


def interpolate_range_row(range_m: np.ndarray, image: np.ndarray, target_range_m: float) -> np.ndarray:
    if target_range_m <= range_m[0]:
        return image[0, :]
    if target_range_m >= range_m[-1]:
        return image[-1, :]
    hi = int(np.searchsorted(range_m, target_range_m))
    lo = hi - 1
    w = (target_range_m - range_m[lo]) / (range_m[hi] - range_m[lo])
    return (1.0 - w) * image[lo, :] + w * image[hi, :]


def corrected_station(station: str, observations: list[dict], group: h5py.Group, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    rows = []
    spectra = []
    times = []
    ranges_km = []
    fit_doppler_hz = []
    seed_doppler_hz = []
    model_doppler_hz = []
    fit_rel_hz = []
    rel_axis = None

    for obs in observations:
        patch_group = nearest_patch(group, float(obs["unix_time"]))
        if not bool(patch_group.attrs.get("valid", False)):
            continue
        patch_time = float(patch_group.attrs["time_unix"])
        if abs(patch_time - float(obs["unix_time"])) > args.max_time_mismatch_s:
            continue

        range_m = patch_group["range_m"][()].astype(float)
        rel_freqs = patch_group["rel_freqs"][()].astype(float)
        raw_db = patch_group["raw_db"][()].astype(float)
        filtered_db = notch_2d_doppler_repetition(raw_db, args.n_coh, args.notch_half_width)
        model = float(patch_group.attrs["model_doppler_hz"])
        seed_range_m = float(patch_group.attrs["seed_range_m"])
        seed_doppler_value_hz = float(patch_group.attrs["picked_doppler_hz"])
        seed_range_km = seed_range_m / 1000.0
        seed_rel_hz = seed_doppler_value_hz - model

        try:
            fit = fit_gaussian_patch(
                range_m / 1000.0,
                rel_freqs,
                filtered_db,
                seed_range_km,
                seed_rel_hz,
                args.search_range_half_width_km,
                args.search_doppler_half_width_hz,
                args.fit_range_half_width_km,
                args.fit_doppler_half_width_hz,
            )
        except Exception as exc:
            rows.append(
                {
                    "station": station,
                    "time_unix": obs["unix_time"],
                    "time_rel_s": float(obs["unix_time"]) - EVENT_T0,
                    "success": False,
                    "error": str(exc),
                }
            )
            continue

        fit_range_m = 1000.0 * fit["range_km"]
        row = interpolate_range_row(range_m, filtered_db, fit_range_m)
        spectra.append(np.interp(rel_freqs, rel_freqs - fit["rel_doppler_hz"], row, left=np.nan, right=np.nan))
        times.append(float(obs["unix_time"]))
        ranges_km.append(fit["range_km"])
        fit_doppler_hz.append(model + fit["rel_doppler_hz"])
        seed_doppler_hz.append(seed_doppler_value_hz)
        model_doppler_hz.append(model)
        fit_rel_hz.append(fit["rel_doppler_hz"])
        rel_axis = rel_freqs.copy()
        rows.append(
            {
                "station": station,
                "time_unix": float(obs["unix_time"]),
                "time_rel_s": float(obs["unix_time"]) - EVENT_T0,
                "patch_time_unix": patch_time,
                "patch_time_offset_s": patch_time - float(obs["unix_time"]),
                "seed_range_m": seed_range_m,
                "fit_range_m": fit_range_m,
                "range_correction_m": fit_range_m - seed_range_m,
                "model_doppler_hz": model,
                "seed_doppler_hz": seed_doppler_value_hz,
                "fit_doppler_hz": model + fit["rel_doppler_hz"],
                "doppler_correction_hz": model + fit["rel_doppler_hz"] - seed_doppler_value_hz,
                "fit_rel_doppler_hz": fit["rel_doppler_hz"],
                "sigma_range_m": 1000.0 * fit["sigma_range_km"],
                "sigma_doppler_hz": fit["sigma_doppler_hz"],
                "amplitude_db": fit["amplitude_db"],
                "cost": fit["cost"],
                "success": bool(fit.get("success", False)),
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
        "patch_time_offset_s",
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


def plot_corrected(results: list[dict], out_fig: Path) -> Path:
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
        mappable = ax.pcolormesh(t, f, z, shading="auto", cmap="inferno", vmin=0.0, vmax=18.0)
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
    p.add_argument("--summary", type=Path, default=OUT_SUMMARY)
    p.add_argument("--out-fig", type=Path, default=OUT_FIG)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--max-time-mismatch-s", type=float, default=0.006)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    args = p.parse_args()

    _payload, observations = load_station_observations()
    results = []
    rows = []
    with h5py.File(args.cache, "r") as h:
        for station in args.stations:
            print(f"{station}: corrected micro-Doppler from {len(observations[station])} retained observations", flush=True)
            result, station_rows = corrected_station(station, observations[station], h[station], args)
            results.append(result)
            rows.extend(station_rows)
    write_summary(args.summary, rows)

    out = plot_corrected(results, args.out_fig)
    print(out)
    print(args.summary)


if __name__ == "__main__":
    main()
