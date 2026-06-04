#!/usr/bin/env python3
"""Fit cached head-echo matched-filter patches with a 2-D Gaussian."""

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

from reports.generate_head_echo_gaussian_batch import display_path, plot_patch, safe_name  # noqa: E402
from reports.generate_head_echo_gaussian_example import fit_gaussian_patch  # noqa: E402
from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, notch_2d_doppler_repetition  # noqa: E402

DEFAULT_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code.h5"
DEFAULT_OUT_DIR = ROOT / "reports" / "figures" / "head_echo_gaussian_fits"
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary.csv"


def fit_cached_patch(station: str, patch_group: h5py.Group, args: argparse.Namespace) -> tuple[dict, dict | None, dict | None, str | None]:
    obs = {
        "station": station,
        "unix_time": float(patch_group.attrs["time_unix"]),
        "range_m": float(patch_group.attrs["seed_range_m"]),
        "doppler_hz": float(patch_group.attrs["picked_doppler_hz"]),
    }
    if not bool(patch_group.attrs.get("valid", False)):
        return obs, None, None, "invalid cached patch"
    patch = {
        "idx": int(patch_group.attrs["idx"]),
        "time": float(patch_group.attrs["time_unix"]),
        "seed_range_m": float(patch_group.attrs["seed_range_m"]),
        "picked_doppler_hz": float(patch_group.attrs["picked_doppler_hz"]),
        "model_doppler_hz": float(patch_group.attrs["model_doppler_hz"]),
        "range_m": patch_group["range_m"][()].astype(float),
        "rel_freqs": patch_group["rel_freqs"][()].astype(float),
        "raw_db": patch_group["raw_db"][()].astype(float),
    }
    patch["filtered_db"] = notch_2d_doppler_repetition(patch["raw_db"], args.n_coh, args.notch_half_width)
    try:
        seed_range_km = obs["range_m"] / 1000.0
        seed_rel_hz = obs["doppler_hz"] - patch["model_doppler_hz"]
        fit = fit_gaussian_patch(
            patch["range_m"] / 1000.0,
            patch["rel_freqs"],
            patch["filtered_db"],
            seed_range_km,
            seed_rel_hz,
            args.search_range_half_width_km,
            args.search_doppler_half_width_hz,
            args.fit_range_half_width_km,
            args.fit_doppler_half_width_hz,
        )
        fit["absolute_doppler_hz"] = patch["model_doppler_hz"] + fit["rel_doppler_hz"]
        fit["range_m"] = fit["range_km"] * 1000.0
        fit["range_correction_m"] = fit["range_m"] - obs["range_m"]
        fit["doppler_correction_hz"] = fit["absolute_doppler_hz"] - obs["doppler_hz"]
        return obs, patch, fit, None
    except Exception as exc:
        return obs, patch, None, str(exc)


def summary_row(station: str, obs: dict, patch: dict | None, fit: dict | None, error: str | None, fig_path: Path, settings: dict) -> dict:
    row = {
        "station": station,
        "idx": int(patch["idx"]) if patch is not None else "",
        "time_unix": obs["unix_time"],
        "time_rel_s": obs["unix_time"] - EVENT_T0,
        "seed_range_m": obs["range_m"],
        "seed_doppler_hz": obs["doppler_hz"],
        "success": bool(fit is not None and fit.get("success", False)),
        "error": error or "",
        "figure": display_path(fig_path),
    }
    row.update(settings)
    if patch is not None:
        row["model_doppler_hz"] = patch["model_doppler_hz"]
    if fit is not None:
        row.update(
            {
                "fit_range_m": fit["range_m"],
                "range_correction_m": fit["range_correction_m"],
                "fit_doppler_hz": fit["absolute_doppler_hz"],
                "doppler_correction_hz": fit["doppler_correction_hz"],
                "fit_rel_doppler_hz": fit["rel_doppler_hz"],
                "sigma_range_m": 1000.0 * fit["sigma_range_km"],
                "sigma_doppler_hz": fit["sigma_doppler_hz"],
                "amplitude_db": fit["amplitude_db"],
                "cost": fit["cost"],
            }
        )
    return row


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "station",
        "idx",
        "time_unix",
        "time_rel_s",
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
        "amplitude_db",
        "cost",
        "success",
        "error",
        "n_coh",
        "n_incoh",
        "step_codes",
        "notch_half_width",
        "figure",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--stations", nargs="+", default=None)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=16)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    settings = {
        "n_coh": args.n_coh,
        "n_incoh": args.n_incoh,
        "step_codes": args.step_codes,
        "notch_half_width": args.notch_half_width,
    }
    rows = []
    with h5py.File(args.cache, "r") as h:
        stations = args.stations
        if stations is None:
            stations = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]
        for station in stations:
            group = h[station]
            keys = sorted([key for key in group.keys() if key.startswith("patch_")])
            if args.limit:
                keys = keys[: args.limit]
            print(f"{station}: fitting {len(keys)} cached patches", flush=True)
            for key in keys:
                obs, patch, fit, error = fit_cached_patch(station, group[key], args)
                idx = int(group[key].attrs["idx"])
                time_rel = obs["unix_time"] - EVENT_T0
                fig_path = args.out_dir / station / safe_name(station, idx, time_rel)
                plot_patch(fig_path, station, obs, patch, fit, error)
                rows.append(summary_row(station, obs, patch, fit, error, fig_path, settings))
    write_summary(args.summary, rows)
    print(f"Wrote {args.summary}", flush=True)
    print(f"Wrote {len(rows)} diagnostic figures under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
