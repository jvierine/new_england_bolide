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

from reports.generate_head_echo_gaussian_batch import display_path, plot_patch, safe_name, smooth_power_db  # noqa: E402
from reports.generate_head_echo_gaussian_example import fit_gaussian_patch, gaussian_power_snr  # noqa: E402
from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, notch_2d_doppler_repetition  # noqa: E402

DEFAULT_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code.h5"
DEFAULT_OUT_DIR = ROOT / "reports" / "figures" / "head_echo_gaussian_fits"
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary.csv"


def coherent_center_time(start_time_unix: float, n_coh: int, code_period_s: float) -> float:
    return float(start_time_unix) + 0.5 * float(n_coh) * float(code_period_s)


def fit_cached_patch(station: str, patch_group: h5py.Group, args: argparse.Namespace) -> tuple[dict, dict | None, dict | None, str | None]:
    patch_start_time = float(patch_group.attrs["time_unix"])
    patch_center_time = coherent_center_time(patch_start_time, args.n_coh, args.code_period_s)
    obs = {
        "station": station,
        "unix_time": patch_center_time,
        "patch_time_unix": patch_start_time,
        "range_m": float(patch_group.attrs["seed_range_m"]),
        "doppler_hz": float(patch_group.attrs["picked_doppler_hz"]),
    }
    if not bool(patch_group.attrs.get("valid", False)):
        return obs, None, None, "invalid cached patch"
    patch = {
        "idx": int(patch_group.attrs["idx"]),
        "time": patch_center_time,
        "patch_time": patch_start_time,
        "seed_range_m": float(patch_group.attrs["seed_range_m"]),
        "picked_doppler_hz": float(patch_group.attrs["picked_doppler_hz"]),
        "model_doppler_hz": float(patch_group.attrs["model_doppler_hz"]),
        "range_m": patch_group["range_m"][()].astype(float),
        "rel_freqs": patch_group["rel_freqs"][()].astype(float),
        "raw_db": patch_group["raw_db"][()].astype(float),
        "coherent_window_s": float(patch_group.attrs.get("coherent_window_s", np.nan)),
        "total_window_s": float(patch_group.attrs.get("total_window_s", np.nan)),
    }
    if "polarization_gamma" in patch_group:
        gamma = patch_group["polarization_gamma"][()].astype(np.complex64)
        patch["polarization_gamma"] = gamma
        patch["polarization_gamma0_real"] = float(np.real(gamma[0]))
        patch["polarization_gamma0_imag"] = float(np.imag(gamma[0]))
        patch["polarization_gamma1_real"] = float(np.real(gamma[1]))
        patch["polarization_gamma1_imag"] = float(np.imag(gamma[1]))
        patch["polarization_gamma0_abs"] = float(np.abs(gamma[0]))
        patch["polarization_gamma1_abs"] = float(np.abs(gamma[1]))
        patch["polarization_relative_phase_deg"] = float(np.angle(gamma[1] / gamma[0], deg=True)) if abs(gamma[0]) > 0 else np.nan
    for key in (
        "polarization_metric",
        "polarization_signal_power",
        "polarization_noise_power",
        "polarization_metric_mask_pixels",
    ):
        if key in patch_group.attrs:
            patch[key] = patch_group.attrs[key]
    if "polarization_metric_kind" in patch_group.attrs:
        patch["polarization_metric_kind"] = str(patch_group.attrs["polarization_metric_kind"])
    if "channel_noise_equalizer" in patch_group:
        eq = patch_group["channel_noise_equalizer"][()]
        patch["channel_noise_equalizer0"] = float(eq[0])
        patch["channel_noise_equalizer1"] = float(eq[1])
    if "channel_noise_powers" in patch_group:
        powers = patch_group["channel_noise_powers"][()]
        patch["channel_noise_power0"] = float(powers[0])
        patch["channel_noise_power1"] = float(powers[1])
    patch["raw_rel_freqs"] = patch["rel_freqs"].copy()
    patch["filtered_unsmoothed_db"] = notch_2d_doppler_repetition(
        patch["raw_db"],
        args.n_coh,
        args.notch_half_width,
        fft_domain=args.fft_filter_domain,
        inpaint_columns=args.inpaint_notched_columns,
    )
    patch["filtered_db"] = smooth_power_db(
        patch["filtered_unsmoothed_db"],
        args.smooth_range_pixels,
        args.smooth_frequency_pixels,
    )
    patch["smooth_range_pixels"] = float(args.smooth_range_pixels)
    patch["smooth_frequency_pixels"] = float(args.smooth_frequency_pixels)
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
        fit.update(
            gaussian_power_snr(
                patch["range_m"] / 1000.0,
                patch["rel_freqs"],
                fit,
                seed_range_km,
                seed_rel_hz,
            )
        )
        return obs, patch, fit, None
    except Exception as exc:
        return obs, patch, None, str(exc)


def summary_row(station: str, obs: dict, patch: dict | None, fit: dict | None, error: str | None, fig_path: Path, settings: dict) -> dict:
    row = {
        "station": station,
        "idx": int(patch["idx"]) if patch is not None else "",
        "time_unix": obs["unix_time"],
        "time_rel_s": obs["unix_time"] - EVENT_T0,
        "patch_time_unix": obs.get("patch_time_unix", ""),
        "patch_time_rel_s": obs["patch_time_unix"] - EVENT_T0 if "patch_time_unix" in obs else "",
        "seed_range_m": obs["range_m"],
        "seed_doppler_hz": obs["doppler_hz"],
        "success": bool(fit is not None and fit.get("success", False)),
        "error": error or "",
        "figure": display_path(fig_path),
    }
    row.update(settings)
    if patch is not None:
        row["model_doppler_hz"] = patch["model_doppler_hz"]
        row["coherent_window_s"] = patch.get("coherent_window_s", "")
        row["total_window_s"] = patch.get("total_window_s", "")
        for key in (
            "polarization_gamma0_real",
            "polarization_gamma0_imag",
            "polarization_gamma1_real",
            "polarization_gamma1_imag",
            "polarization_gamma0_abs",
            "polarization_gamma1_abs",
            "polarization_relative_phase_deg",
            "polarization_metric_kind",
            "polarization_metric_mask_pixels",
            "polarization_metric",
            "polarization_signal_power",
            "polarization_noise_power",
            "channel_noise_equalizer0",
            "channel_noise_equalizer1",
            "channel_noise_power0",
            "channel_noise_power1",
        ):
            if key in patch:
                row[key] = patch[key]
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
                "range_doppler_correlation": fit["range_doppler_correlation"],
                "amplitude_db": fit["amplitude_db"],
                "signal_power_gaussian": fit["signal_power_gaussian"],
                "noise_power_floor": fit["noise_power_floor"],
                "snr_linear": fit["snr_linear"],
                "snr_db": fit["snr_db"],
                "noise_bandwidth_hz": fit["noise_bandwidth_hz"],
                "noise_range_width_km": fit["noise_range_width_km"],
                "noise_n_range_bins": fit["noise_n_range_bins"],
                "noise_n_doppler_bins": fit["noise_n_doppler_bins"],
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
        "signal_power_gaussian",
        "noise_power_floor",
        "snr_linear",
        "snr_db",
        "noise_bandwidth_hz",
        "noise_range_width_km",
        "noise_n_range_bins",
        "noise_n_doppler_bins",
        "cost",
        "success",
        "error",
        "n_coh",
        "n_incoh",
        "step_codes",
        "notch_half_width",
        "fft_filter_domain",
        "inpaint_notched_columns",
        "smooth_range_pixels",
        "smooth_frequency_pixels",
        "coherent_window_s",
        "total_window_s",
        "polarization_gamma0_real",
        "polarization_gamma0_imag",
        "polarization_gamma1_real",
        "polarization_gamma1_imag",
        "polarization_gamma0_abs",
        "polarization_gamma1_abs",
        "polarization_relative_phase_deg",
        "polarization_metric_kind",
        "polarization_metric_mask_pixels",
        "polarization_metric",
        "polarization_signal_power",
        "polarization_noise_power",
        "channel_noise_equalizer0",
        "channel_noise_equalizer1",
        "channel_noise_power0",
        "channel_noise_power1",
        "code_period_s",
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
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--fft-filter-domain", choices=("db", "linear"), default="linear")
    p.add_argument("--inpaint-notched-columns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=16.0)
    p.add_argument("--code-period-s", type=float, default=0.01)
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
        "fft_filter_domain": args.fft_filter_domain,
        "inpaint_notched_columns": bool(args.inpaint_notched_columns),
        "smooth_range_pixels": args.smooth_range_pixels,
        "smooth_frequency_pixels": args.smooth_frequency_pixels,
        "coherent_window_s": "",
        "total_window_s": "",
        "code_period_s": args.code_period_s,
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
