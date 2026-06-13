#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit_ballistic_head_trail import head_predictions_from_states  # noqa: E402
from fit_combined_head_trail_eci import cache_head_station_states  # noqa: E402
from estimate_head_echo_trajectory import station_geometry  # noqa: E402

DEFAULT_SNR_H5 = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_rdmf_baseline.h5"
DEFAULT_RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_OUT_DIR = ROOT / "reports" / "head_echo_python_selection_v0p2p8_txsteer_pol"
DEFAULT_TRAJECTORY_JSON = (
    ROOT
    / "products"
    / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5"
    / "ballistic_head_trail_correlated_fit.json"
)
DEFAULT_DATA_DIR = ROOT / "2026_05_30_bolide"
STATIONS = ("beverly", "clinton", "pomfret")


def trajectory_state(path: Path, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload["model"]
    tm = np.asarray(model["time_unix"], dtype=float)
    order = np.argsort(tm)
    tm = tm[order]
    pos = np.asarray(model["position_gcrs_m"], dtype=float)[order]
    vel = np.asarray(model["velocity_gcrs_mps"], dtype=float)[order]
    pos_i = np.column_stack([np.interp(times, tm, pos[:, i]) for i in range(3)])
    vel_i = np.column_stack([np.interp(times, tm, vel[:, i]) for i in range(3)])
    return pos_i, vel_i


def trajectory_doppler(station: str, data_dir: Path, trajectory_json: Path, times: np.ndarray) -> np.ndarray:
    geom = station_geometry(data_dir, [station])
    obs = cache_head_station_states([{"station": station, "unix_time": float(t)} for t in times], geom)
    pos, vel = trajectory_state(trajectory_json, times)
    _ranges, dopplers = head_predictions_from_states(pos, vel, obs)
    return dopplers


def station_candidates(
    station: str,
    snr_h5: Path,
    rdmf_dir: Path,
    data_dir: Path,
    trajectory_json: Path,
    snr_dataset: str,
    half_width_px: int,
) -> tuple[pd.DataFrame, dict]:
    with h5py.File(snr_h5, "r") as h:
        g = h[station]
        times = np.asarray(g["time_unix"][()], dtype=float)
        ranges_km = np.asarray(g["range_km"][()], dtype=float)
        model_range_km = np.asarray(g["model_range_km"][()], dtype=float)
        snr_db = np.asarray(g[snr_dataset][()], dtype=float)

    with h5py.File(rdmf_dir / f"rdmf_peak_{station}_merged.h5", "r") as h:
        rdmf_times = np.asarray(h["t_sample"][()], dtype=float) / float(h["sample_rate"][()])
        rdmf_ranges_km = np.asarray(h["range_km"][()], dtype=float)
        fvec = np.asarray(h["fvec"][()], dtype=float)

    if snr_db.shape != (rdmf_times.size, rdmf_ranges_km.size):
        raise ValueError(f"{station}: SNR shape {snr_db.shape} does not match RDMF time/range shape")
    if not np.allclose(times, rdmf_times):
        raise ValueError(f"{station}: SNR times do not match RDMF times")
    if not np.allclose(ranges_km, rdmf_ranges_km):
        raise ValueError(f"{station}: SNR ranges do not match RDMF ranges")

    range_step_m = float(1000.0 * np.nanmedian(np.diff(ranges_km)))
    doppler_step_hz = float(np.nanmedian(np.diff(np.sort(fvec))))
    model_doppler_hz = trajectory_doppler(station, data_dir, trajectory_json, times)
    rows = []
    for ti, center_km in enumerate(model_range_km):
        if not np.isfinite(center_km):
            continue
        center = int(np.nanargmin(np.abs(ranges_km - center_km)))
        lo = max(0, center - half_width_px)
        hi = min(ranges_km.size, center + half_width_px + 1)
        row = snr_db[ti, lo:hi]
        if not np.any(np.isfinite(row)):
            continue
        rel = int(np.nanargmax(row))
        ri = lo + rel
        amp = float(snr_db[ti, ri])
        if not np.isfinite(amp):
            continue
        rows.append(
            {
                "station": station,
                "idx": len(rows),
                "time_unix": float(times[ti]),
                "time_rel_s": float(times[ti] - 1780164380.0),
                "patch_time_unix": float(times[ti]),
                "patch_time_rel_s": float(times[ti] - 1780164380.0),
                "seed_range_m": float(1000.0 * model_range_km[ti]),
                "fit_range_m": float(1000.0 * ranges_km[ri]),
                "range_correction_m": float(1000.0 * (ranges_km[ri] - model_range_km[ti])),
                "seed_doppler_hz": float(model_doppler_hz[ti]),
                "fit_doppler_hz": float(model_doppler_hz[ti]),
                "doppler_correction_hz": 0.0,
                "model_doppler_hz": float(model_doppler_hz[ti]),
                "fit_rel_doppler_hz": float(model_doppler_hz[ti]),
                "sigma_range_m": range_step_m,
                "sigma_doppler_hz": max(25.0, doppler_step_hz),
                "range_doppler_correlation": 0.0,
                "amplitude_db": amp,
                "signal_power_gaussian": np.nan,
                "noise_power_floor": np.nan,
                "snr_linear": float(10.0 ** (amp / 10.0)),
                "snr_db": amp,
                "noise_bandwidth_hz": 100.0,
                "noise_range_width_km": np.nan,
                "noise_n_range_bins": np.nan,
                "noise_n_doppler_bins": np.nan,
                "cost": 0.0,
                "success": True,
                "error": "",
                "n_coh": 1,
                "n_incoh": 2,
                "step_codes": 1,
                "notch_half_width": np.nan,
                "fft_filter_domain": "",
                "inpaint_notched_columns": "",
                "smooth_range_pixels": np.nan,
                "smooth_frequency_pixels": np.nan,
                "coherent_window_s": np.nan,
                "total_window_s": np.nan,
                "polarization_gamma0_real": np.nan,
                "polarization_gamma0_imag": np.nan,
                "polarization_gamma1_real": np.nan,
                "polarization_gamma1_imag": np.nan,
                "polarization_gamma0_abs": np.nan,
                "polarization_gamma1_abs": np.nan,
                "polarization_relative_phase_deg": np.nan,
                "polarization_metric_kind": snr_dataset,
                "polarization_metric_mask_pixels": 2 * half_width_px + 1,
                "polarization_metric": amp,
                "polarization_signal_power": np.nan,
                "polarization_noise_power": np.nan,
                "channel_noise_equalizer0": np.nan,
                "channel_noise_equalizer1": np.nan,
                "channel_noise_power0": np.nan,
                "channel_noise_power1": np.nan,
                "code_period_s": 0.01,
                "figure": "",
            }
        )

    meta = {
        "station": station,
        "n_candidates": len(rows),
        "range_step_m": range_step_m,
        "doppler_step_hz": doppler_step_hz,
        "doppler_source": "trajectory-implied bistatic Doppler from fit JSON",
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    p = argparse.ArgumentParser(description="Build all-station txsteer_pol candidate CSV for the measurement selector GUI.")
    p.add_argument("--snr-h5", type=Path, default=DEFAULT_SNR_H5)
    p.add_argument("--rdmf-dir", type=Path, default=DEFAULT_RDMF_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY_JSON)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--snr-dataset", default="txsteer_pol_snr_db")
    p.add_argument("--half-width-px", type=int, default=10)
    args = p.parse_args()

    frames = []
    station_meta = []
    for station in args.stations:
        df, meta = station_candidates(
            station,
            args.snr_h5,
            args.rdmf_dir,
            args.data_dir,
            args.trajectory_json,
            args.snr_dataset,
            args.half_width_px,
        )
        frames.append(df)
        station_meta.append(meta)
    out = pd.concat(frames, ignore_index=True).sort_values(["station", "time_unix", "idx"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = args.out_dir / "head_echo_measurement_candidates.csv"
    state_json = args.out_dir / "head_echo_python_selection_state.json"
    summary_json = args.out_dir / "head_echo_selection_candidate_summary.json"
    out.to_csv(candidates_csv, index=False)
    state = {f"{row.station}:{int(row.idx)}": {"range": "unknown", "doppler": "unknown"} for row in out.itertuples(index=False)}
    state_json.write_text(json.dumps({"selections": state}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "version": "0.2.8",
        "description": "All stations use txsteer_pol RDMF pixel candidates. No measurements are preselected.",
        "snr_h5": str(args.snr_h5),
        "rdmf_dir": str(args.rdmf_dir),
        "trajectory_json": str(args.trajectory_json),
        "snr_dataset": args.snr_dataset,
        "selection": f"maximum within +/-{args.half_width_px} range pixels of model_range_km",
        "doppler_source": "trajectory-implied bistatic Doppler; txsteer_pol product is range-time only",
        "station_meta": station_meta,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(candidates_csv)
    print(state_json)
    print(out.groupby("station").size())


if __name__ == "__main__":
    main()
