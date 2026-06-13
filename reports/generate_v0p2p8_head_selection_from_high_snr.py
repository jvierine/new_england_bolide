#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS = ROOT / "reports" / "head_echo_python_selection_v0p2p7_zp16" / "head_echo_python_measurement_selection.csv"
DEFAULT_SNR_H5 = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_rdmf_baseline.h5"
DEFAULT_RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_OUT_DIR = ROOT / "reports" / "head_echo_python_selection_v0p2p8_highsnr"


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes", "real"})


def select_high_snr_pomfret(
    snr_h5: Path,
    rdmf_dir: Path,
    snr_dataset: str,
    snr_cutoff_db: float,
    half_width_px: int,
    station: str,
) -> tuple[pd.DataFrame, dict]:
    with h5py.File(snr_h5, "r") as h:
        g = h[station]
        times = np.asarray(g["time_unix"][()], dtype=float)
        ranges_km = np.asarray(g["range_km"][()], dtype=float)
        model_range_km = np.asarray(g["model_range_km"][()], dtype=float)
        snr_db = np.asarray(g[snr_dataset][()], dtype=float)

    with h5py.File(rdmf_dir / f"rdmf_peak_{station}_merged.h5", "r") as h:
        d_head = np.asarray(h["D_head"][()], dtype=float) if "D_head" in h else np.asarray(h["D"][()], dtype=float)
        rdmf_times = np.asarray(h["t_sample"][()], dtype=float) / float(h["sample_rate"][()])
        rdmf_ranges_km = np.asarray(h["range_km"][()], dtype=float)
        fvec = np.asarray(h["fvec"][()], dtype=float)

    if snr_db.shape != d_head.shape:
        raise ValueError(f"{station}: SNR shape {snr_db.shape} does not match Doppler shape {d_head.shape}")
    if not np.allclose(times, rdmf_times):
        raise ValueError(f"{station}: SNR times do not match RDMF times")
    if not np.allclose(ranges_km, rdmf_ranges_km):
        raise ValueError(f"{station}: SNR ranges do not match RDMF ranges")

    range_step_m = float(1000.0 * np.nanmedian(np.diff(ranges_km)))
    doppler_step_hz = float(np.nanmedian(np.diff(np.sort(fvec))))
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
        if not np.isfinite(amp) or amp < snr_cutoff_db:
            continue
        rows.append(
            {
                "station": station,
                "idx": len(rows),
                "time_unix": float(times[ti]),
                "time_rel_s": float(times[ti] - 1780164380.0),
                "fit_range_m": float(1000.0 * ranges_km[ri]),
                "fit_doppler_hz": float(d_head[ti, ri]),
                "sigma_range_m": range_step_m,
                "sigma_doppler_hz": max(25.0, doppler_step_hz),
                "amplitude_db": amp,
                "cost": 0.0,
                "selection": "real",
                "selection_range": "real",
                "selection_doppler": "real",
                "real": True,
                "real_range": True,
                "real_doppler": True,
                "source": f"{snr_dataset}:max_pm{half_width_px}px",
                "source_range_idx": int(ri),
            }
        )
    meta = {
        "station": station,
        "snr_h5": str(snr_h5),
        "rdmf_dir": str(rdmf_dir),
        "snr_dataset": snr_dataset,
        "snr_cutoff_db": snr_cutoff_db,
        "half_width_px": half_width_px,
        "range_step_m": range_step_m,
        "doppler_step_hz": doppler_step_hz,
        "n_selected": len(rows),
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    p = argparse.ArgumentParser(description="Create v0.2.8 head-selection CSV from high-SNR Pomfret RDMF pixels.")
    p.add_argument("--previous-selection", type=Path, default=DEFAULT_PREVIOUS)
    p.add_argument("--snr-h5", type=Path, default=DEFAULT_SNR_H5)
    p.add_argument("--rdmf-dir", type=Path, default=DEFAULT_RDMF_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--station", default="pomfret")
    p.add_argument("--snr-dataset", default="txsteer_pol_snr_db")
    p.add_argument("--snr-cutoff-db", type=float, default=15.0)
    p.add_argument("--half-width-px", type=int, default=10)
    args = p.parse_args()

    prev = pd.read_csv(args.previous_selection)
    prev["real_range"] = bool_series(prev["real_range"])
    prev["real_doppler"] = bool_series(prev["real_doppler"])
    keep_prev = prev[prev["station"].astype(str).str.lower() != args.station].copy()
    pomfret, meta = select_high_snr_pomfret(
        args.snr_h5,
        args.rdmf_dir,
        args.snr_dataset,
        args.snr_cutoff_db,
        args.half_width_px,
        args.station,
    )
    shared_cols = [
        "station",
        "idx",
        "time_unix",
        "time_rel_s",
        "fit_range_m",
        "fit_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "amplitude_db",
        "cost",
        "selection",
        "selection_range",
        "selection_doppler",
        "real",
        "real_range",
        "real_doppler",
    ]
    out = pd.concat([keep_prev[shared_cols], pomfret[shared_cols]], ignore_index=True)
    out = out.sort_values(["station", "time_unix", "idx"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "head_echo_python_measurement_selection.csv"
    json_path = args.out_dir / "head_echo_python_measurement_selection.json"
    state_path = args.out_dir / "head_echo_python_selection_state.json"
    summary_path = args.out_dir / "head_echo_python_selection_summary.json"
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(out.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8")
    state = {
        f"{row.station}:{int(row.idx)}": {
            "range": "real" if bool(row.real_range) else "unknown",
            "doppler": "real" if bool(row.real_doppler) else "unknown",
        }
        for row in out.itertuples(index=False)
    }
    state_path.write_text(json.dumps({"selections": state}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "version": "0.2.8",
        "description": "Beverly and Clinton selections inherited from v0.2.7; Pomfret replaced by high-SNR trajectory-steered TX and polarization-matched RDMF pixel selections.",
        "previous_selection": str(args.previous_selection),
        "counts_all_rows": out.groupby("station").size().to_dict(),
        "counts_selected_rows": out[out["real_range"] | out["real_doppler"]].groupby("station").size().to_dict(),
        "pomfret_high_snr": meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(csv_path)
    print(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
