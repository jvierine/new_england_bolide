#!/usr/bin/env python3
"""Build selector measurements from the memo RDMF tx-steer/pol-match product.

This intentionally follows reports/generate_station_head_snr_rdmf_baseline.py:
the five stored decoded-TX peak voltages are trajectory-steered, receive
polarization is selected from the TX-steered peak inside the Gaussian 1-sigma
area, and the stored RDMF noise product is left unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0  # noqa: E402
from reports.generate_station_head_rcs_figure import STATION_LABELS  # noqa: E402
from reports.generate_station_head_snr_rdmf_baseline import (  # noqa: E402
    DATA_DIR,
    DEFAULT_GAUSSIAN_SUMMARY,
    DEFAULT_TRAJECTORY,
    RDMF_DIR,
    STATIONS,
    load_gaussian_summary,
    load_trajectory,
    match_gaussian_rows,
    model_bistatic_range_km,
    steering_vectors,
)
from angle_departure_estimator import load_station_yaml  # noqa: E402
from estimate_head_echo_trajectory import station_geometry  # noqa: E402
from fit_ballistic_head_trail import head_predictions_from_states  # noqa: E402
from fit_combined_head_trail_eci import cache_head_station_states  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "reports" / "head_echo_python_selection_v0p2p8_rdmf_txsteer_pol"
DEFAULT_OUTPUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "head_echo_rdmf_txsteer_pol_measurements.h5"
DEFAULT_OUTPUT_PNG = ROOT / "reports" / "figures" / "head_echo_rdmf_txsteer_pol_measurements.png"


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
    return np.asarray(dopplers, dtype=np.float64)


def compute_station_product(
    station: str,
    args: argparse.Namespace,
    traj: dict[str, np.ndarray],
    gaussian_summary: dict[str, dict[str, np.ndarray]],
) -> dict:
    path = args.rdmf_dir / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(path, "r") as h:
        p_head = h["P_head"][()].astype(np.float64) if "P_head" in h else h["P"][()].astype(np.float64)
        noise = h["N"][()].astype(np.float64)
        d_head = h["D_head"][()].astype(np.float64) if "D_head" in h else h["D"][()].astype(np.float64)
        voltage = h["voltage_peak"][()].astype(np.complex64) if "voltage_peak" in h else None
        time_unix = h["t_sample"][()].astype(np.float64) / float(h["sample_rate"][()])
        range_km = h["range_km"][()].astype(np.float64)
        fvec = h["fvec"][()].astype(np.float64)
        model_range_km = model_bistatic_range_km(h, traj, time_unix)
        pol_gamma = h["pol_gamma"][()].astype(np.complex64) if "pol_gamma" in h else None
        pol_reference_range_idx = h["pol_peak_range_idx"][()].astype(np.int32) if "pol_peak_range_idx" in h else None

    if voltage is None:
        txsteer_pol_snr_db = 10.0 * np.log10(np.maximum(p_head / np.maximum(noise, 1e-30), 1e-30))
        if pol_gamma is None:
            pol_gamma = np.full((len(time_unix), 2), np.nan + 1.0j * np.nan, dtype=np.complex64)
        if pol_reference_range_idx is None:
            pol_reference_range_idx = np.full(len(time_unix), -1, dtype=np.int32)
        gaussian_valid = np.ones(len(time_unix), dtype=np.uint8)
    else:
        site = load_station_yaml(DATA_DIR, station)
        voltage_power = np.sum(np.abs(voltage) ** 2, axis=(1, 3), dtype=np.float64) / float(voltage.shape[1] * voltage.shape[3])
        rdmf_power_scale = p_head / np.maximum(voltage_power, 1e-30)
        steering = steering_vectors(site, traj, time_unix, float(args.phase_sign))
        if voltage.shape[-1] != steering.shape[-1]:
            raise ValueError(f"{station}: voltage has {voltage.shape[-1]} TX channels but steering has {steering.shape[-1]}")

        tx_beam = np.einsum("tcrx,tx->tcr", voltage, np.conj(steering), optimize=True)
        txsteer_raw_power = np.sum(np.abs(tx_beam) ** 2, axis=1, dtype=np.float64) / float(voltage.shape[1] * voltage.shape[3])
        fit = match_gaussian_rows(station, time_unix, gaussian_summary, float(args.gaussian_max_dt_s))
        range_m = range_km * 1000.0
        path_range_idx = np.argmin(np.abs(range_km[None, :] - model_range_km[:, None]), axis=1)

        txsteer_pol_raw_power = np.empty_like(txsteer_raw_power)
        pol_reference_range_idx = np.empty(len(time_unix), dtype=np.int32)
        pol_gamma = np.empty((len(time_unix), voltage.shape[1]), dtype=np.complex64)
        for ti in range(len(time_unix)):
            if bool(fit["gaussian_valid"][ti]):
                sig_r = max(float(fit["sigma_range_m"][ti]), 1e-12)
                sig_d = max(float(fit["sigma_doppler_hz"][ti]), 1e-12)
                rho = float(np.clip(fit["range_doppler_correlation"][ti], -0.95, 0.95))
                rr = (range_m - float(fit["fit_range_m"][ti])) / sig_r
                dd = (d_head[ti] - float(fit["fit_doppler_hz"][ti])) / sig_d
                q = (rr * rr - 2.0 * rho * rr * dd + dd * dd) / max(1.0 - rho * rho, 1e-12)
                mask = np.isfinite(q) & (q <= 1.0) & np.isfinite(txsteer_raw_power[ti])
            else:
                mask = np.zeros_like(range_m, dtype=bool)
                mask[path_range_idx[ti]] = True
            if not np.any(mask):
                mask = np.isfinite(txsteer_raw_power[ti])

            ref_ri = int(np.nanargmax(np.where(mask, txsteer_raw_power[ti], -np.inf)))
            pol_reference_range_idx[ti] = ref_ri
            peak = tx_beam[ti, :, ref_ri].astype(np.complex128)
            gamma = peak / max(np.linalg.norm(peak), 1e-30)
            pol_gamma[ti] = gamma.astype(np.complex64)
            combined = np.einsum("c,cr->r", np.conj(gamma), tx_beam[ti], optimize=True)
            txsteer_pol_raw_power[ti] = np.abs(combined) ** 2 / float(voltage.shape[3])

        txsteer_pol_power = txsteer_pol_raw_power * rdmf_power_scale
        txsteer_pol_snr_db = 10.0 * np.log10(np.maximum(txsteer_pol_power / np.maximum(noise, 1e-30), 1e-30))
        gaussian_valid = fit["gaussian_valid"].astype(np.uint8)
    model_doppler_hz = trajectory_doppler(station, args.data_dir, args.trajectory_json, time_unix)

    return {
        "station": station,
        "time_unix": time_unix,
        "range_km": range_km,
        "d_head": d_head,
        "noise": noise,
        "fvec": fvec,
        "model_range_km": model_range_km,
        "model_doppler_hz": model_doppler_hz,
        "txsteer_pol_snr_db": txsteer_pol_snr_db,
        "pol_reference_range_idx": pol_reference_range_idx,
        "pol_gamma": pol_gamma,
        "gaussian_valid": gaussian_valid,
        "rdmf_n_coh": 1,
        "rdmf_n_incoh": 2,
        "rdmf_step_codes": 1,
    }


def measurement_rows(data: dict, args: argparse.Namespace) -> list[dict]:
    times = data["time_unix"]
    ranges = data["range_km"]
    snr = data["txsteer_pol_snr_db"]
    d_head = data["d_head"]
    model_range = data["model_range_km"]
    model_doppler = data["model_doppler_hz"]
    rows = []
    range_step_m = float(1000.0 * np.nanmedian(np.diff(ranges)))
    doppler_step_hz = float(np.nanmedian(np.diff(np.unique(np.round(data["fvec"], 6)))))
    for ti, center_km in enumerate(model_range):
        if not np.isfinite(center_km):
            continue
        center = int(np.nanargmin(np.abs(ranges - center_km)))
        lo = max(0, center - int(args.search_range_half_width_px))
        hi = min(len(ranges), center + int(args.search_range_half_width_px) + 1)
        search = snr[ti, lo:hi]
        if not np.any(np.isfinite(search)):
            continue
        ri = lo + int(np.nanargmax(search))
        amp = float(snr[ti, ri])
        if not np.isfinite(amp):
            continue
        rows.append(
            {
                "station": data["station"],
                "idx": len(rows),
                "time_unix": float(times[ti]),
                "time_rel_s": float(times[ti] - EVENT_T0),
                "patch_time_unix": float(times[ti]),
                "patch_time_rel_s": float(times[ti] - EVENT_T0),
                "seed_range_m": float(1000.0 * model_range[ti]),
                "fit_range_m": float(1000.0 * ranges[ri]),
                "range_correction_m": float(1000.0 * (ranges[ri] - model_range[ti])),
                "seed_doppler_hz": float(model_doppler[ti]),
                "fit_doppler_hz": float(d_head[ti, ri]),
                "doppler_correction_hz": float(d_head[ti, ri] - model_doppler[ti]),
                "model_doppler_hz": float(model_doppler[ti]),
                "fit_rel_doppler_hz": float(d_head[ti, ri] - model_doppler[ti]),
                "sigma_range_m": range_step_m,
                "sigma_doppler_hz": max(10.0, doppler_step_hz),
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
                "polarization_gamma0_real": float(np.real(data["pol_gamma"][ti, 0])),
                "polarization_gamma0_imag": float(np.imag(data["pol_gamma"][ti, 0])),
                "polarization_gamma1_real": float(np.real(data["pol_gamma"][ti, 1])),
                "polarization_gamma1_imag": float(np.imag(data["pol_gamma"][ti, 1])),
                "polarization_gamma0_abs": float(np.abs(data["pol_gamma"][ti, 0])),
                "polarization_gamma1_abs": float(np.abs(data["pol_gamma"][ti, 1])),
                "polarization_relative_phase_deg": float(np.angle(data["pol_gamma"][ti, 1] / data["pol_gamma"][ti, 0], deg=True))
                if abs(data["pol_gamma"][ti, 0]) > 0
                else np.nan,
                "polarization_metric_kind": "rdmf_txsteer_gaussian_1sigma_reference_pixel",
                "polarization_metric_mask_pixels": np.nan,
                "polarization_metric": amp,
                "polarization_signal_power": np.nan,
                "polarization_noise_power": np.nan,
                "channel_noise_equalizer0": np.nan,
                "channel_noise_equalizer1": np.nan,
                "channel_noise_power0": np.nan,
                "channel_noise_power1": np.nan,
                "code_period_s": 0.01,
                "rdmf_range_idx": int(ri),
                "rdmf_pol_reference_range_idx": int(data["pol_reference_range_idx"][ti]),
                "gaussian_reference_valid": bool(data["gaussian_valid"][ti]),
                "figure": "",
            }
        )
    return rows


def write_h5(path: Path, products: list[dict], rows: pd.DataFrame, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Selector measurements from memo RDMF trajectory TX steering and receive polarization matching"
        h.attrs["source_script"] = "reports/generate_txsteer_pol_rdmf_measurements.py"
        h.attrs["source_logic"] = "reports/generate_station_head_snr_rdmf_baseline.py"
        h.attrs["rdmf_dir"] = str(args.rdmf_dir)
        h.attrs["trajectory_json"] = str(args.trajectory_json)
        h.attrs["gaussian_summary_csv"] = str(args.gaussian_summary_csv)
        h.attrs["search_range_half_width_px"] = int(args.search_range_half_width_px)
        h.attrs["phase_sign"] = float(args.phase_sign)
        h.create_dataset("stations", data=np.asarray([p["station"].encode("utf-8") for p in products]))
        for product in products:
            g = h.create_group(product["station"])
            for key in (
                "time_unix",
                "range_km",
                "d_head",
                "model_range_km",
                "model_doppler_hz",
                "txsteer_pol_snr_db",
                "pol_reference_range_idx",
                "pol_gamma",
                "gaussian_valid",
            ):
                g.create_dataset(key, data=product[key], compression="gzip" if np.ndim(product[key]) else None)
        rec = rows.to_records(index=False)
        h.attrs["measurement_csv"] = str(args.out_dir / "head_echo_measurement_candidates.csv")


def write_selector_state(out_dir: Path, rows: pd.DataFrame) -> None:
    state = {f"{row.station}:{int(row.idx)}": {"range": "unknown", "doppler": "unknown"} for row in rows.itertuples(index=False)}
    (out_dir / "head_echo_python_selection_state.json").write_text(json.dumps({"selections": state}, indent=2) + "\n", encoding="utf-8")


def plot_measurements(rows: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.6), sharex="col", constrained_layout=True)
    for row_i, station in enumerate([s for s in STATIONS if s in set(rows["station"])]):
        g = rows[rows["station"] == station].sort_values("time_rel_s")
        axr, axd = axes[row_i]
        axr.plot(g["time_rel_s"], g["seed_range_m"] / 1000.0, color="0.2", lw=1.2, ls="--", alpha=0.7, label="trajectory")
        sc = axr.scatter(g["time_rel_s"], g["fit_range_m"] / 1000.0, c=g["snr_db"], s=22, cmap="viridis", vmin=0.0, vmax=23.0)
        axr.set_ylabel(f"{STATION_LABELS.get(station, station)}\nRange (km)")
        axr.grid(True, color="0.88")
        axr.legend(loc="best", fontsize=8)
        axd.plot(g["time_rel_s"], g["seed_doppler_hz"], color="0.2", lw=1.2, ls="--", alpha=0.7)
        axd.scatter(g["time_rel_s"], g["fit_doppler_hz"], c=g["snr_db"], s=22, cmap="viridis", vmin=0.0, vmax=23.0)
        axd.set_ylabel("Doppler (Hz)")
        axd.grid(True, color="0.88")
    axes[-1, 0].set_xlabel("Seconds after 18:06:20 UTC")
    axes[-1, 1].set_xlabel("Seconds after 18:06:20 UTC")
    axes[0, 0].set_title("RDMF tx-steer/pol-match range")
    axes[0, 1].set_title("RDMF tx-steer/pol-match Doppler")
    cb = fig.colorbar(sc, ax=axes, location="right", shrink=0.92, pad=0.012)
    cb.set_label("RDMF SNR (dB)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--rdmf-dir", type=Path, default=RDMF_DIR)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--gaussian-summary-csv", type=Path, default=DEFAULT_GAUSSIAN_SUMMARY)
    p.add_argument("--gaussian-max-dt-s", type=float, default=0.03)
    p.add_argument("--phase-sign", type=float, default=-1.0)
    p.add_argument("--search-range-half-width-px", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    p.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PNG)
    args = p.parse_args()

    traj = load_trajectory(args.trajectory_json)
    gaussian_summary = load_gaussian_summary(args.gaussian_summary_csv)
    products = [compute_station_product(station, args, traj, gaussian_summary) for station in args.stations]
    rows = pd.DataFrame([row for product in products for row in measurement_rows(product, args)])
    rows = rows.sort_values(["station", "time_unix", "idx"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "head_echo_measurement_candidates.csv"
    rows.to_csv(summary_csv, index=False)
    # The selector script accepts this file directly via --summary.
    rows.to_csv(args.out_dir / "head_echo_gaussian_fit_summary.csv", index=False)
    write_selector_state(args.out_dir, rows)
    write_h5(args.output_h5, products, rows, args)
    plot_measurements(rows, args.output_plot)
    print(summary_csv, flush=True)
    print(args.output_h5, flush=True)
    print(args.output_plot, flush=True)
    print(rows.groupby("station").size(), flush=True)
    print(rows.groupby("station")["snr_db"].agg(["min", "median", "max"]), flush=True)


if __name__ == "__main__":
    main()
