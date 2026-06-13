#!/usr/bin/env python3
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.generate_station_head_rcs_figure import (  # noqa: E402
    C_MPS,
    K_BOLTZMANN,
    geodetic_to_ecef,
    target_ecef_from_fit,
)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def load_station_gps(data_dir: Path, station: str) -> tuple[np.ndarray, np.ndarray, float]:
    with (data_dir / f"analysis-config-millstone_{station}.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    system = cfg["System"]
    return (
        np.asarray(system["gps_tx"], dtype=float),
        np.asarray(system["gps_rx"], dtype=float),
        float(system["frequency"]),
    )


def path_ranges_m(data_dir: Path, station: str, fit_json: Path, times_unix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gps_tx, gps_rx, frequency_hz = load_station_gps(data_dir, station)
    tx = geodetic_to_ecef(np.array([gps_tx[0]]), np.array([gps_tx[1]]), np.array([gps_tx[2]]))[0]
    rx = geodetic_to_ecef(np.array([gps_rx[0]]), np.array([gps_rx[1]]), np.array([gps_rx[2]]))[0]
    target = target_ecef_from_fit(fit_json, times_unix)
    r_tx_m = np.linalg.norm(target - tx[None, :], axis=1)
    r_rx_m = np.linalg.norm(target - rx[None, :], axis=1)
    return r_tx_m, r_rx_m, frequency_hz


def pick_path_snr_db(group: h5py.Group, dataset: str, half_width_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snr = np.asarray(group[dataset][()], dtype=float)
    times = np.asarray(group["time_unix"][()], dtype=float)
    ranges = np.asarray(group["range_km"][()], dtype=float)
    model_range = np.asarray(group["model_range_km"][()], dtype=float)
    out = np.full(times.shape, np.nan, dtype=float)
    pick_range = np.full(times.shape, np.nan, dtype=float)
    for ti, center_km in enumerate(model_range):
        if not np.isfinite(center_km):
            continue
        center = int(np.nanargmin(np.abs(ranges - center_km)))
        lo = max(0, center - half_width_px)
        hi = min(ranges.size, center + half_width_px + 1)
        row = snr[ti, lo:hi]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        rel = int(np.nanargmax(row))
        out[ti] = row[rel]
        pick_range[ti] = ranges[lo + rel]
    return times, pick_range, out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Pomfret-only head-echo RCS along the modeled range path.")
    p.add_argument(
        "--input-h5",
        type=Path,
        default=ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_rdmf_baseline.h5",
    )
    p.add_argument("--station", default="pomfret")
    p.add_argument("--snr-dataset", default="txsteer_pol_snr_db")
    p.add_argument("--half-width-px", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    p.add_argument(
        "--fit-json",
        type=Path,
        default=ROOT
        / "products"
        / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5"
        / "ballistic_head_trail_correlated_fit.json",
    )
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "figures" / "head_echo_rcs_estimate_range_path.png")
    p.add_argument("--summary-tex", type=Path, default=ROOT / "reports" / "head_echo_rcs_path_summary.tex")
    p.add_argument("--metadata-json", type=Path, default=None)
    p.add_argument("--pt-w", type=float, default=500.0)
    p.add_argument("--tx-gain-db", type=float, default=7.0)
    p.add_argument("--rx-gain-db", type=float, default=3.0)
    p.add_argument("--system-temp-k", type=float, default=1.54e4)
    p.add_argument("--bandwidth-hz", type=float, default=100.0)
    p.add_argument("--snr-cutoff-db", type=float, default=15.0)
    args = p.parse_args()

    with h5py.File(args.input_h5, "r") as h:
        if args.station not in h:
            raise ValueError(f"{args.input_h5} does not contain station {args.station!r}")
        times, picked_range_km, snr_db = pick_path_snr_db(h[args.station], args.snr_dataset, args.half_width_px)

    r_tx_m, r_rx_m, frequency_hz = path_ranges_m(args.data_dir, args.station, args.fit_json, times)
    wavelength_m = C_MPS / frequency_hz
    noise_power_w = K_BOLTZMANN * args.system_temp_k * args.bandwidth_hz
    snr_linear = 10.0 ** (snr_db / 10.0)
    tx_gain = db_to_linear(args.tx_gain_db)
    rx_gain = db_to_linear(args.rx_gain_db)
    sigma_m2 = (
        snr_linear
        * noise_power_w
        * (4.0 * np.pi) ** 3
        * r_tx_m**2
        * r_rx_m**2
        / (args.pt_w * tx_gain * rx_gain * wavelength_m**2)
    )
    rcs_dbsm = 10.0 * np.log10(np.maximum(sigma_m2, 1e-60))
    finite = np.isfinite(rcs_dbsm) & np.isfinite(snr_db) & (snr_db >= args.snr_cutoff_db)

    rel_t = times - np.nanmin(times)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, constrained_layout=True)
    ax0.plot(rel_t[finite], rcs_dbsm[finite], color="#2166ac", lw=1.8)
    ax0.set_ylabel("RCS (dBsm)")
    ax0.grid(True, color="0.9", linewidth=0.6)
    ax0.set_title("Pomfret RCS from trajectory-steered, polarization-matched RDMF SNR")
    ax1.plot(rel_t[finite], snr_db[finite], color="#b2182b", lw=1.4)
    ax1.set_ylabel("Selected SNR (dB)")
    ax1.set_xlabel("Seconds after 2026-05-30 18:06:19 UTC")
    ax1.grid(True, color="0.9", linewidth=0.6)
    ax0.text(
        0.01,
        0.96,
        f"G_t={args.tx_gain_db:.1f} dB, G_r={args.rx_gain_db:.1f} dB, SNR >= {args.snr_cutoff_db:.1f} dB",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n": int(np.count_nonzero(finite)),
        "median_dbsm": float(np.nanmedian(rcs_dbsm[finite])),
        "p05_dbsm": float(np.nanpercentile(rcs_dbsm[finite], 5.0)),
        "p95_dbsm": float(np.nanpercentile(rcs_dbsm[finite], 95.0)),
        "min_dbsm": float(np.nanmin(rcs_dbsm[finite])),
        "max_dbsm": float(np.nanmax(rcs_dbsm[finite])),
        "median_snr_db": float(np.nanmedian(snr_db[finite])),
        "noise_power_w": float(noise_power_w),
        "snr_cutoff_db": float(args.snr_cutoff_db),
    }
    args.summary_tex.parent.mkdir(parents=True, exist_ok=True)
    args.summary_tex.write_text(
        "% Generated by reports/generate_pomfret_rcs_range_path.py\n"
        f"\\newcommand{{\\PomfretPathRcsN}}{{{stats['n']:,}}}\n"
        f"\\newcommand{{\\PomfretPathRcsMedianDbsm}}{{{stats['median_dbsm']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsFifthDbsm}}{{{stats['p05_dbsm']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsNinetyFifthDbsm}}{{{stats['p95_dbsm']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsMinDbsm}}{{{stats['min_dbsm']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsMaxDbsm}}{{{stats['max_dbsm']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsMedianSnrDb}}{{{stats['median_snr_db']:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsTxGainDb}}{{{args.tx_gain_db:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsRxGainDb}}{{{args.rx_gain_db:.1f}}}\n"
        f"\\newcommand{{\\PomfretPathRcsSnrCutoffDb}}{{{args.snr_cutoff_db:.1f}}}\n",
        encoding="utf-8",
    )
    metadata_path = args.metadata_json or args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "reports/generate_pomfret_rcs_range_path.py",
                "input_h5": str(args.input_h5),
                "station": args.station,
                "snr_dataset": args.snr_dataset,
                "selection": f"maximum within +/-{args.half_width_px} range pixels of model_range_km",
                "pt_w": args.pt_w,
                "tx_gain_db": args.tx_gain_db,
                "rx_gain_db": args.rx_gain_db,
                "system_temp_k": args.system_temp_k,
                "bandwidth_hz": args.bandwidth_hz,
                "snr_cutoff_db": args.snr_cutoff_db,
                "noise_power_w": noise_power_w,
                "frequency_hz": frequency_hz,
                "summary": stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(args.summary_tex)
    print(metadata_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
