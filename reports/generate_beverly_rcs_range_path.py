#!/usr/bin/env python3
"""Generate Beverly head-echo RCS from the corrected SNR progression trace."""

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

DEFAULT_INPUT = ROOT / "products" / "sparse_residual_noise_power" / "head_echo_snr_rdmf_progression_stage_noise_beverly_zoom.h5"
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "head_echo_rcs_estimate_range_path.png"
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_rcs_path_summary.tex"
DEFAULT_FIT = (
    ROOT
    / "products"
    / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5"
    / "ballistic_head_trail_correlated_fit.json"
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


def altitude_km_from_fit(fit_json: Path, times_unix: np.ndarray) -> np.ndarray:
    payload = json.loads(fit_json.read_text(encoding="utf-8"))
    model = payload["model"]
    tm = np.asarray(model["time_unix"], dtype=float)
    alt_m = np.asarray(model["alt_m"], dtype=float)
    order = np.argsort(tm)
    return np.interp(times_unix, tm[order], alt_m[order]) / 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--station", default="beverly")
    p.add_argument("--snr-dataset", default="txsteer_pol_snr_db")
    p.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    p.add_argument("--fit-json", type=Path, default=DEFAULT_FIT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summary-tex", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--metadata-json", type=Path, default=None)
    p.add_argument("--pt-w", type=float, default=500.0)
    p.add_argument("--tx-gain-db", type=float, default=7.0)
    p.add_argument("--rx-gain-db", type=float, default=3.0)
    p.add_argument("--system-temp-k", type=float, default=1.54e4)
    p.add_argument("--bandwidth-hz", type=float, default=100.0)
    p.add_argument("--snr-cutoff-db", type=float, default=None)
    p.add_argument("--xlim-start-rel", type=float, default=-1.0)
    p.add_argument("--xlim-end-rel", type=float, default=2.0)
    p.add_argument(
        "--altitude-ticks-km",
        type=float,
        nargs="+",
        default=[110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
        help="Altitude values shown on the top axis; 90 and 70 km bracket the head-echo interval.",
    )
    args = p.parse_args()

    with h5py.File(args.input_h5, "r") as h:
        times = np.asarray(h["time_unix"][()], dtype=float)
        rel_t = np.asarray(h["time_rel_s"][()], dtype=float)
        snr_db = np.asarray(h[args.snr_dataset][()], dtype=float)
        input_attrs = {key: str(val) for key, val in h.attrs.items()}

    r_tx_m, r_rx_m, frequency_hz = path_ranges_m(args.data_dir, args.station, args.fit_json, times)
    altitude_km = altitude_km_from_fit(args.fit_json, times)
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
    finite = (
        np.isfinite(rcs_dbsm)
        & np.isfinite(snr_db)
        & (rel_t >= float(args.xlim_start_rel))
        & (rel_t <= float(args.xlim_end_rel))
    )
    if args.snr_cutoff_db is not None:
        finite &= snr_db >= float(args.snr_cutoff_db)
    if not np.any(finite):
        raise ValueError("No finite Beverly samples in requested time interval")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.5), constrained_layout=True)
    ax.plot(rel_t[finite], rcs_dbsm[finite], color="#2166ac", lw=1.8)
    ax.set_xlim(float(args.xlim_start_rel), float(args.xlim_end_rel))
    ax.set_ylabel("RCS (dBsm)")
    ax.set_xlabel("Seconds after 18:06:20 UTC")
    ax.grid(True, color="0.9", linewidth=0.6)
    top = ax.secondary_xaxis("top")
    top.set_xlabel("Altitude (km)")
    finite_rel_t = rel_t[finite]
    finite_altitude = altitude_km[finite]
    if finite_rel_t.size:
        altitude_order = np.argsort(finite_altitude)
        altitude_sorted = finite_altitude[altitude_order]
        rel_t_by_altitude = finite_rel_t[altitude_order]
        requested_altitudes = np.asarray(args.altitude_ticks_km, dtype=float)
        in_range = (requested_altitudes >= np.nanmin(altitude_sorted)) & (
            requested_altitudes <= np.nanmax(altitude_sorted)
        )
        tick_altitudes = requested_altitudes[in_range]
        tick_times = np.interp(tick_altitudes, altitude_sorted, rel_t_by_altitude)
        top.set_xticks(tick_times)
        top.set_xticklabels([f"{alt:g}" for alt in tick_altitudes])
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
        "snr_cutoff_db": None if args.snr_cutoff_db is None else float(args.snr_cutoff_db),
        "xlim_start_rel": float(args.xlim_start_rel),
        "xlim_end_rel": float(args.xlim_end_rel),
    }

    args.summary_tex.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "% Generated by reports/generate_beverly_rcs_range_path.py\n",
        f"\\newcommand{{\\BeverlyPathRcsN}}{{{stats['n']:,}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsMedianDbsm}}{{{stats['median_dbsm']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsFifthDbsm}}{{{stats['p05_dbsm']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsNinetyFifthDbsm}}{{{stats['p95_dbsm']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsMinDbsm}}{{{stats['min_dbsm']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsMaxDbsm}}{{{stats['max_dbsm']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsMedianSnrDb}}{{{stats['median_snr_db']:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsTxGainDb}}{{{args.tx_gain_db:.1f}}}\n",
        f"\\newcommand{{\\BeverlyPathRcsRxGainDb}}{{{args.rx_gain_db:.1f}}}\n",
    ]
    if args.snr_cutoff_db is None:
        summary_lines.append("\\newcommand{\\BeverlyPathRcsSnrCutoffDb}{none}\n")
    else:
        summary_lines.append(f"\\newcommand{{\\BeverlyPathRcsSnrCutoffDb}}{{{float(args.snr_cutoff_db):.1f}}}\n")
    args.summary_tex.write_text("".join(summary_lines), encoding="utf-8")
    metadata_path = args.metadata_json or args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": "reports/generate_beverly_rcs_range_path.py",
                "input_h5": str(args.input_h5),
                "input_attrs": input_attrs,
                "station": args.station,
                "snr_dataset": args.snr_dataset,
                "selection": "modeled Beverly range bin from stage-noise-normalized trace",
                "pt_w": args.pt_w,
                "tx_gain_db": args.tx_gain_db,
                "rx_gain_db": args.rx_gain_db,
                "system_temp_k": args.system_temp_k,
                "bandwidth_hz": args.bandwidth_hz,
                "snr_cutoff_db": args.snr_cutoff_db,
                "xlim_start_rel": args.xlim_start_rel,
                "xlim_end_rel": args.xlim_end_rel,
                "altitude_ticks_km": args.altitude_ticks_km,
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
