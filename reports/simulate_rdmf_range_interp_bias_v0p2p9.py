#!/usr/bin/env python3
"""Synthetic v0.2.9 RDMF range-interpolation bias test.

This simulates a head echo from the v0.2.8 trajectory at 1 MHz, decimates to
the 100 kHz raw-voltage cadence, and runs the same trajectory-steered,
polarization-matched RDMF bank used for v0.2.9.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from range_doppler_mf_bolide import (  # noqa: E402
    bpsk,
    get_codes,
    interpolate_rect,
    load_trajectory,
    range_doppler_bank,
    read_config,
    trajectory_bistatic_range_m,
    trajectory_doppler_hz,
    tx_aod_steering_vector,
)


DEFAULT_TRAJECTORY = ROOT / "products" / "ballistic_head_trail_correlated_fit_v0p2p8_rinterp4_zp16" / "ballistic_head_trail_correlated_fit.json"
DEFAULT_V029 = ROOT / "products" / "sparse_residual_noise_power" / "head_echo_rdmf_txsteer_pol_rinterp4_zp16_3incoh_measurements_180618_180623.h5"
DEFAULT_NOISE = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_noise_power_1s.h5"
DEFAULT_OUT = ROOT / "products" / "synthetic_rdmf_bias_v0p2p9_rinterp4_zp16_3incoh_1mhz"
STATIONS = ("beverly", "clinton", "pomfret")


def unix_from_hms(hms: str) -> float:
    dt = datetime.fromisoformat(f"2026-05-30T{hms}+00:00")
    return dt.timestamp()


def residual_noise_power(noise_h5: Path, station: str, t0: float, t1: float) -> float:
    with h5py.File(noise_h5, "r") as h:
        g = h[station]
        t = g["time_unix"][()]
        p = g["dc_subtracted_power"][()]
    use = (t >= t0) & (t <= t1)
    if not np.any(use):
        use = np.ones_like(t, dtype=bool)
    return float(np.nanmedian(p[use]))


def v029_snr_profile(v029_h5: Path, station: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(v029_h5, "r") as h:
        g = h[station]
        t = g["time_unix"][()]
        snr = np.nanmax(g["txsteer_pol_snr_db"][()], axis=1)
    return np.asarray(t, dtype=np.float64), np.asarray(snr, dtype=np.float64)


def code_highres(seed: int, code_len: int, oversample: int) -> np.ndarray:
    code = bpsk(seed, code_len)
    code = code / np.max(np.abs(code))
    return np.repeat(code.astype(np.complex64), oversample)


def synthesize_station(
    station: str,
    data_dir: Path,
    trajectory: dict[str, np.ndarray],
    trajectory_path: Path,
    v029_h5: Path,
    noise_h5: Path,
    t0: float,
    t1: float,
    oversample: int,
    target_snr_percentile: float,
    snr_boost_db: float,
    fmin: float,
    fmax: float,
    rng: np.random.Generator,
) -> dict:
    config = read_config(data_dir, station)
    sample_rate = 100_000.0
    high_rate = sample_rate * oversample
    code_len = int(config["code_len"])
    seeds = np.asarray(config["seeds"], dtype=np.int64)
    dt = t1 - t0
    n_raw = int(round(dt * sample_rate))
    # Extra high-rate tail accommodates delayed code copies near the end.
    max_raw_delay = int(np.ceil(config["rmax_idx"] + 200))
    n_high = (n_raw + max_raw_delay + code_len + 4) * oversample
    noise_power = residual_noise_power(noise_h5, station, t0, t1)
    sigma100 = np.sqrt(noise_power / 2.0)
    z_high = np.zeros((2, n_high), dtype=np.complex64)
    high_codes = [code_highres(int(seed), code_len, oversample) for seed in seeds]
    raw_codes = [bpsk(int(seed), code_len) / 1.0 for seed in seeds]

    t_snr, snr_db = v029_snr_profile(v029_h5, station)
    target_snr_db = float(np.nanpercentile(snr_db, target_snr_percentile))
    target_snr_db += float(snr_boost_db)
    target_snr_linear = 10.0 ** (target_snr_db / 10.0)
    # Matched-filter peak power is roughly |amp|^2 * code_len^2 * n_tx after
    # the tx-steered sum and /n_tx normalization. This sets the right order of
    # SNR before the script measures the actual recovered value.
    amp = np.sqrt(max(noise_power, 1e-30) * target_snr_linear / max(code_len**2 * len(seeds), 1))
    pol = np.array([1.0 + 0.0j, 0.55 * np.exp(1.0j * 0.7)], dtype=np.complex64)
    pol = pol / np.linalg.norm(pol)

    step_times = t0 + np.arange(int(np.floor(dt * 100.0)), dtype=np.float64) * 0.01
    true_range_m = trajectory_bistatic_range_m(config, trajectory, step_times)
    true_doppler_hz = trajectory_doppler_hz(config, str(data_dir), station, trajectory, step_times)
    true_range_raw_idx = true_range_m / 3000.0
    delay_high = np.rint(true_range_raw_idx * oversample).astype(np.int64)

    for i, t_step in enumerate(step_times):
        start_high = int(round((t_step - t0) * high_rate))
        dop = float(true_doppler_hz[i])
        phase = np.exp(2.0j * np.pi * dop * (np.arange(code_len * oversample) / high_rate)).astype(np.complex64)
        steering, _unit = tx_aod_steering_vector(config, trajectory, float(t_step), phase_sign=-1.0)
        for tx_i, code in enumerate(high_codes):
            lo = start_high + int(delay_high[i])
            hi = lo + len(code)
            if lo < 0 or hi > n_high:
                continue
            echo = amp * steering[tx_i] * code * phase
            z_high[0, lo:hi] += pol[0] * echo
            z_high[1, lo:hi] += pol[1] * echo

    # Decimate by integrating over each 10 us raw-voltage sample. Add noise at
    # the 100 kHz level after decimation so the noise power matches the measured
    # sparse-residual voltage power used in the memo.
    z100 = z_high[:, : n_raw * oversample].reshape(2, n_raw, oversample).mean(axis=2)
    noise = sigma100 * (rng.standard_normal(z100.shape) + 1.0j * rng.standard_normal(z100.shape))
    z100 = (z100 + noise).astype(np.complex64)
    z_proc = [interpolate_rect(z100[ch], 4) for ch in range(2)]

    t_grid = np.linspace(t0, t1, int(np.ceil(dt / 0.01)) + 1)
    full_range = trajectory_bistatic_range_m(config, trajectory, t_grid)
    gate_m = 3000.0 / 4.0
    rmin_idx = max(0, int(np.floor(np.nanmin(full_range / gate_m))) - int(round(20_000.0 / gate_m)))
    rmax_idx = int(np.ceil(np.nanmax(full_range / gate_m))) + int(round(20_000.0 / gate_m))

    result = range_doppler_bank(
        channel_data=z_proc,
        sample_rate=sample_rate,
        abs_index0=int(round(t0 * sample_rate)),
        codes=get_codes(seeds, code_len, 1, range_interp=4),
        code_len=code_len,
        rmin_idx=rmin_idx,
        rmax_idx=rmax_idx,
        range_resolution_m=3000.0,
        range_interp=4,
        n_coh=1,
        codes_per_step=1,
        fmin=fmin,
        fmax=fmax,
        ignore_freq=150.0,
        head_min_doppler_hz=0.0,
        doppler_zero_pad_factor=16,
        tx_aod_beamform=True,
        tx_aod_trajectory=trajectory,
        tx_aod_config=config,
        tx_aod_phase_sign=-1.0,
        polarization_combine=True,
        head_incoherent_codes=3,
        head_search_range_half_km=20.0,
        head_search_doppler_half_hz=1000.0,
        data_dir=str(data_dir),
        station=station,
        progress_every=100,
        checkpoint_every=0,
    )

    time_unix = result["t_sample"].astype(np.float64) / sample_rate
    truth_range = trajectory_bistatic_range_m(config, trajectory, time_unix)
    truth_doppler = trajectory_doppler_hz(config, str(data_dir), station, trajectory, time_unix)
    peak_ri = result["pol_peak_range_idx"].astype(np.int64)
    peak_di = result["pol_peak_doppler_idx"].astype(np.int64)
    rec_range_m = result["range_km"][peak_ri].astype(np.float64) * 1000.0
    rec_doppler = result["fvec"][peak_di].astype(np.float64)
    snr_img = result["P_head"] / np.maximum(result["N"], 1e-30)
    rec_snr_db = 10.0 * np.log10(np.maximum(snr_img[np.arange(len(peak_ri)), peak_ri], 1e-30))

    return {
        "station": station,
        "time_unix": time_unix,
        "truth_range_m": truth_range,
        "truth_doppler_hz": truth_doppler,
        "recovered_range_m": rec_range_m,
        "recovered_doppler_hz": rec_doppler,
        "range_error_m": rec_range_m - truth_range,
        "doppler_error_hz": rec_doppler - truth_doppler,
        "snr_db": rec_snr_db,
        "target_snr_db": np.full_like(rec_snr_db, target_snr_db),
        "noise_power_100khz": np.full_like(rec_snr_db, noise_power),
        "amplitude": np.full_like(rec_snr_db, amp),
        "range_km": result["range_km"],
        "fvec": result["fvec"],
        "P_head": result["P_head"],
        "D_head": result["D_head"],
        "N": result["N"],
    }


def write_outputs(out_dir: Path, products: list[dict], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "synthetic_rdmf_range_interp_bias.h5"
    csv_path = out_dir / "synthetic_rdmf_range_interp_bias_summary.csv"
    fig_path = out_dir / "synthetic_rdmf_range_interp_bias.png"

    with h5py.File(h5_path.with_suffix(".h5.tmp"), "w") as h:
        h.attrs["source"] = "reports/simulate_rdmf_range_interp_bias_v0p2p9.py"
        h.attrs["description"] = "1 MHz synthetic echo decimated to 100 kHz, processed by v0.2.9 RDMF settings"
        h.attrs["trajectory_json"] = str(args.trajectory_json)
        h.attrs["v029_measurement_h5"] = str(args.v029_measurement_h5)
        h.attrs["noise_h5"] = str(args.noise_h5)
        h.attrs["oversample"] = int(args.oversample)
        h.attrs["range_interp"] = 4
        h.attrs["doppler_zero_pad_factor"] = 16
        h.attrs["head_incoherent_codes"] = 3
        for product in products:
            g = h.create_group(product["station"])
            for key, val in product.items():
                if key == "station":
                    continue
                g.create_dataset(key, data=val)
    h5_path.with_suffix(".h5.tmp").replace(h5_path)

    rows = []
    for p in products:
        use = np.isfinite(p["range_error_m"]) & np.isfinite(p["doppler_error_hz"])
        rows.append(
            {
                "station": p["station"],
                "n": int(np.sum(use)),
                "snr_db_median": float(np.nanmedian(p["snr_db"][use])),
                "snr_db_p95": float(np.nanpercentile(p["snr_db"][use], 95)),
                "range_error_m_median": float(np.nanmedian(p["range_error_m"][use])),
                "range_error_m_p05": float(np.nanpercentile(p["range_error_m"][use], 5)),
                "range_error_m_p95": float(np.nanpercentile(p["range_error_m"][use], 95)),
                "doppler_error_hz_median": float(np.nanmedian(p["doppler_error_hz"][use])),
                "doppler_error_hz_p05": float(np.nanpercentile(p["doppler_error_hz"][use], 5)),
                "doppler_error_hz_p95": float(np.nanpercentile(p["doppler_error_hz"][use], 95)),
            }
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True, constrained_layout=True)
    for p in products:
        t_rel = p["time_unix"] - unix_from_hms("18:06:20")
        axes[0].plot(t_rel, p["range_error_m"], ".", ms=2.5, alpha=0.65, label=p["station"])
        axes[1].plot(t_rel, p["doppler_error_hz"], ".", ms=2.5, alpha=0.65, label=p["station"])
    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[1].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel("Recovered - true range (m)")
    axes[1].set_ylabel("Recovered - true Doppler (Hz)")
    axes[1].set_xlabel("Time relative to 2026-05-30 18:06:20 UTC (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Synthetic v0.2.9 RDMF bias test: 1 MHz echo -> 100 kHz voltage -> range_interp=4")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {h5_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    print(json.dumps(rows, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    p.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--v029-measurement-h5", type=Path, default=DEFAULT_V029)
    p.add_argument("--noise-h5", type=Path, default=DEFAULT_NOISE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--t0", default="18:06:18")
    p.add_argument("--t1", default="18:06:23")
    p.add_argument("--oversample", type=int, default=10)
    p.add_argument("--target-snr-percentile", type=float, default=75.0)
    p.add_argument("--snr-boost-db", type=float, default=0.0)
    p.add_argument("--fmin", type=float, default=-5000.0)
    p.add_argument("--fmax", type=float, default=2000.0)
    p.add_argument("--mpi-dispatch", action="store_true")
    p.add_argument("--seed", type=int, default=20260613)
    args = p.parse_args()

    trajectory = load_trajectory(args.trajectory_json)
    rank = 0
    size = 1
    comm = None
    if args.mpi_dispatch:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    stations = [station for i, station in enumerate(args.stations) if i % size == rank]
    rng = np.random.default_rng(args.seed + rank)
    t0 = unix_from_hms(args.t0)
    t1 = unix_from_hms(args.t1)
    products = []
    for station in stations:
        print(f"{station}: synthetic 1 MHz echo -> 100 kHz -> v0.2.9 RDMF", flush=True)
        products.append(
            synthesize_station(
                station=station,
                data_dir=args.data_dir,
                trajectory=trajectory,
                trajectory_path=args.trajectory_json,
                v029_h5=args.v029_measurement_h5,
                noise_h5=args.noise_h5,
                t0=t0,
                t1=t1,
                oversample=args.oversample,
                target_snr_percentile=args.target_snr_percentile,
                snr_boost_db=args.snr_boost_db,
                fmin=args.fmin,
                fmax=args.fmax,
                rng=rng,
            )
        )
    if args.mpi_dispatch:
        gathered = comm.gather(products, root=0)
        if rank != 0:
            return
        products = [product for chunk in gathered for product in chunk]
    write_outputs(args.out_dir, products, args)


if __name__ == "__main__":
    main()
