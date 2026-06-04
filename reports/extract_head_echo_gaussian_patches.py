#!/usr/bin/env python3
"""Extract head-echo range-Doppler matched-filter patches for later fitting."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml  # noqa: E402
from estimate_head_echo_trajectory import predict_dopplers, predict_ranges, station_geometry  # noqa: E402
from reports.generate_head_microdoppler import STATIONS, load_station_observations  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    DATA_DIR,
    EVENT_T0,
    compute_filtered_patch,
    init_station,
)

DEFAULT_ANNOTATIONS = ROOT / "annotations" / "head_echo_rois.json"
DEFAULT_OUT = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code.h5"


def common_time_grid(annotations_path: Path, stations: list[str], pad_s: float, dt_s: float) -> np.ndarray:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    spans = []
    for station in stations:
        entry = payload["stations"][station]
        spans.append((float(entry["time_range"]["start_unix"]), float(entry["time_range"]["end_unix"])))
    t0 = min(t[0] for t in spans) - pad_s
    t1 = max(t[1] for t in spans) + pad_s
    n = int(np.floor((t1 - t0) / dt_s + 0.5)) + 1
    return t0 + dt_s * np.arange(n, dtype=float)


def predicted_observations(payload: dict, stations: list[str], times: np.ndarray) -> dict[str, list[dict]]:
    geom = station_geometry(DATA_DIR, stations)
    tx_site = load_station_yaml(DATA_DIR, payload["used_stations"][0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        payload["fit_parameters_enu_at_tx"]["position_ref_m"],
        payload["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    out = {}
    for station in stations:
        model_obs = [{"station": station, "unix_time": float(t)} for t in times]
        ranges_m = predict_ranges(params, model_obs, geom, payload["t_ref_unix"], tx0, rot)
        dopplers_hz = predict_dopplers(params, model_obs, geom, payload["t_ref_unix"], tx0, rot)
        rows = [
            {"station": station, "unix_time": float(t), "range_m": float(r), "doppler_hz": float(d)}
            for t, r, d in zip(times, ranges_m, dopplers_hz)
        ]
        out[station] = rows
    return out


def process_item(item: tuple[int, dict]) -> dict | None:
    patch = compute_filtered_patch(item)
    if patch is None:
        idx, obs = item
        return {
            "idx": int(idx),
            "time": float(obs["unix_time"]),
            "seed_range_m": float(obs["range_m"]),
            "picked_doppler_hz": float(obs["doppler_hz"]),
            "valid": False,
        }
    patch["valid"] = True
    return patch


def write_station_group(h: h5py.File, station: str, rows: list[dict]) -> None:
    g = h.create_group(station)
    g.attrs["n_patches"] = len(rows)
    for row in rows:
        p = g.create_group(f"patch_{int(row['idx']):04d}")
        p.attrs["idx"] = int(row["idx"])
        p.attrs["time_unix"] = float(row["time"])
        p.attrs["time_rel_s"] = float(row["time"]) - EVENT_T0
        p.attrs["seed_range_m"] = float(row["seed_range_m"])
        p.attrs["picked_doppler_hz"] = float(row["picked_doppler_hz"])
        p.attrs["valid"] = bool(row.get("valid", False))
        if not row.get("valid", False):
            continue
        p.attrs["model_doppler_hz"] = float(row["model_doppler_hz"])
        p.create_dataset("range_m", data=np.asarray(row["range_m"], dtype=np.float32), compression="gzip", compression_opts=1)
        p.create_dataset("rel_freqs", data=np.asarray(row["rel_freqs"], dtype=np.float32), compression="gzip", compression_opts=1)
        p.create_dataset("raw_db", data=np.asarray(row["raw_db"], dtype=np.float32), compression="gzip", compression_opts=1)


def run_station(station: str, observations: list[dict], payload: dict, args: argparse.Namespace) -> list[dict]:
    print(f"{station}: loading raw data", flush=True)
    init_station(payload, station, args)
    items = list(enumerate(observations))
    n_workers = min(max(1, args.workers), len(items))
    print(f"{station}: extracting {len(items)} predicted patches with {n_workers} worker(s)", flush=True)
    if n_workers == 1:
        rows = [process_item(item) for item in items]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            rows = []
            for row in pool.imap(process_item, items, chunksize=1):
                rows.append(row)
                done = len(rows)
                if done == 1 or done == len(items) or done % max(1, len(items) // 8) == 0:
                    print(f"{station}: {done}/{len(items)} patches returned", flush=True)
    return [row for row in rows if row is not None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--time-pad-s", type=float, default=0.2)
    p.add_argument("--time-step-s", type=float, default=0.01)
    p.add_argument(
        "--retained-head-times",
        action="store_true",
        help="Use retained head-echo observation times, but seed range/Doppler from the same trajectory prediction used in trail_echo_roi.py.",
    )
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=16)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    payload, manual_observations = load_station_observations()
    stations = list(args.stations)
    if args.retained_head_times:
        observations = {}
        common_times = []
        for station in stations:
            times = np.asarray([float(obs["unix_time"]) for obs in manual_observations[station]], dtype=float)
            observations[station] = predicted_observations(payload, [station], times)[station]
            common_times.extend(times.tolist())
        times = np.asarray(sorted(set(common_times)), dtype=float)
    else:
        times = common_time_grid(args.annotations, stations, args.time_pad_s, args.time_step_s)
        observations = predicted_observations(payload, stations, times)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.attrs["description"] = "Predicted-grid head-echo range-Doppler matched-filter patches for local 2D FFT notch and Gaussian fitting"
        h.attrs["event_t0_unix"] = EVENT_T0
        h.attrs["time_pad_s"] = args.time_pad_s
        h.attrs["time_step_s"] = args.time_step_s
        h.attrs["retained_head_times"] = bool(args.retained_head_times)
        h.attrs["n_coh"] = args.n_coh
        h.attrs["n_incoh"] = args.n_incoh
        h.attrs["step_codes"] = args.step_codes
        h.attrs["range_half_width_km"] = args.range_half_width_km
        h.attrs["doppler_half_width_hz"] = args.doppler_half_width_hz
        h.create_dataset("common_time_unix", data=times)
        h["stations"] = np.asarray([s.encode("utf-8") for s in stations])
        for station in stations:
            rows = run_station(station, observations[station], payload, args)
            rows.sort(key=lambda row: row["idx"])
            write_station_group(h, station, rows)
            h.flush()
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
