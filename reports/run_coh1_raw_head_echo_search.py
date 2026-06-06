#!/usr/bin/env python3
"""Run a one-code, unwhitened head-echo Gaussian search."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import fit_gaussian_patch, smooth_power_db  # noqa: E402
from reports.generate_head_microdoppler import STATIONS, detrend_window, load_station_observations  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, G, best_fit_doppler_hz, init_station  # noqa: E402

OUT_H5 = ROOT / "reports" / "coh1_raw_head_echo_test" / "cache" / "coh1_raw_head_echo_search.h5"
OUT_CSV = ROOT / "reports" / "coh1_raw_head_echo_test" / "coh1_raw_head_echo_search_summary.csv"


def compute_raw_patch(idx: int, obs: dict, args: argparse.Namespace) -> dict:
    station = G["station"]
    bulk = best_fit_doppler_hz(G["doppler_model"], station, float(obs["unix_time"]))
    fvec_full = G["fvec_full"]
    fidx = np.where(np.abs(fvec_full - bulk) <= args.doppler_half_width_hz)[0]
    rel_f = fvec_full[fidx] - bulk

    gate_m = G["gate_m"]
    seed_range_m = float(obs["range_m"])
    ridx = np.arange(
        max(0, int(np.floor((seed_range_m - args.range_half_width_km * 1000.0) / gate_m))),
        int(np.ceil((seed_range_m + args.range_half_width_km * 1000.0) / gate_m)) + 1,
        dtype=np.int64,
    )

    raw_sample = int(round(float(obs["unix_time"]) * G["sample_rate"] - G["t0_sample"]))
    start_i = raw_sample * G["range_interp"]
    read_len = G["win_len"] + int(ridx[-1]) + 1
    incoh_stop_i = start_i + (args.n_incoh - 1) * G["step_len"] + read_len
    if start_i < 0 or incoh_stop_i >= min(map(len, G["zc"])):
        raise RuntimeError("Selected observation is outside the available RF span")

    power = np.zeros((len(ridx), len(fidx)), dtype=np.float64)
    for incoh_i in range(args.n_incoh):
        incoh_start_i = start_i + incoh_i * G["step_len"]
        for z in G["zc"]:
            z0 = detrend_window(z[incoh_start_i : incoh_start_i + read_len])
            windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
            for code in G["codes"]:
                zspec = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
                power += np.abs(zspec) ** 2

    return {
        "idx": idx,
        "station": station,
        "time_rel": float(obs["unix_time"]) - EVENT_T0,
        "range_m": ridx.astype(np.float64) * gate_m,
        "rel_freqs": rel_f,
        "raw_power": power,
        "seed_range_m": seed_range_m,
        "seed_doppler_hz": float(obs["doppler_hz"]),
        "model_doppler_hz": bulk,
    }


def db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-30))


def fit_patch(patch: dict, obs: dict, args: argparse.Namespace) -> dict:
    raw_db = db(patch["raw_power"])
    smoothed_db = smooth_power_db(raw_db, args.smooth_range_pixels, args.smooth_frequency_pixels)
    seed_range_km = float(obs["range_m"]) / 1000.0
    seed_rel_hz = float(obs["doppler_hz"]) - float(patch["model_doppler_hz"])
    fit = fit_gaussian_patch(
        patch["range_m"] / 1000.0,
        patch["rel_freqs"],
        smoothed_db,
        seed_range_km,
        seed_rel_hz,
        args.search_range_half_width_km,
        args.search_doppler_half_width_hz,
        args.fit_range_half_width_km,
        args.fit_doppler_half_width_hz,
    )
    fit["absolute_doppler_hz"] = float(patch["model_doppler_hz"]) + fit["rel_doppler_hz"]
    fit["range_m"] = fit["range_km"] * 1000.0
    fit["range_correction_m"] = fit["range_m"] - float(obs["range_m"])
    fit["doppler_correction_hz"] = fit["absolute_doppler_hz"] - float(obs["doppler_hz"])
    fit["seed_rel_doppler_hz"] = seed_rel_hz
    return fit


def process_item(item: tuple[int, dict]) -> dict:
    idx, obs = item
    try:
        patch = compute_raw_patch(idx, obs, G["args"])
        fit = fit_patch(patch, obs, G["args"])
        return {"idx": idx, "obs": obs, "patch": patch, "fit": fit, "error": ""}
    except Exception as exc:
        return {"idx": idx, "obs": obs, "patch": None, "fit": None, "error": str(exc)}


def row_from_result(station: str, result: dict, args: argparse.Namespace) -> dict:
    obs = result["obs"]
    patch = result["patch"]
    fit = result["fit"]
    row = {
        "station": station,
        "idx": result["idx"],
        "time_unix": float(obs["unix_time"]),
        "time_rel_s": float(obs["unix_time"]) - EVENT_T0,
        "seed_range_m": float(obs["range_m"]),
        "seed_doppler_hz": float(obs["doppler_hz"]),
        "success": bool(fit is not None and fit.get("success", False)),
        "error": result["error"],
        "n_coh": args.n_coh,
        "n_incoh": args.n_incoh,
        "step_codes": args.step_codes,
        "smooth_range_pixels": args.smooth_range_pixels,
        "smooth_frequency_pixels": args.smooth_frequency_pixels,
    }
    if patch is not None:
        row["model_doppler_hz"] = float(patch["model_doppler_hz"])
    if fit is not None:
        row.update(
            {
                "fit_range_m": fit["range_m"],
                "range_correction_m": fit["range_correction_m"],
                "fit_doppler_hz": fit["absolute_doppler_hz"],
                "doppler_correction_hz": fit["doppler_correction_hz"],
                "seed_rel_doppler_hz": fit["seed_rel_doppler_hz"],
                "fit_rel_doppler_hz": fit["rel_doppler_hz"],
                "sigma_range_m": 1000.0 * fit["sigma_range_km"],
                "sigma_doppler_hz": fit["sigma_doppler_hz"],
                "range_doppler_correlation": fit["range_doppler_correlation"],
                "amplitude_db": fit["amplitude_db"],
                "cost": fit["cost"],
            }
        )
    return row


def write_result_group(parent: h5py.Group, result: dict) -> None:
    g = parent.create_group(f"patch_{int(result['idx']):04d}")
    obs = result["obs"]
    g.attrs["idx"] = int(result["idx"])
    g.attrs["time_unix"] = float(obs["unix_time"])
    g.attrs["time_rel"] = float(obs["unix_time"]) - EVENT_T0
    g.attrs["seed_range_m"] = float(obs["range_m"])
    g.attrs["seed_doppler_hz"] = float(obs["doppler_hz"])
    g.attrs["valid"] = result["patch"] is not None
    g.attrs["success"] = result["fit"] is not None and bool(result["fit"].get("success", False))
    g.attrs["error"] = result["error"]
    patch = result["patch"]
    if patch is not None:
        g.attrs["model_doppler_hz"] = float(patch["model_doppler_hz"])
        for key in ("range_m", "rel_freqs", "raw_power"):
            g.create_dataset(key, data=patch[key], compression="gzip", compression_opts=3)
    fit = result["fit"]
    if fit is not None:
        fg = g.create_group("gaussian_fit")
        for key, val in fit.items():
            if isinstance(val, str):
                fg.attrs[key] = val
            else:
                fg.attrs[key] = float(val) if np.ndim(val) == 0 else val


def write_csv(path: Path, rows: list[dict]) -> None:
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
        "seed_rel_doppler_hz",
        "fit_rel_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "range_doppler_correlation",
        "amplitude_db",
        "cost",
        "success",
        "error",
        "n_coh",
        "n_incoh",
        "step_codes",
        "smooth_range_pixels",
        "smooth_frequency_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_station(station: str, observations: list[dict], payload: dict, args: argparse.Namespace, out_h5: h5py.File) -> list[dict]:
    print(f"{station}: initializing", flush=True)
    init_station(payload, station, args)
    items = list(enumerate(observations))
    if args.limit is not None:
        items = items[: args.limit]
    n_workers = min(args.workers, len(items))
    print(f"{station}: processing {len(items)} observations with {n_workers} workers", flush=True)
    if n_workers <= 1:
        results = [process_item(item) for item in items]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = []
            for result in pool.imap_unordered(process_item, items, chunksize=1):
                results.append(result)
                done = len(results)
                if done == 1 or done == len(items) or done % max(1, len(items) // 10) == 0:
                    print(f"{station}: {done}/{len(items)} returned", flush=True)
    results.sort(key=lambda res: res["idx"])
    sg = out_h5.create_group(station)
    rows = []
    for result in results:
        write_result_group(sg, result)
        rows.append(row_from_result(station, result, args))
    ok = sum(1 for row in rows if row["success"])
    print(f"{station}: {ok}/{len(rows)} Gaussian fits succeeded", flush=True)
    out_h5.flush()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-h5", type=Path, default=OUT_H5)
    p.add_argument("--summary", type=Path, default=OUT_CSV)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=1)
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=1.0)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    args = p.parse_args()

    payload, observations = load_station_observations()
    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with h5py.File(args.out_h5, "w") as h:
        h.attrs["description"] = "One-code unwhitened head-echo Gaussian search"
        h.attrs["event_t0_unix"] = EVENT_T0
        h.attrs["n_coh"] = args.n_coh
        h.attrs["n_incoh"] = args.n_incoh
        h.attrs["step_codes"] = args.step_codes
        h.attrs["smooth_range_pixels"] = args.smooth_range_pixels
        h.attrs["smooth_frequency_pixels"] = args.smooth_frequency_pixels
        h.create_dataset("stations", data=np.asarray([s.encode("utf-8") for s in args.stations]))
        for station in args.stations:
            all_rows.extend(run_station(station, observations[station], payload, args, h))
    write_csv(args.summary, all_rows)
    print(f"Wrote {args.out_h5}", flush=True)
    print(f"Wrote {args.summary}", flush=True)
    print(f"{sum(row['success'] for row in all_rows)}/{len(all_rows)} total Gaussian fits succeeded", flush=True)


if __name__ == "__main__":
    main()
