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
from reports.generate_head_echo_gaussian_example import (  # noqa: E402
    fit_gaussian_patch,
    gaussian_one_sigma_mask,
    smooth_power_db,
)
import reports.generate_head_microdoppler_2dfft_notched as md2d  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    DATA_DIR,
    DEFAULT_SPARSE_PRODUCT_DIR,
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


def restrict_observations(observations: dict[str, list[dict]], args: argparse.Namespace) -> dict[str, list[dict]]:
    start = args.time_rel_start_s
    stop = args.time_rel_stop_s
    if start is None and stop is None:
        return observations
    restricted = {}
    for station, rows in observations.items():
        keep = []
        for row in rows:
            time_rel = float(row["unix_time"]) - EVENT_T0
            if start is not None and time_rel < start:
                continue
            if stop is not None and time_rel > stop:
                continue
            keep.append(row)
        restricted[station] = keep
    return restricted


def reference_polarization_mask(item: tuple[int, dict], args: argparse.Namespace) -> tuple[np.ndarray, dict] | tuple[None, None]:
    if not (
        bool(getattr(args, "polarization_combine", False))
        and str(getattr(args, "polarization_method", "phase")) == "grid_image"
        and str(getattr(args, "polarization_image_metric", "peak")) == "gaussian_1sigma"
    ):
        return None, None
    old_combine = bool(args.polarization_combine)
    old_mask = md2d.G.pop("polarization_metric_mask", None)
    args.polarization_combine = False
    try:
        patch = compute_filtered_patch(item)
    finally:
        args.polarization_combine = old_combine
        if old_mask is not None:
            md2d.G["polarization_metric_mask"] = old_mask
    if patch is None:
        return None, None
    _idx, obs = item
    filtered_db = smooth_power_db(
        patch["filtered_db"],
        float(getattr(args, "smooth_range_pixels", 4.0)),
        float(getattr(args, "smooth_frequency_pixels", 16.0)),
    )
    seed_range_km = float(obs["range_m"]) / 1000.0
    seed_rel_hz = float(obs["doppler_hz"]) - float(patch["model_doppler_hz"])
    fit = fit_gaussian_patch(
        patch["range_m"] / 1000.0,
        patch["rel_freqs"],
        filtered_db,
        seed_range_km,
        seed_rel_hz,
        float(getattr(args, "search_range_half_width_km", 10.0)),
        float(getattr(args, "search_doppler_half_width_hz", 700.0)),
        float(getattr(args, "fit_range_half_width_km", 5.0)),
        float(getattr(args, "fit_doppler_half_width_hz", 300.0)),
    )
    mask = gaussian_one_sigma_mask(patch["range_m"] / 1000.0, patch["rel_freqs"], fit)
    return mask, fit


def process_item(item: tuple[int, dict]) -> dict | None:
    args = md2d.G["args"]
    ref_fit = None
    ref_mask, ref_fit = reference_polarization_mask(item, args)
    if ref_mask is not None:
        md2d.G["polarization_metric_mask"] = ref_mask
    patch = compute_filtered_patch(item)
    md2d.G.pop("polarization_metric_mask", None)
    if patch is None:
        idx, obs = item
        return {
            "idx": int(idx),
            "time": float(obs["unix_time"]),
            "seed_range_m": float(obs["range_m"]),
            "picked_doppler_hz": float(obs["doppler_hz"]),
            "valid": False,
        }
    if ref_mask is not None:
        patch["polarization_metric_mask"] = ref_mask
    if ref_fit is not None:
        patch["polarization_reference_fit"] = ref_fit
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
        p.attrs["coherent_window_s"] = float(row.get("coherent_window_s", np.nan))
        p.attrs["total_window_s"] = float(row.get("total_window_s", np.nan))
        p.attrs["n_coh"] = int(row.get("n_coh", -1))
        p.attrs["n_incoh"] = int(row.get("n_incoh", -1))
        p.attrs["step_codes"] = int(row.get("step_codes", -1))
        p.attrs["doppler_zero_pad_factor"] = int(row.get("doppler_zero_pad_factor", -1))
        if "channel_noise_equalizer" in row and len(row["channel_noise_equalizer"]):
            p.create_dataset("channel_noise_equalizer", data=np.asarray(row["channel_noise_equalizer"], dtype=np.float32))
        if "channel_noise_powers" in row and len(row["channel_noise_powers"]):
            p.create_dataset("channel_noise_powers", data=np.asarray(row["channel_noise_powers"], dtype=np.float64))
        if "polarization_gamma" in row and len(row["polarization_gamma"]):
            p.create_dataset("polarization_gamma", data=np.asarray(row["polarization_gamma"], dtype=np.complex64))
        for key in ("polarization_metric", "polarization_signal_power", "polarization_noise_power"):
            if key in row and np.isfinite(row[key]):
                p.attrs[key] = float(row[key])
        if row.get("polarization_metric_kind"):
            p.attrs["polarization_metric_kind"] = str(row["polarization_metric_kind"])
        if int(row.get("polarization_metric_mask_pixels", 0)) > 0:
            p.attrs["polarization_metric_mask_pixels"] = int(row["polarization_metric_mask_pixels"])
        if "polarization_metric_mask" in row:
            p.create_dataset("polarization_metric_mask", data=np.asarray(row["polarization_metric_mask"], dtype=np.uint8), compression="gzip", compression_opts=1)
        if "polarization_reference_fit" in row:
            rg = p.create_group("polarization_reference_fit")
            for key, val in row["polarization_reference_fit"].items():
                rg.attrs[key] = val
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
    global DATA_DIR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--time-pad-s", type=float, default=0.2)
    p.add_argument("--time-step-s", type=float, default=0.01)
    p.add_argument(
        "--retained-head-times",
        action="store_true",
        help="Use retained head-echo observation times, but seed range/Doppler from the same trajectory prediction used in trail_echo_roi.py.",
    )
    p.add_argument(
        "--retained-time-extension-s",
        type=float,
        default=0.0,
        help="When using retained head times, replace each station's retained times with a regular grid expanded by this many seconds before and after that station's retained time span.",
    )
    p.add_argument("--time-rel-start-s", type=float, default=None)
    p.add_argument("--time-rel-stop-s", type=float, default=None)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--fft-filter-domain", choices=("db", "linear"), default="linear")
    p.add_argument("--inpaint-notched-columns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=16.0)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--dc-cancel-seconds", type=float, default=5.0)
    p.add_argument("--dc-cancel-start-second", type=float, default=0.0)
    p.add_argument("--sparse-voltage-subtract", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sparse-product-dir", type=Path, default=DEFAULT_SPARSE_PRODUCT_DIR)
    p.add_argument("--sparse-voltage-mode", choices=("full", "dc"), default="full")
    p.add_argument("--sparse-voltage-gain", type=float, default=1.0)
    p.add_argument("--polarization-combine", action="store_true")
    p.add_argument("--polarization-method", choices=("phase", "grid", "grid_image"), default="phase")
    p.add_argument("--polarization-image-metric", choices=("peak", "gaussian_1sigma"), default="peak")
    p.add_argument("--polarization-equalize-noise", action="store_true")
    p.add_argument("--polarization-equalize-block-samples", type=int, default=100000)
    p.add_argument("--polarization-grid-theta", type=int, default=16)
    p.add_argument("--polarization-grid-phase", type=int, default=32)
    p.add_argument("--polarization-phase-chunk", type=int, default=32)
    p.add_argument("--product-version", default="unspecified")
    p.add_argument("--product-version-description", default="")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    DATA_DIR = args.data_dir
    md2d.DATA_DIR = args.data_dir

    payload, manual_observations = load_station_observations()
    stations = list(args.stations)
    if args.retained_head_times:
        observations = {}
        common_times = []
        for station in stations:
            times = np.asarray([float(obs["unix_time"]) for obs in manual_observations[station]], dtype=float)
            if args.retained_time_extension_s > 0.0:
                start = float(np.min(times)) - args.retained_time_extension_s
                stop = float(np.max(times)) + args.retained_time_extension_s
                n = int(np.floor((stop - start) / args.time_step_s + 0.5)) + 1
                times = start + args.time_step_s * np.arange(n, dtype=float)
            observations[station] = predicted_observations(payload, [station], times)[station]
            common_times.extend(times.tolist())
        times = np.asarray(sorted(set(common_times)), dtype=float)
    else:
        times = common_time_grid(args.annotations, stations, args.time_pad_s, args.time_step_s)
        observations = predicted_observations(payload, stations, times)
    observations = restrict_observations(observations, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.attrs["description"] = "Predicted-grid head-echo range-Doppler matched-filter patches for local 2D FFT notch and Gaussian fitting"
        h.attrs["head_echo_product_version"] = str(args.product_version)
        h.attrs["head_echo_product_version_description"] = str(args.product_version_description)
        h.attrs["event_t0_unix"] = EVENT_T0
        h.attrs["time_pad_s"] = args.time_pad_s
        h.attrs["time_step_s"] = args.time_step_s
        h.attrs["retained_head_times"] = bool(args.retained_head_times)
        h.attrs["retained_time_extension_s"] = float(args.retained_time_extension_s)
        h.attrs["time_rel_start_s"] = np.nan if args.time_rel_start_s is None else float(args.time_rel_start_s)
        h.attrs["time_rel_stop_s"] = np.nan if args.time_rel_stop_s is None else float(args.time_rel_stop_s)
        h.attrs["n_coh"] = args.n_coh
        h.attrs["n_incoh"] = args.n_incoh
        h.attrs["step_codes"] = args.step_codes
        h.attrs["doppler_zero_pad_factor"] = args.doppler_zero_pad_factor
        h.attrs["range_half_width_km"] = args.range_half_width_km
        h.attrs["doppler_half_width_hz"] = args.doppler_half_width_hz
        h.attrs["fft_filter_domain"] = str(args.fft_filter_domain)
        h.attrs["inpaint_notched_columns"] = bool(args.inpaint_notched_columns)
        h.attrs["smooth_range_pixels"] = float(args.smooth_range_pixels)
        h.attrs["smooth_frequency_pixels"] = float(args.smooth_frequency_pixels)
        h.attrs["search_range_half_width_km"] = float(args.search_range_half_width_km)
        h.attrs["search_doppler_half_width_hz"] = float(args.search_doppler_half_width_hz)
        h.attrs["fit_range_half_width_km"] = float(args.fit_range_half_width_km)
        h.attrs["fit_doppler_half_width_hz"] = float(args.fit_doppler_half_width_hz)
        h.attrs["dc_cancel"] = bool(args.dc_cancel)
        h.attrs["dc_subtract"] = "complex_mean_per_channel_read"
        h.attrs["dc_cancel_seconds"] = float(args.dc_cancel_seconds)
        h.attrs["dc_cancel_start_second"] = float(args.dc_cancel_start_second)
        h.attrs["sparse_voltage_subtract"] = bool(args.sparse_voltage_subtract)
        h.attrs["sparse_product_dir"] = str(args.sparse_product_dir)
        h.attrs["sparse_voltage_mode"] = str(args.sparse_voltage_mode)
        h.attrs["sparse_voltage_gain"] = float(args.sparse_voltage_gain)
        h.attrs["polarization_combine"] = bool(args.polarization_combine)
        h.attrs["polarization_method"] = str(args.polarization_method)
        h.attrs["polarization_image_metric"] = str(args.polarization_image_metric)
        h.attrs["polarization_equalize_noise"] = bool(args.polarization_equalize_noise)
        h.attrs["polarization_equalize_block_samples"] = int(args.polarization_equalize_block_samples)
        h.attrs["polarization_grid_theta"] = int(args.polarization_grid_theta)
        h.attrs["polarization_grid_phase"] = int(args.polarization_grid_phase)
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
