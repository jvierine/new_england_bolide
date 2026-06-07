#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

from range_doppler_mf_bolide import bpsk, interpolate_rect, read_config, read_channel_files, rf_file_for_time
from station_config import STATION_ORDER, short_station_name
from trail_echo_deconv import (
    DEFAULT_DATA_DIR,
    DEFAULT_FIG_DIR,
    DEFAULT_TRAJECTORY_JSON,
    crop_for_station,
    resolve_path,
    robust_db,
    trajectory_range_limits,
    unix_to_mpl,
)

try:
    from trail_echo_avx import range_doppler_peak_scores_avx
except Exception:
    range_doppler_peak_scores_avx = None


DEFAULT_PRODUCT_DIR = Path("products/trail_echo_sparse_deconv")
RANGE_DOPPLER_ENGINE = "auto"


def precompute_shifted_codes(codes: np.ndarray, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cache delayed TX codes as complex64 for repeated range-Doppler scans.

    shifted_codes has shape (n_tx, n_range, code_len).  code_matrix has columns
    ordered as (range0_tx0, range0_tx1, ..., range1_tx0, ...), matching the
    coefficient layout used by the least-squares solver.
    """
    n_tx, code_len = codes.shape
    sample_idx = np.arange(code_len, dtype=np.int64)
    shifted_codes = np.empty((n_tx, len(ranges), code_len), dtype=np.complex64)
    for ri, raw_range in enumerate(ranges):
        shifted_codes[:, ri, :] = codes[:, (sample_idx - int(raw_range)) % code_len]
    code_matrix = np.transpose(shifted_codes, (2, 1, 0)).reshape(code_len, len(ranges) * n_tx)
    return shifted_codes, np.ascontiguousarray(code_matrix, dtype=np.complex64)


def matched_filter_range_scores(residual: np.ndarray, shifted_codes: np.ndarray) -> np.ndarray:
    """Score ranges using single-TX matched filters over all channels/codes."""
    n_tx, n_range, code_len = shifted_codes.shape
    y = residual.reshape(code_len, -1)
    scores = np.zeros(n_range, dtype=np.float64)
    for si in range(n_range):
        shifted = shifted_codes[:, si, :]
        mf = shifted.conj() @ y
        scores[si] = float(np.sum(np.abs(mf) ** 2)) / float(code_len**2 * n_tx * y.shape[1])
    return scores


def range_doppler_peak_scores(
    y_block: np.ndarray,
    shifted_codes: np.ndarray,
    window: np.ndarray,
    window_norm: float,
    doppler_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return max-over-Doppler matched-filter power and Doppler for each range."""
    use_avx = RANGE_DOPPLER_ENGINE in {"auto", "avx"}
    if use_avx and range_doppler_peak_scores_avx is not None:
        try:
            power, peak_bin = range_doppler_peak_scores_avx(y_block, shifted_codes, window, window_norm)
            return power.astype(np.float64), doppler_hz[peak_bin].astype(np.float32)
        except Exception:
            if RANGE_DOPPLER_ENGINE == "avx":
                raise
    n_codes, n_ch, code_len = y_block.shape
    n_tx, n_range, _ = shifted_codes.shape
    power = np.zeros(n_range, dtype=np.float64)
    doppler = np.zeros(n_range, dtype=np.float32)
    for si in range(n_range):
        p = np.zeros(len(doppler_hz), dtype=np.float64)
        shifted = shifted_codes[:, si, :]
        for chi in range(n_ch):
            amp = y_block[:, chi, :] @ shifted.conj().T / float(code_len)
            spec = np.fft.fftshift(np.fft.fft(amp * window[:, None], axis=0), axes=0)
            p += np.sum(np.abs(spec) ** 2, axis=1) / float(window_norm * n_ch * n_tx)
        peak_i = int(np.argmax(p))
        power[si] = float(p[peak_i])
        doppler[si] = float(doppler_hz[peak_i])
    return power, doppler


def support_from_scores(
    scores: np.ndarray,
    ranges_all: np.ndarray,
    threshold_db: float,
    neighbor_gates: int,
) -> set[int]:
    finite = np.isfinite(scores)
    if not np.any(finite):
        return set()
    floor = float(np.nanmedian(scores[finite]))
    keep = scores >= floor * 10.0 ** (threshold_db / 10.0)
    support: set[int] = set()
    for ri in ranges_all[keep]:
        support = support_with_neighbors(support, int(ri), ranges_all, neighbor_gates)
    return support


def load_candidate_ranges(
    station: str,
    candidate_dir: Path,
    t_block: np.ndarray,
    ranges_all: np.ndarray,
    snr_threshold_db: float,
    neighbor_gates: int,
    range_gate_km: float,
) -> list[set[int]] | None:
    path = candidate_dir / f"trail_deconv_{station}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as h:
        cand_t = h["time_unix"][()]
        cand_ranges = np.rint(h["range_km"][()] / range_gate_km).astype(np.int32)
        cand_snr = robust_db(h["peak_power"][()])
    out = []
    valid = set(int(ri) for ri in ranges_all)
    for t in t_block:
        ti = int(np.argmin(np.abs(cand_t - float(t))))
        seed_ranges = cand_ranges[cand_snr[ti] >= snr_threshold_db]
        support: set[int] = set()
        for ri in seed_ranges:
            for rj in range(int(ri) - neighbor_gates, int(ri) + neighbor_gates + 1):
                if rj in valid:
                    support.add(rj)
        out.append(support)
    return out


def neighbor_gates_from_km(distance_km: float, range_resolution_m: float) -> int:
    gate_km = range_resolution_m / 1000.0
    return max(0, int(np.ceil(float(distance_km) / gate_km)))


def support_with_neighbors(support: set[int], center: int, ranges_all: np.ndarray, half_width: int) -> set[int]:
    lo = int(ranges_all[0])
    hi = int(ranges_all[-1])
    for ri in range(center - half_width, center + half_width + 1):
        if lo <= ri <= hi:
            support.add(int(ri))
    return support


def support_indices(support: set[int], ranges_all: np.ndarray) -> np.ndarray:
    if not support:
        return np.array([], dtype=np.int64)
    lo = int(ranges_all[0])
    hi = int(ranges_all[-1])
    return np.asarray([ri - lo for ri in support if lo <= ri <= hi], dtype=np.int64)


def solve_on_support(
    y_block: np.ndarray,
    code_matrix: np.ndarray,
    ranges_all: np.ndarray,
    n_tx: int,
    support: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, complex, np.ndarray, np.ndarray]:
    """Solve TX amplitudes and a constant voltage offset for a full block.

    Parameters
    ----------
    y_block:
        Complex samples with shape (n_codes, n_channels, code_len).
    code_matrix:
        Precomputed delayed-code matrix with shape (code_len, n_tx * n_ranges).
    support:
        Active raw range indices.

    Returns
    -------
    coeff:
        Shape (n_codes, n_channels, n_tx, n_support).
    dc_voltage:
        One constant voltage offset shared by the full block.
    residual:
        Same shape as y_block.
    model:
        Same shape as y_block.
    """
    n_codes, n_ch, code_len = y_block.shape
    support_i = support.astype(np.int64) - int(ranges_all[0])
    col_i = (support_i[:, None] * n_tx + np.arange(n_tx, dtype=np.int64)[None, :]).ravel()
    a = code_matrix[:, col_i]
    y = np.moveaxis(y_block, -1, 0).reshape(code_len, n_codes * n_ch)

    if a.shape[1] == 0:
        dc_voltage = complex(np.mean(y))
        x = np.empty((0, y.shape[1]), dtype=np.complex64)
    else:
        ones = np.ones((code_len, 1), dtype=np.complex64)
        x0, *_ = np.linalg.lstsq(a, y, rcond=rcond)
        x1, *_ = np.linalg.lstsq(a, ones, rcond=rcond)
        residual_y = y - a @ x0
        residual_one = ones - a @ x1
        denom = y.shape[1] * np.vdot(residual_one[:, 0], residual_one[:, 0])
        if abs(denom) <= 1e-20:
            dc_voltage = 0.0j
        else:
            dc_voltage = complex(np.vdot(residual_one[:, 0], np.sum(residual_y, axis=1)) / denom)
        x, *_ = np.linalg.lstsq(a, y - dc_voltage * ones, rcond=rcond)

    model = a @ x
    model += dc_voltage
    residual = (y - model).reshape(code_len, n_codes, n_ch)
    residual = np.moveaxis(residual, 0, -1).astype(np.complex64)
    model = model.reshape(code_len, n_codes, n_ch)
    model = np.moveaxis(model, 0, -1).astype(np.complex64)
    coeff = x.reshape(len(support), n_tx, n_codes, n_ch)
    coeff = np.moveaxis(coeff, [0, 1, 2, 3], [3, 2, 0, 1]).astype(np.complex64)
    return coeff, dc_voltage, residual, model


def select_sparse_support(
    y_block: np.ndarray,
    shifted_codes: np.ndarray,
    code_matrix: np.ndarray,
    n_tx: int,
    ranges_all: np.ndarray,
    threshold_snr: float,
    neighbor_gates: int,
    max_active_ranges: int,
    max_iter: int,
    rcond: float,
    residual_peak_fraction: float,
    candidate_support: set[int] | None,
    window: np.ndarray,
    window_norm: float,
    doppler_hz: np.ndarray,
    use_doppler_scores: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    support: set[int] = set()
    history = []
    empty_support = np.asarray([], dtype=np.int32)
    coeff, dc_voltage, residual, model = solve_on_support(y_block, code_matrix, ranges_all, n_tx, empty_support, rcond)
    initial_peak = None

    for _ in range(max_iter):
        if use_doppler_scores:
            scores, _ = range_doppler_peak_scores(residual, shifted_codes, window, window_norm, doppler_hz)
        else:
            scores = matched_filter_range_scores(residual, shifted_codes)
        if candidate_support:
            candidate_i = support_indices(candidate_support, ranges_all)
            mask = np.ones(len(scores), dtype=bool)
            mask[candidate_i] = False
            scores[mask] = -np.inf
        active_i = support_indices(support, ranges_all)
        if len(active_i):
            scores[active_i] = -np.inf
        if not np.any(np.isfinite(scores)):
            break
        floor = float(np.median(scores))
        finite_scores = scores[np.isfinite(scores)]
        floor = float(np.median(finite_scores))
        sigma = 1.4826 * float(np.median(np.abs(finite_scores - floor)))
        scale = max(floor + sigma, 1e-20)
        best_i = int(np.argmax(scores))
        best_range = int(ranges_all[best_i])
        if initial_peak is None:
            initial_peak = float(scores[best_i])
        best_snr = float(scores[best_i] / scale)
        history.append((best_range, best_snr, len(support)))
        weak_relative = initial_peak is not None and float(scores[best_i]) < residual_peak_fraction * initial_peak
        if best_snr < threshold_snr and weak_relative and support:
            break
        before = len(support)
        support = support_with_neighbors(support, best_range, ranges_all, neighbor_gates)
        if len(support) > max_active_ranges:
            if use_doppler_scores:
                full_scores, _ = range_doppler_peak_scores(y_block, shifted_codes, window, window_norm, doppler_hz)
            else:
                full_scores = matched_filter_range_scores(y_block, shifted_codes)
            ranked = sorted(support, key=lambda ri: full_scores[int(ri - ranges_all[0])], reverse=True)
            support = set(sorted(ranked[:max_active_ranges]))
        support_arr = np.asarray(sorted(support), dtype=np.int32)
        coeff, dc_voltage, residual, model = solve_on_support(y_block, code_matrix, ranges_all, n_tx, support_arr, rcond)
        if len(support) >= max_active_ranges:
            break

    support_arr = np.asarray(sorted(support), dtype=np.int32)
    coeff, dc_voltage, residual, model = solve_on_support(y_block, code_matrix, ranges_all, n_tx, support_arr, rcond)
    meta = {
        "iterations": int(len(history)),
        "history_range_idx": np.asarray([h[0] for h in history], dtype=np.int32),
        "history_snr": np.asarray([h[1] for h in history], dtype=np.float32),
        "residual_fraction": float(np.sum(np.abs(residual) ** 2) / max(np.sum(np.abs(y_block) ** 2), 1e-20)),
    }
    return support_arr, coeff, dc_voltage, residual, meta


def process_station(args: argparse.Namespace, station: str) -> Path:
    rank = int(getattr(args, "mpi_rank", 0))
    size = int(getattr(args, "mpi_size", 1))
    cfg = read_config(args.data_dir, station)
    code_len_i = cfg["code_len"] * args.range_interp
    sample_rate_i = None
    range_resolution_m_i = args.range_resolution_m / float(args.range_interp)
    rmin = cfg["rmin_idx"] * args.range_interp if args.rmin_idx is None else args.rmin_idx
    rmax = cfg["rmax_idx"] * args.range_interp if args.rmax_idx is None else args.rmax_idx
    if args.max_range_km is not None:
        rmax = min(rmax, int(round(args.max_range_km / (range_resolution_m_i / 1000.0))))
    ranges_all = np.arange(rmin, rmax + 1, dtype=np.int32)
    range_km = ranges_all.astype(np.float32) * (range_resolution_m_i / 1000.0)

    station_dir = Path(args.data_dir) / station
    channel_data = []
    sample_rate = None
    abs_index0 = None
    for ch in args.channels:
        files = rf_file_for_time(station_dir, ch, args.iso_hour)
        z, sr, idx0 = read_channel_files(files, start_second=args.start_second, seconds=args.seconds)
        z = interpolate_rect(z, args.range_interp)
        sample_rate_i = sr * args.range_interp if sample_rate_i is None else sample_rate_i
        n_code_samples = (len(z) // code_len_i) * code_len_i
        z = z[:n_code_samples].reshape(-1, code_len_i)
        channel_data.append(z)
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
        if sample_rate != sr or abs_index0 != idx0:
            raise ValueError(f"{station}: channel {ch} time base mismatch")

    n_codes_total = min(z.shape[0] for z in channel_data)
    if args.max_codes is not None:
        n_codes_total = min(n_codes_total, args.max_codes)
    n_blocks = n_codes_total // args.coherent_codes
    n_codes_used = n_blocks * args.coherent_codes
    if n_blocks <= 0:
        raise ValueError(f"{station}: not enough data")
    channel_data = [z[:n_codes_used] for z in channel_data]
    y_all = np.stack(channel_data, axis=1)

    codes = []
    for seed in cfg["seeds"]:
        code = interpolate_rect(bpsk(int(seed), cfg["code_len"]), args.range_interp)
        code = code / np.max(np.abs(code))
        codes.append(code.astype(np.complex64))
    codes = np.asarray(codes, dtype=np.complex64)
    shifted_codes, code_matrix = precompute_shifted_codes(codes, ranges_all)
    dt_code = code_len_i / float(sample_rate_i)
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(args.coherent_codes, d=dt_code)).astype(np.float32)
    window = np.hanning(args.coherent_codes).astype(np.float32)
    window_norm = float(np.sum(window**2))

    n_tx = len(cfg["seeds"])
    n_ch = len(args.channels)
    n_range = len(ranges_all)
    peak_power = np.zeros((n_blocks, n_range), dtype=np.float32)
    peak_doppler = np.zeros((n_blocks, n_range), dtype=np.float32)
    candidate_peak_power = np.zeros((n_blocks, n_range), dtype=np.float32)
    candidate_peak_doppler = np.zeros((n_blocks, n_range), dtype=np.float32)
    residual_power = np.zeros((n_blocks, n_range), dtype=np.float32)
    support_mask = np.zeros((n_blocks, n_range), dtype=np.uint8)
    support_count = np.zeros(n_blocks, dtype=np.int16)
    residual_fraction = np.zeros(n_blocks, dtype=np.float32)
    support_range_idx = np.full((n_blocks, args.max_active_ranges), -1, dtype=np.int32)
    support_voltage = np.zeros((n_blocks, args.coherent_codes, n_ch, n_tx, args.max_active_ranges), dtype=np.complex64)
    dc_voltage = np.zeros(n_blocks, dtype=np.complex64)

    t_block = (abs_index0 + (np.arange(n_blocks) * args.coherent_codes + 0.5 * args.coherent_codes) * cfg["code_len"]) / float(sample_rate)
    time_resolution_s = args.coherent_codes * cfg["code_len"] / float(sample_rate)
    candidate_by_block = load_candidate_ranges(
        station,
        Path(args.candidate_product_dir),
        t_block,
        ranges_all,
        args.candidate_snr_db,
        args.candidate_neighbor_gates,
        range_resolution_m_i / 1000.0,
    )
    if args.candidate_source != "deconv":
        candidate_by_block = None
    t0 = time.time()
    local_done = np.zeros(n_blocks, dtype=bool)
    block_iter = range(rank, n_blocks, size)
    for bi in block_iter:
        c0 = bi * args.coherent_codes
        c1 = c0 + args.coherent_codes
        y_block = y_all[c0:c1]
        block_candidate_support = candidate_by_block[bi] if candidate_by_block is not None else None
        if args.candidate_source == "rdmf":
            rd_power, rd_doppler = range_doppler_peak_scores(y_block, shifted_codes, window, window_norm, doppler_hz)
            candidate_peak_power[bi] = rd_power.astype(np.float32)
            candidate_peak_doppler[bi] = rd_doppler.astype(np.float32)
            block_candidate_support = support_from_scores(
                rd_power,
                ranges_all,
                args.candidate_rd_snr_db,
                args.candidate_neighbor_gates,
            )
        elif args.candidate_source == "none":
            block_candidate_support = None
        support, coeff, dc_coeff, residual, meta = select_sparse_support(
            y_block,
            shifted_codes,
            code_matrix,
            n_tx,
            ranges_all,
            threshold_snr=args.support_threshold_snr,
            neighbor_gates=args.neighbor_gates,
            max_active_ranges=args.max_active_ranges,
            max_iter=args.max_iter,
            rcond=args.lstsq_rcond,
            residual_peak_fraction=args.residual_peak_fraction,
            candidate_support=block_candidate_support,
            window=window,
            window_norm=window_norm,
            doppler_hz=doppler_hz,
            use_doppler_scores=args.support_score == "rdmf",
        )
        n_support = min(len(support), args.max_active_ranges)
        support_count[bi] = n_support
        local_done[bi] = True
        dc_voltage[bi] = dc_coeff
        residual_fraction[bi] = meta["residual_fraction"]
        residual_power[bi] = matched_filter_range_scores(residual, shifted_codes).astype(np.float32)
        if n_support:
            support_range_idx[bi, :n_support] = support[:n_support]
            support_voltage[bi, :, :, :, :n_support] = coeff[:, :, :, :n_support]
            support_mask[bi, np.searchsorted(ranges_all, support[:n_support])] = 1
            for si, ri in enumerate(support[:n_support]):
                dense_i = int(ri - ranges_all[0])
                p = np.zeros(len(doppler_hz), dtype=np.float32)
                for chi in range(n_ch):
                    for txi in range(n_tx):
                        spec = np.fft.fftshift(np.fft.fft(coeff[:, chi, txi, si] * window))
                        p += (np.abs(spec) ** 2 / (window_norm * n_ch * n_tx)).astype(np.float32)
                peak_i = int(np.argmax(p))
                peak_power[bi, dense_i] = p[peak_i]
                peak_doppler[bi, dense_i] = doppler_hz[peak_i]
        if bi == 0 or (bi + 1) == n_blocks or (args.progress_every and (bi + 1) % args.progress_every == 0):
            print(
                f"{station}: rank {rank}/{size} block {bi + 1}/{n_blocks} support={n_support} "
                f"resid={residual_fraction[bi]:.3f} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    comm = getattr(args, "mpi_comm", None)
    if comm is not None and size > 1:
        gathered = comm.gather(
            {
                "done": local_done,
                "peak_power": peak_power,
                "peak_doppler": peak_doppler,
                "candidate_peak_power": candidate_peak_power,
                "candidate_peak_doppler": candidate_peak_doppler,
                "residual_power": residual_power,
                "support_mask": support_mask,
                "support_count": support_count,
                "residual_fraction": residual_fraction,
                "support_range_idx": support_range_idx,
                "support_voltage": support_voltage,
                "dc_voltage": dc_voltage,
            },
            root=0,
        )
        if rank != 0:
            return Path(args.product_dir) / f"trail_sparse_deconv_{station}.h5"
        peak_power.fill(0.0)
        peak_doppler.fill(0.0)
        candidate_peak_power.fill(0.0)
        candidate_peak_doppler.fill(0.0)
        residual_power.fill(0.0)
        support_mask.fill(0)
        support_count.fill(0)
        residual_fraction.fill(0.0)
        support_range_idx.fill(-1)
        support_voltage.fill(0.0)
        dc_voltage.fill(0.0)
        for part in gathered:
            rows = part["done"]
            peak_power[rows] = part["peak_power"][rows]
            peak_doppler[rows] = part["peak_doppler"][rows]
            candidate_peak_power[rows] = part["candidate_peak_power"][rows]
            candidate_peak_doppler[rows] = part["candidate_peak_doppler"][rows]
            residual_power[rows] = part["residual_power"][rows]
            support_mask[rows] = part["support_mask"][rows]
            support_count[rows] = part["support_count"][rows]
            residual_fraction[rows] = part["residual_fraction"][rows]
            support_range_idx[rows] = part["support_range_idx"][rows]
            support_voltage[rows] = part["support_voltage"][rows]
            dc_voltage[rows] = part["dc_voltage"][rows]

    out_path = Path(args.product_dir) / f"trail_sparse_deconv_{station}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as h:
        h["peak_power"] = peak_power
        h["peak_doppler_hz"] = peak_doppler
        h["candidate_peak_power"] = candidate_peak_power
        h["candidate_peak_doppler_hz"] = candidate_peak_doppler
        h["residual_power"] = residual_power
        h["support_mask"] = support_mask
        h["support_count"] = support_count
        h["residual_fraction"] = residual_fraction
        h["support_range_idx"] = support_range_idx
        h["support_voltage"] = support_voltage
        h["dc_voltage"] = dc_voltage
        h["time_unix"] = t_block.astype(np.float64)
        h["range_km"] = range_km
        h["range_idx"] = ranges_all
        h["doppler_hz"] = doppler_hz
        h["channels"] = np.asarray(args.channels, dtype=np.int32)
        h["seeds"] = np.asarray(cfg["seeds"], dtype=np.int64)
        h["sample_rate"] = float(sample_rate)
        h["interpolated_sample_rate"] = float(sample_rate_i)
        h["t0_sample"] = int(abs_index0)
        h["code_len"] = int(code_len_i)
        h["raw_code_len"] = int(cfg["code_len"])
        h["range_interp"] = int(args.range_interp)
        h["coherent_codes"] = int(args.coherent_codes)
        h["station"] = np.bytes_(station)
        h.attrs["method"] = "Sparse active-range pursuit followed by simultaneous all-transmit-code plus constant-DC least squares on the active support."
        h.attrs["support_selection"] = "Iterative single-TX matched-filter residual range picking with neighboring range gates added after each selected peak."
        h.attrs["time_resolution_s"] = float(time_resolution_s)
        h.attrs["range_resolution_km"] = float(range_resolution_m_i / 1000.0)
        h.attrs["candidate_product_dir"] = str(args.candidate_product_dir)
        h.attrs["candidate_source"] = str(args.candidate_source)
        h.attrs["candidate_snr_db"] = float(args.candidate_snr_db)
        h.attrs["candidate_rd_snr_db"] = float(args.candidate_rd_snr_db)
        h.attrs["support_score"] = str(args.support_score)
        h.attrs["range_doppler_engine"] = str(args.range_doppler_engine)
        h.attrs["support_threshold_snr"] = float(args.support_threshold_snr)
        h.attrs["neighbor_gates"] = int(args.neighbor_gates)
        h.attrs["support_neighbor_km"] = float(args.support_neighbor_km)
        h.attrs["candidate_neighbor_gates"] = int(args.candidate_neighbor_gates)
        h.attrs["candidate_neighbor_km"] = float(args.candidate_neighbor_km)
        h.attrs["max_active_ranges"] = int(args.max_active_ranges)
    tmp.replace(out_path)
    print(f"{station}: wrote {out_path}", flush=True)
    return out_path


def load_product(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        peak_power = h["peak_power"][()]
        snr_db = robust_db(np.where(peak_power > 0, peak_power, np.nanmedian(peak_power[peak_power > 0]) if np.any(peak_power > 0) else 1e-12))
        snr_db = np.where(h["support_mask"][()] > 0, snr_db, np.nan)
        residual_power = h["residual_power"][()] if "residual_power" in h else np.full_like(peak_power, np.nan)
        doppler_width_hz = h["doppler_width_hz"][()] if "doppler_width_hz" in h else None
        if doppler_width_hz is None and "support_voltage" in h:
            doppler_width_hz = compute_doppler_width_grid(h)
        candidate_peak_power = h["candidate_peak_power"][()] if "candidate_peak_power" in h else np.full_like(peak_power, np.nan)
        return {
            "station": h["station"][()].decode(),
            "time_unix": h["time_unix"][()],
            "time_resolution_s": float(h.attrs.get("time_resolution_s", h["coherent_codes"][()] * h["code_len"][()] / h["sample_rate"][()])),
            "range_km": h["range_km"][()],
            "snr_db": snr_db,
            "candidate_db": robust_db(candidate_peak_power),
            "residual_db": robust_db(residual_power),
            "peak_doppler_hz": h["peak_doppler_hz"][()],
            "doppler_width_hz": doppler_width_hz,
            "support_mask": h["support_mask"][()],
        }


def compute_doppler_width_grid(h: h5py.File) -> np.ndarray:
    support_voltage = h["support_voltage"][()]
    support_range_idx = h["support_range_idx"][()]
    range_idx = h["range_idx"][()]
    doppler_hz = h["doppler_hz"][()]
    n_blocks, _n_codes, _n_ch, _n_tx, max_support = support_voltage.shape
    out = np.full((n_blocks, len(range_idx)), np.nan, dtype=np.float32)
    idx0 = int(range_idx[0])
    for bi in range(n_blocks):
        for si in range(max_support):
            ri = int(support_range_idx[bi, si])
            if ri < idx0:
                continue
            v = support_voltage[bi, :, :, :, si]
            if not np.any(v):
                continue
            spec = np.fft.fftshift(np.fft.fft(v, axis=0), axes=0)
            power = np.mean(np.abs(spec) ** 2, axis=(1, 2))
            floor = np.nanmedian(power)
            w = np.maximum(power - floor, 0.0)
            if not np.any(w > 0):
                w = power
            norm = float(np.sum(w))
            if norm <= 0:
                continue
            mu = float(np.sum(doppler_hz * w) / norm)
            width = float(np.sqrt(np.sum(((doppler_hz - mu) ** 2) * w) / norm))
            out[bi, ri - idx0] = width
    return out


def edges_from_centers(x: np.ndarray, step: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        dx = 1.0 if step is None else float(step)
        return np.array([x[0] - 0.5 * dx, x[0] + 0.5 * dx])
    if step is None:
        dx = np.diff(x)
        edges = np.empty(len(x) + 1, dtype=float)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * dx[0]
        edges[-1] = x[-1] + 0.5 * dx[-1]
        return edges
    dx = float(step)
    return np.r_[x - 0.5 * dx, x[-1] + 0.5 * dx]


def plot_products(
    products: list[Path],
    fig_dir: Path,
    data_dir: Path,
    trajectory_json: Path,
    altitude_min_km: float,
    altitude_max_km: float,
    range_pad_km: float,
    doppler_color_abs_hz: float,
    plot_range_mode: str,
    plot_residuals: bool,
    plot_hsv: bool,
    hsv_doppler_abs_hz: float,
    hsv_value_max_db: float,
    hsv_width_full_saturation_hz: float,
    hsv_width_zero_saturation_hz: float,
) -> tuple[Path, ...]:
    data = [load_product(p) for p in products]
    if plot_range_mode == "trajectory":
        limits = trajectory_range_limits(data, data_dir, trajectory_json, altitude_min_km, altitude_max_km, range_pad_km)
    else:
        limits = {}
    vals = []
    for d in data:
        keep_r, _ = crop_for_station(d, limits)
        vals.append(d["snr_db"][:, keep_r].ravel())
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    vmin = max(-4.0, float(np.nanpercentile(vals, 5.0))) if len(vals) else -4.0
    vmax = min(28.0, float(np.nanpercentile(vals, 99.5))) if len(vals) else 12.0

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    if len(data) == 1:
        axes = [axes]
    mesh = None
    for ax, d in zip(axes, data, strict=True):
        keep_r, ylim = crop_for_station(d, limits)
        mesh = ax.pcolormesh(
            unix_to_mpl(edges_from_centers(d["time_unix"], d["time_resolution_s"])),
            edges_from_centers(d["range_km"][keep_r]),
            d["snr_db"][:, keep_r].T,
            shading="flat",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
        ax.set_ylim(ylim)
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes, label="Sparse support peak power (dB)", shrink=0.92)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rti_out = fig_dir / "trail_sparse_deconv_range_time_intensity.png"
    fig.savefig(rti_out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    residual_vals = []
    residual_out = None
    if plot_residuals:
        for d in data:
            keep_r, _ = crop_for_station(d, limits)
            residual_vals.append(d["residual_db"][:, keep_r].ravel())
        residual_vals = np.concatenate(residual_vals)
        residual_vals = residual_vals[np.isfinite(residual_vals)]
        rvmin = max(-8.0, float(np.nanpercentile(residual_vals, 2.0))) if len(residual_vals) else -8.0
        rvmax = min(20.0, float(np.nanpercentile(residual_vals, 99.0))) if len(residual_vals) else 10.0
        fig, axes = plt.subplots(len(data), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        if len(data) == 1:
            axes = [axes]
        mesh = None
        for ax, d in zip(axes, data, strict=True):
            keep_r, ylim = crop_for_station(d, limits)
            mesh = ax.pcolormesh(
                unix_to_mpl(edges_from_centers(d["time_unix"], d["time_resolution_s"])),
                edges_from_centers(d["range_km"][keep_r]),
                d["residual_db"][:, keep_r].T,
                shading="flat",
                cmap="magma",
                vmin=rvmin,
                vmax=rvmax,
            )
            ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
            ax.set_ylim(ylim)
            ax.grid(True, color="0.9", linewidth=0.4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[-1].set_xlabel("UTC on 2026-05-30")
        fig.colorbar(mesh, ax=axes, label="Post-fit residual matched-filter power (dB)", shrink=0.92)
        residual_out = fig_dir / "trail_sparse_deconv_residual_range_time.png"
        fig.savefig(residual_out, dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    if len(data) == 1:
        axes = [axes]
    mesh = None
    for ax, d in zip(axes, data, strict=True):
        keep_r, ylim = crop_for_station(d, limits)
        dop = np.where(d["support_mask"][:, keep_r] > 0, d["peak_doppler_hz"][:, keep_r], np.nan)
        mesh = ax.pcolormesh(
            unix_to_mpl(edges_from_centers(d["time_unix"], d["time_resolution_s"])),
            edges_from_centers(d["range_km"][keep_r]),
            dop.T,
            shading="flat",
            cmap="coolwarm",
            vmin=-doppler_color_abs_hz,
            vmax=doppler_color_abs_hz,
        )
        ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
        ax.set_ylim(ylim)
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes, label="Sparse support peak Doppler shift (Hz)", shrink=0.92)
    dop_out = fig_dir / "trail_sparse_deconv_peak_doppler_time.png"
    fig.savefig(dop_out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    hsv_out = None
    if plot_hsv:
        fig, axes = plt.subplots(len(data), 1, figsize=(13.5, 10), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.80, top=0.98, bottom=0.08, hspace=0.10)
        if len(data) == 1:
            axes = [axes]
        for ax, d in zip(axes, data, strict=True):
            keep_r, ylim = crop_for_station(d, limits)
            rgb = hsv_doppler_image(
                d["snr_db"][:, keep_r],
                d["peak_doppler_hz"][:, keep_r],
                d["doppler_width_hz"][:, keep_r],
                doppler_abs_hz=hsv_doppler_abs_hz,
                value_max_db=hsv_value_max_db,
                width_full_saturation_hz=hsv_width_full_saturation_hz,
                width_zero_saturation_hz=hsv_width_zero_saturation_hz,
            )
            ax.imshow(
                np.transpose(rgb, (1, 0, 2)),
                origin="lower",
                aspect="auto",
                extent=[
                    unix_to_mpl(edges_from_centers(d["time_unix"], d["time_resolution_s"]))[0],
                    unix_to_mpl(edges_from_centers(d["time_unix"], d["time_resolution_s"]))[-1],
                    edges_from_centers(d["range_km"][keep_r])[0],
                    edges_from_centers(d["range_km"][keep_r])[-1],
                ],
                interpolation="nearest",
            )
            ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
            ax.set_ylim(ylim)
            ax.grid(True, color="0.85", linewidth=0.35, alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[-1].set_xlabel("UTC on 2026-05-30")
        add_hsv_legend(fig, hsv_doppler_abs_hz, hsv_value_max_db, hsv_width_full_saturation_hz, hsv_width_zero_saturation_hz)
        hsv_out = fig_dir / "trail_sparse_deconv_hsv_doppler_intensity.png"
        fig.savefig(hsv_out, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return tuple(p for p in (rti_out, residual_out, dop_out, hsv_out) if p is not None)


def hsv_doppler_image(
    intensity_db: np.ndarray,
    doppler_hz: np.ndarray,
    width_hz: np.ndarray,
    doppler_abs_hz: float,
    value_max_db: float,
    width_full_saturation_hz: float,
    width_zero_saturation_hz: float,
) -> np.ndarray:
    intensity_db = np.where(np.isfinite(intensity_db), intensity_db, 0.0)
    value = np.clip(intensity_db, 0.0, value_max_db) / float(value_max_db)
    # Restrict Doppler hue to a blue-magenta-red ramp: redshift is red, blueshift is blue.
    doppler_scaled = (np.clip(doppler_hz, -doppler_abs_hz, doppler_abs_hz) + doppler_abs_hz) / (2.0 * doppler_abs_hz)
    hue = (2.0 / 3.0) + doppler_scaled / 3.0
    hue = np.nan_to_num(hue, nan=0.5)
    width = np.nan_to_num(width_hz, nan=width_zero_saturation_hz, posinf=width_zero_saturation_hz)
    saturation = 1.0 - (width - width_full_saturation_hz) / max(width_zero_saturation_hz - width_full_saturation_hz, 1e-6)
    saturation = np.clip(saturation, 0.0, 1.0)
    saturation = np.where(value > 0.0, saturation, 0.0)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv_to_rgb(hsv)


def add_hsv_legend(
    fig: plt.Figure,
    doppler_abs_hz: float,
    value_max_db: float,
    width_full_saturation_hz: float,
    width_zero_saturation_hz: float,
) -> None:
    ax = fig.add_axes([0.84, 0.17, 0.055, 0.66])
    d = np.linspace(-doppler_abs_hz, doppler_abs_hz, 160)
    v = np.linspace(0.0, value_max_db, 220)
    dd, vv = np.meshgrid(d, v)
    ww = np.full_like(dd, width_full_saturation_hz)
    rgb = hsv_doppler_image(vv, dd, ww, doppler_abs_hz, value_max_db, width_full_saturation_hz, width_zero_saturation_hz)
    ax.imshow(rgb, origin="lower", aspect="auto", extent=[-doppler_abs_hz, doppler_abs_hz, 0.0, value_max_db])
    ax.set_xlabel("Hz", fontsize=9)
    ax.set_ylabel("dB", fontsize=9)
    ax.set_title("Hue/value", fontsize=9)
    ax.tick_params(labelsize=8)
    ax2 = fig.add_axes([0.93, 0.17, 0.025, 0.66])
    widths = np.linspace(width_zero_saturation_hz, width_full_saturation_hz, 220)
    hsv = np.zeros((220, 1, 3), dtype=float)
    hsv[:, :, 0] = 0.0
    hsv[:, :, 1] = np.linspace(0.0, 1.0, 220)[:, None]
    hsv[:, :, 2] = 1.0
    ax2.imshow(hsv_to_rgb(hsv), origin="lower", aspect="auto", extent=[0.0, 1.0, width_zero_saturation_hz, width_full_saturation_hz])
    ax2.set_xticks([])
    ax2.set_ylabel("width Hz", fontsize=9)
    ax2.tick_params(labelsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description="Sparse simultaneous all-TX trail deconvolution prototype.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--stations", nargs="+", default=STATION_ORDER)
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--start-second", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--coherent-codes", type=int, default=50)
    p.add_argument("--range-interp", type=int, default=1)
    p.add_argument("--range-resolution-m", type=float, default=3000.0)
    p.add_argument("--rmin-idx", type=int, default=None)
    p.add_argument("--rmax-idx", type=int, default=None)
    p.add_argument("--max-range-km", type=float, default=None)
    p.add_argument("--support-threshold-snr", type=float, default=8.0)
    p.add_argument("--support-neighbor-km", type=float, default=10.0)
    p.add_argument("--max-active-ranges", type=int, default=48)
    p.add_argument("--max-iter", type=int, default=18)
    p.add_argument("--lstsq-rcond", type=float, default=1e-4)
    p.add_argument("--residual-peak-fraction", type=float, default=0.08)
    p.add_argument("--candidate-source", choices=["rdmf", "deconv", "none"], default="rdmf")
    p.add_argument("--candidate-product-dir", default="products/trail_echo_deconv")
    p.add_argument("--candidate-snr-db", type=float, default=6.0)
    p.add_argument("--candidate-rd-snr-db", type=float, default=3.0)
    p.add_argument("--candidate-neighbor-km", type=float, default=10.0)
    p.add_argument("--support-score", choices=["rdmf", "stationary"], default="rdmf")
    p.add_argument("--range-doppler-engine", choices=["auto", "avx", "numpy"], default="auto")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--trajectory-json", default=str(DEFAULT_TRAJECTORY_JSON))
    p.add_argument("--altitude-min-km", type=float, default=10.0)
    p.add_argument("--altitude-max-km", type=float, default=150.0)
    p.add_argument("--trajectory-range-pad-km", type=float, default=9.0)
    p.add_argument("--doppler-color-abs-hz", type=float, default=12.0)
    p.add_argument("--plot-range-mode", choices=["full", "trajectory"], default="full")
    p.add_argument("--plot-residuals", action="store_true", default=True)
    p.add_argument("--no-plot-residuals", action="store_false", dest="plot_residuals")
    p.add_argument("--plot-hsv", action="store_true", default=False)
    p.add_argument("--hsv-doppler-abs-hz", type=float, default=6.0)
    p.add_argument("--hsv-value-max-db", type=float, default=25.0)
    p.add_argument("--hsv-width-full-saturation-hz", type=float, default=0.5)
    p.add_argument("--hsv-width-zero-saturation-hz", type=float, default=6.0)
    p.add_argument("--product-dir", default=str(DEFAULT_PRODUCT_DIR))
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    global RANGE_DOPPLER_ENGINE
    RANGE_DOPPLER_ENGINE = args.range_doppler_engine
    effective_range_resolution_m = args.range_resolution_m / float(args.range_interp)
    args.neighbor_gates = neighbor_gates_from_km(args.support_neighbor_km, effective_range_resolution_m)
    args.candidate_neighbor_gates = neighbor_gates_from_km(args.candidate_neighbor_km, effective_range_resolution_m)

    comm = None
    rank = 0
    size = 1
    if not args.plot_only:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        except ModuleNotFoundError:
            comm = None
            rank = 0
            size = 1
    args.mpi_comm = comm
    args.mpi_rank = rank
    args.mpi_size = size

    product_dir = Path(args.product_dir)
    if args.plot_only:
        products = [product_dir / f"trail_sparse_deconv_{station}.h5" for station in args.stations]
    else:
        products = [process_station(args, station) for station in args.stations]
    if rank != 0:
        return
    plot_paths = plot_products(
        products,
        Path(args.fig_dir),
        resolve_path(args.data_dir),
        resolve_path(args.trajectory_json),
        args.altitude_min_km,
        args.altitude_max_km,
        args.trajectory_range_pad_km,
        args.doppler_color_abs_hz,
        args.plot_range_mode,
        args.plot_residuals,
        args.plot_hsv,
        args.hsv_doppler_abs_hz,
        args.hsv_value_max_db,
        args.hsv_width_full_saturation_hz,
        args.hsv_width_zero_saturation_hz,
    )
    for path in plot_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
