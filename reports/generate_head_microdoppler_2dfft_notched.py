#!/usr/bin/env python3
"""Generate head-echo micro-Doppler using a 2-D FFT notch per RD patch."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler import (  # noqa: E402
    STATION_LABELS,
    STATIONS,
    best_fit_doppler_hz,
    detrend_window,
    estimate_range_gate_dc,
    get_codes,
    load_doppler_model,
    load_station_observations,
    read_channel_files,
    read_station_channels,
    rf_file_for_time,
)
from reports.sparse_voltage_subtraction import (  # noqa: E402
    DEFAULT_SPARSE_PRODUCT_DIR,
    subtract_sparse_voltage_from_raw,
)

RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
DATA_DIR = ROOT / "2026_05_30_bolide"
FIG_DIR = ROOT / "reports" / "figures"
OUT_FIG = FIG_DIR / "head_echo_microdoppler_2dfft_notched.png"
OUT_CACHE = ROOT / "reports" / "head_echo_microdoppler_2dfft_notched.h5"
OUT_SUMMARY = ROOT / "reports" / "head_microdoppler_summary.tex"
EVENT_T0 = 1780164380.0
DEFAULT_PATCH_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_polarized_residual_dcmean.h5"
DEFAULT_FIT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_v0p2p6_polarized_residual_dcmean.csv"

G = {}


def robust_channel_noise_power(files: list[Path], block_samples: int) -> float:
    powers = []
    z, _sr, _idx0 = read_channel_files(files, start_second=0.0, seconds=None)
    block_samples = max(1024, int(block_samples))
    for i0 in range(0, len(z), block_samples):
        block = z[i0 : i0 + block_samples]
        if len(block) < 1024:
            continue
        center = np.median(block.real) + 1.0j * np.median(block.imag)
        powers.append(float(np.mean(np.abs(block - center) ** 2)))
    if not powers:
        raise ValueError("No blocks available for channel-noise power estimate")
    return float(np.median(powers))


def estimate_channel_noise_equalizer(
    data_dir: Path,
    station: str,
    channels: np.ndarray,
    iso_hour: str,
    block_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(channels) != 2:
        raise ValueError("Channel-noise equalization currently expects exactly two channels")
    powers = []
    for ch in channels:
        powers.append(robust_channel_noise_power(rf_file_for_time(data_dir, station, int(ch), iso_hour), block_samples))
    powers_arr = np.asarray(powers, dtype=np.float64)
    if not np.all(np.isfinite(powers_arr)) or np.any(powers_arr <= 0.0):
        raise ValueError(f"Invalid channel-noise powers for {station}: {powers_arr}")
    scales = np.sqrt(powers_arr[0] / powers_arr).astype(np.float32)
    scales[0] = 1.0
    return scales, powers_arr


def doppler_repetition_notch_columns(shape: tuple[int, int], period_bins: int, notch_half_width: int) -> np.ndarray:
    _ny, nx = shape
    fund = nx // period_bins
    cols = np.zeros(nx, dtype=bool)
    if fund <= 0:
        return cols
    for hidx in range(1, nx // (2 * fund) + 1):
        k = hidx * fund
        if k > nx // 2:
            break
        for kk in (k, (-k) % nx):
            for col in range(kk - notch_half_width, kk + notch_half_width + 1):
                cols[col % nx] = True
    return cols


def inpaint_notched_doppler_columns(img_db: np.ndarray, notched_cols: np.ndarray) -> np.ndarray:
    out = np.asarray(img_db, dtype=float).copy()
    nx = out.shape[1]
    for col in np.flatnonzero(notched_cols):
        left = (col - 1) % nx
        while notched_cols[left]:
            left = (left - 1) % nx
        right = (col + 1) % nx
        while notched_cols[right]:
            right = (right + 1) % nx
        out[:, col] = 0.5 * (out[:, left] + out[:, right])
    return out


def notch_2d_doppler_repetition(
    img_db: np.ndarray,
    period_bins: int,
    notch_half_width: int,
    *,
    fft_domain: str = "db",
    inpaint_columns: bool = False,
) -> np.ndarray:
    notched_cols = doppler_repetition_notch_columns(img_db.shape, period_bins, notch_half_width)
    if not np.any(notched_cols):
        return np.asarray(img_db, dtype=float)
    notch = np.broadcast_to(notched_cols[None, :], img_db.shape)
    if fft_domain == "db":
        img0 = img_db - np.mean(img_db)
        F = np.fft.fft2(img0)
        filtered = np.fft.ifft2(F * (~notch)).real + np.mean(img_db)
    elif fft_domain == "linear":
        power = 10.0 ** (np.asarray(img_db, dtype=float) / 10.0)
        power0 = power - np.mean(power)
        F = np.fft.fft2(power0)
        filtered_power = np.fft.ifft2(F * (~notch)).real + np.mean(power)
        filtered = 10.0 * np.log10(np.maximum(filtered_power, 1e-30))
    else:
        raise ValueError(f"Unknown fft_domain {fft_domain!r}")
    if inpaint_columns:
        filtered = inpaint_notched_doppler_columns(filtered, notched_cols)
    return filtered


def init_station(payload: dict, station: str, args: argparse.Namespace) -> None:
    doppler_model = load_doppler_model(payload, DATA_DIR)
    rdmf_file = RDMF_DIR / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(rdmf_file, "r") as h:
        channels = h["channels"][()]
        seeds = h["seeds"][()]
        code_len = int(h["code_len"][()])
        range_interp = int(h["range_interp"][()])
        sample_rate = float(h["sample_rate"][()])
        range_resolution_m = float(h["range_resolution_m"][()])
        start_second = float(h.attrs["start_second"])
        seconds = max(float(h.attrs["seconds"]), 10.0)
        iso_hour = str(h.attrs["iso_hour"])
        t0_sample = float(h["t0_sample"][()])
        dc_cancel = bool(getattr(args, "dc_cancel", False))
        dc_cancel_seconds = float(getattr(args, "dc_cancel_seconds", float(h["dc_cancel_seconds"][()]) if "dc_cancel_seconds" in h else 5.0))
        dc_cancel_start_second = float(
            getattr(args, "dc_cancel_start_second", float(h["dc_cancel_start_second"][()]) if "dc_cancel_start_second" in h else 0.0)
        )

    sample_rate_i = sample_rate * range_interp
    code_len_i = code_len * range_interp
    win_len = args.n_coh * code_len_i
    step_len = args.step_codes * code_len_i
    coherent_window_s = args.n_coh * code_len / sample_rate
    total_window_s = ((args.n_incoh - 1) * args.step_codes + args.n_coh) * code_len / sample_rate
    doppler_zero_pad_factor = int(getattr(args, "doppler_zero_pad_factor", 1))
    if doppler_zero_pad_factor < 1:
        raise ValueError("doppler_zero_pad_factor must be at least 1")
    fft_len = doppler_zero_pad_factor * win_len
    fvec_full = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / sample_rate_i))
    gate_m = range_resolution_m / range_interp
    codes = get_codes(seeds, code_len, args.n_coh, range_interp)
    sparse_subtract = bool(getattr(args, "sparse_voltage_subtract", True))
    sparse_meta = None
    if sparse_subtract:
        sparse_product_dir = Path(getattr(args, "sparse_product_dir", DEFAULT_SPARSE_PRODUCT_DIR))
        sparse_product = sparse_product_dir / f"trail_sparse_deconv_{station}.h5"
        zc, _sr, abs_index0, sparse_meta = subtract_sparse_voltage_from_raw(
            data_dir=DATA_DIR,
            station=station,
            channels=channels,
            iso_hour=iso_hour,
            start_second=start_second,
            seconds=seconds + 0.5,
            range_interp=range_interp,
            sparse_product=sparse_product,
            gain=float(getattr(args, "sparse_voltage_gain", 1.0)),
            mode=str(getattr(args, "sparse_voltage_mode", "full")),
        )
        dc_cancel = False
    else:
        zc, _sr, abs_index0 = read_station_channels(
            DATA_DIR,
            station,
            channels,
            iso_hour,
            start_second=start_second,
            seconds=seconds + 0.5,
            range_interp=range_interp,
        )
    if abs(abs_index0 - t0_sample) > 1:
        raise ValueError(f"{station}: raw t0 {abs_index0} does not match RDMF t0 {t0_sample}")
    channel_noise_equalizer = np.ones(len(channels), dtype=np.float32)
    channel_noise_powers = np.full(len(channels), np.nan, dtype=np.float64)
    if bool(getattr(args, "polarization_equalize_noise", False)):
        channel_noise_equalizer, channel_noise_powers = estimate_channel_noise_equalizer(
            DATA_DIR,
            station,
            channels,
            iso_hour,
            int(getattr(args, "polarization_equalize_block_samples", 100000)),
        )
        for chi, scale in enumerate(channel_noise_equalizer):
            zc[chi] = (zc[chi] * scale).astype(np.complex64, copy=False)
        print(
            f"{station}: channel noise powers {channel_noise_powers[0]:.6g}, {channel_noise_powers[1]:.6g}; "
            f"equalizer {channel_noise_equalizer[0]:.6g}, {channel_noise_equalizer[1]:.6g}",
            flush=True,
        )

    G.clear()
    G.update(
        {
            "args": args,
            "station": station,
            "doppler_model": doppler_model,
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_rate_i": sample_rate_i,
            "t0_sample": t0_sample,
            "fvec_full": fvec_full,
            "gate_m": gate_m,
            "win_len": win_len,
            "fft_len": fft_len,
            "step_len": step_len,
            "codes": codes,
            "zc": zc,
            "channel_noise_equalizer": channel_noise_equalizer,
            "channel_noise_powers": channel_noise_powers,
            "dc_cancel": dc_cancel,
            "sparse_subtraction": sparse_meta,
            "dc_cancel_seconds": dc_cancel_seconds,
            "dc_cancel_start_second": dc_cancel_start_second,
            "range_interp": range_interp,
            "iso_hour": iso_hour,
            "coherent_window_s": coherent_window_s,
            "total_window_s": total_window_s,
            "doppler_zero_pad_factor": doppler_zero_pad_factor,
        }
    )


def compute_filtered_patch(item: tuple[int, dict]) -> dict | None:
    idx, obs = item
    args = G["args"]
    station = G["station"]
    bulk = best_fit_doppler_hz(G["doppler_model"], station, float(obs["unix_time"]))
    fvec_full = G["fvec_full"]
    fidx = np.where(np.abs(fvec_full - bulk) <= args.doppler_half_width_hz)[0]
    rel_f = fvec_full[fidx] - bulk

    gate_m = G["gate_m"]
    head_range_m = float(obs["range_m"])
    ridx = np.arange(
        max(0, int(np.floor((head_range_m - args.range_half_width_km * 1000.0) / gate_m))),
        int(np.ceil((head_range_m + args.range_half_width_km * 1000.0) / gate_m)) + 1,
        dtype=np.int64,
    )
    raw_sample = int(round(float(obs["unix_time"]) * G["sample_rate"] - G["t0_sample"]))
    start_i = raw_sample * G["range_interp"]
    read_len = G["win_len"] + int(ridx[-1]) + 1
    incoh_stop_i = start_i + (args.n_incoh - 1) * G["step_len"] + read_len
    if start_i < 0 or incoh_stop_i >= min(map(len, G["zc"])):
        return None

    dc_offsets = None
    if G["dc_cancel"]:
        zc_cal, _sr_cal, _idx_cal = read_station_channels(
            DATA_DIR,
            station,
            G["channels"],
            G["iso_hour"],
            start_second=G["dc_cancel_start_second"],
            # The averaged DC interval remains dc_cancel_seconds long; the
            # extra trailing read samples only allow delayed range gates to be
            # evaluated over that same interval.
            seconds=G["dc_cancel_seconds"] + (float(ridx[-1]) + 1.0) / G["sample_rate_i"] + 2.0,
            range_interp=G["range_interp"],
        )
        if bool(getattr(args, "polarization_equalize_noise", False)):
            for chi, scale in enumerate(G["channel_noise_equalizer"]):
                zc_cal[chi] = (zc_cal[chi] * scale).astype(np.complex64, copy=False)
        dc_offsets = estimate_range_gate_dc(zc_cal, ridx, G["sample_rate_i"], G["dc_cancel_seconds"])

    G.pop("polarization_result", None)
    gamma = None
    S = None
    if bool(getattr(args, "polarization_combine", False)) and str(getattr(args, "polarization_method", "phase")) == "grid_image":
        gamma, S = estimate_image_peak_polarization_gamma(start_i, read_len, ridx, fidx, dc_offsets)
    elif bool(getattr(args, "polarization_combine", False)):
        seed_rel_hz = float(obs["doppler_hz"]) - bulk
        seed_range_i = int(np.argmin(np.abs(ridx.astype(np.float64) * gate_m - head_range_m)))
        seed_freq_i = int(np.argmin(np.abs(rel_f - seed_rel_hz)))
        gamma = estimate_head_echo_polarization_gamma(start_i, read_len, ridx, fidx, seed_range_i, seed_freq_i, dc_offsets)

    if S is None:
        S = np.zeros((len(ridx), len(fidx)), dtype=np.float64)
        for incoh_i in range(args.n_incoh):
            incoh_start_i = start_i + incoh_i * G["step_len"]
            channel_iter = enumerate(G["zc"])
            combined_dc_offsets = None
            if gamma is not None:
                zc = np.zeros(read_len, dtype=np.complex64)
                for weight, z in zip(gamma, G["zc"], strict=True):
                    zc += weight * detrend_window(z[incoh_start_i : incoh_start_i + read_len])
                channel_iter = enumerate([zc])
                if dc_offsets is not None:
                    combined_dc_offsets = np.sum(gamma[:, None] * dc_offsets, axis=0, dtype=np.complex64)[None, :]
            for chi, z in channel_iter:
                z0 = z if gamma is not None else detrend_window(z[incoh_start_i : incoh_start_i + read_len])
                windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
                if dc_offsets is not None:
                    offsets = combined_dc_offsets if combined_dc_offsets is not None else dc_offsets
                    windows = windows - offsets[chi, :, None]
                for code in G["codes"]:
                    Z = np.fft.fftshift(
                        np.fft.fft(windows * np.conj(code), n=G["fft_len"], axis=1),
                        axes=1,
                    )[:, fidx]
                    S += np.abs(Z) ** 2

    img_db = 10.0 * np.log10(np.maximum(S, 1e-30))
    filtered_db = notch_2d_doppler_repetition(
        img_db,
        args.n_coh,
        args.notch_half_width,
        fft_domain=args.fft_filter_domain,
        inpaint_columns=args.inpaint_notched_columns,
    )
    range_m = ridx.astype(np.float64) * gate_m
    polarization_result = G.get("polarization_result", {})
    return {
        "idx": idx,
        "time": float(obs["unix_time"]),
        "range_m": range_m,
        "seed_range_m": head_range_m,
        "model_doppler_hz": bulk,
        "picked_doppler_hz": float(obs["doppler_hz"]),
        "rel_freqs": rel_f,
        "raw_db": img_db,
        "filtered_db": filtered_db,
        "coherent_window_s": G["coherent_window_s"],
        "total_window_s": G["total_window_s"],
        "n_coh": int(args.n_coh),
        "n_incoh": int(args.n_incoh),
        "step_codes": int(args.step_codes),
        "doppler_zero_pad_factor": int(G["doppler_zero_pad_factor"]),
        "channel_noise_equalizer": np.asarray(G.get("channel_noise_equalizer", []), dtype=np.float32),
        "channel_noise_powers": np.asarray(G.get("channel_noise_powers", []), dtype=np.float64),
        "polarization_gamma": np.asarray(gamma, dtype=np.complex64) if gamma is not None else np.asarray([], dtype=np.complex64),
        "polarization_metric": float(polarization_result.get("metric", np.nan)),
        "polarization_signal_power": float(polarization_result.get("signal_power", np.nan)),
        "polarization_noise_power": float(polarization_result.get("noise_power", np.nan)),
        "polarization_metric_kind": str(polarization_result.get("metric_kind", "")),
        "polarization_metric_mask_pixels": int(polarization_result.get("mask_pixels", 0)),
    }


def estimate_head_echo_polarization_gamma(
    start_i: int,
    read_len: int,
    ridx: np.ndarray,
    fidx: np.ndarray,
    seed_range_i: int,
    seed_freq_i: int,
    dc_offsets: np.ndarray | None,
) -> np.ndarray | None:
    if len(G["zc"]) != 2:
        raise ValueError("Polarization combining currently expects exactly two channels")
    args = G["args"]
    responses = np.zeros((2, len(G["codes"])), dtype=np.complex128)
    for chi, z in enumerate(G["zc"]):
        z0 = detrend_window(z[start_i : start_i + read_len])
        windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
        if dc_offsets is not None:
            windows = windows - dc_offsets[chi, :, None]
        row = windows[seed_range_i]
        for code_i, code in enumerate(G["codes"]):
            spec = np.fft.fftshift(np.fft.fft(row * np.conj(code), n=G["fft_len"]))
            responses[chi, code_i] = spec[fidx[seed_freq_i]]
    if str(getattr(args, "polarization_method", "phase")) == "grid":
        return grid_search_polarization_gamma(responses)
    cross = np.vdot(responses[1], responses[0])
    if not np.isfinite(cross) or abs(cross) == 0.0:
        gamma = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64)
    else:
        gamma = np.array([1.0 + 0.0j, np.exp(1.0j * np.angle(cross))], dtype=np.complex64)
    gamma /= np.sqrt(float(np.sum(np.abs(gamma) ** 2)))
    return gamma


def grid_search_polarization_gamma(responses: np.ndarray) -> np.ndarray:
    """Grid-search normalized two-channel polarization states.

    Candidate states are gamma = [cos(theta), exp(i phi) sin(theta)] with
    theta in [0, pi/2].  With equal white channel noise this keeps unit noise
    gain, so maximizing summed matched-filter power at the head-echo bin is
    equivalent to maximizing SNR.
    """
    args = G["args"]
    n_theta = int(getattr(args, "polarization_grid_theta", 91))
    n_phase = int(getattr(args, "polarization_grid_phase", 361))
    theta = np.linspace(0.0, 0.5 * np.pi, max(2, n_theta), dtype=np.float64)
    phase = np.linspace(-np.pi, np.pi, max(4, n_phase), endpoint=False, dtype=np.float64)
    best_power = -np.inf
    best_gamma = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64)
    for th in theta:
        a = np.cos(th)
        b_amp = np.sin(th)
        gamma0 = a + 0.0j
        gamma1 = b_amp * np.exp(1.0j * phase)
        combined = gamma0 * responses[0][None, :] + gamma1[:, None] * responses[1][None, :]
        power = np.sum(np.abs(combined) ** 2, axis=1)
        idx = int(np.nanargmax(power))
        if float(power[idx]) > best_power:
            best_power = float(power[idx])
            best_gamma = np.array([gamma0, gamma1[idx]], dtype=np.complex64)
    best_gamma /= np.sqrt(float(np.sum(np.abs(best_gamma) ** 2)))
    return best_gamma


def channel_code_images(
    start_i: int,
    read_len: int,
    ridx: np.ndarray,
    fidx: np.ndarray,
    dc_offsets: np.ndarray | None,
) -> np.ndarray:
    args = G["args"]
    out = np.empty((args.n_incoh, len(G["zc"]), len(G["codes"]), len(ridx), len(fidx)), dtype=np.complex64)
    for incoh_i in range(args.n_incoh):
        incoh_start_i = start_i + incoh_i * G["step_len"]
        for chi, z in enumerate(G["zc"]):
            z0 = detrend_window(z[incoh_start_i : incoh_start_i + read_len])
            windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
            if dc_offsets is not None:
                windows = windows - dc_offsets[chi, :, None]
            for code_i, code in enumerate(G["codes"]):
                out[incoh_i, chi, code_i] = np.fft.fftshift(
                    np.fft.fft(windows * np.conj(code), n=G["fft_len"], axis=1),
                    axes=1,
                )[:, fidx].astype(np.complex64)
    return out


def estimate_image_peak_polarization_gamma(
    start_i: int,
    read_len: int,
    ridx: np.ndarray,
    fidx: np.ndarray,
    dc_offsets: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(G["zc"]) != 2:
        raise ValueError("Polarization combining currently expects exactly two channels")
    args = G["args"]
    images = channel_code_images(start_i, read_len, ridx, fidx, dc_offsets)
    z0 = images[:, 0]
    z1 = images[:, 1]
    raw0 = np.empty((args.n_incoh, read_len), dtype=np.complex64)
    raw1 = np.empty((args.n_incoh, read_len), dtype=np.complex64)
    for incoh_i in range(args.n_incoh):
        incoh_start_i = start_i + incoh_i * G["step_len"]
        raw0[incoh_i] = detrend_window(G["zc"][0][incoh_start_i : incoh_start_i + read_len])
        raw1[incoh_i] = detrend_window(G["zc"][1][incoh_start_i : incoh_start_i + read_len])
    n_theta = int(getattr(args, "polarization_grid_theta", 91))
    n_phase = int(getattr(args, "polarization_grid_phase", 361))
    theta = np.linspace(0.0, 0.5 * np.pi, max(2, n_theta), dtype=np.float64)
    phase = np.linspace(-np.pi, np.pi, max(4, n_phase), endpoint=False, dtype=np.float64)
    best_metric = -np.inf
    best_signal_power = np.nan
    best_noise_power = np.nan
    best_gamma = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64)
    best_image = None
    metric_mask = G.get("polarization_metric_mask")
    if metric_mask is not None:
        metric_mask = np.asarray(metric_mask, dtype=bool)
        if metric_mask.shape != (len(ridx), len(fidx)):
            raise ValueError(
                "polarization_metric_mask shape "
                f"{metric_mask.shape} does not match RD image shape {(len(ridx), len(fidx))}"
            )
        if not np.any(metric_mask):
            raise ValueError("polarization_metric_mask is empty")
    phase_chunk = int(getattr(args, "polarization_phase_chunk", 32))
    phase_chunk = max(1, phase_chunk)
    for th in theta:
        a = np.cos(th)
        b_amp = np.sin(th)
        for p0 in range(0, len(phase), phase_chunk):
            ph = phase[p0 : p0 + phase_chunk]
            gamma1 = (b_amp * np.exp(1.0j * ph)).astype(np.complex64)
            combined = a * z0[None, ...] + gamma1[:, None, None, None, None] * z1[None, ...]
            power_images = np.sum(np.abs(combined) ** 2, axis=(1, 2), dtype=np.float64)
            if metric_mask is None:
                signal_power = np.nanmax(power_images.reshape(power_images.shape[0], -1), axis=1)
            else:
                signal_power = np.nansum(power_images[:, metric_mask], axis=1)
            raw_combined = a * raw0[None, ...] + gamma1[:, None, None] * raw1[None, ...]
            noise_power = np.mean(np.abs(raw_combined) ** 2, axis=(1, 2), dtype=np.float64)
            metric = signal_power / np.maximum(noise_power, 1e-30)
            idx = int(np.nanargmax(metric))
            score = float(metric[idx])
            if score > best_metric:
                best_metric = score
                best_signal_power = float(signal_power[idx])
                best_noise_power = float(noise_power[idx])
                best_gamma = np.array([a + 0.0j, gamma1[idx]], dtype=np.complex64)
                best_image = power_images[idx].copy()
    best_gamma /= np.sqrt(float(np.sum(np.abs(best_gamma) ** 2)))
    if best_image is None:
        raise RuntimeError("Polarization grid-image search failed")
    G["polarization_result"] = {
        "metric": best_metric,
        "signal_power": best_signal_power,
        "noise_power": best_noise_power,
        "metric_kind": "gaussian_mask_sum" if metric_mask is not None else "image_peak",
        "mask_pixels": int(np.sum(metric_mask)) if metric_mask is not None else 0,
    }
    return best_gamma, best_image


def process_observation(item: tuple[int, dict]) -> dict | None:
    patch = compute_filtered_patch(item)
    if patch is None:
        return None
    head_range_m = patch["seed_range_m"]
    range_m = patch["range_m"]
    filtered_db = patch["filtered_db"]
    row_idx = int(np.argmin(np.abs(range_m - head_range_m)))
    return {
        "idx": patch["idx"],
        "time": patch["time"],
        "range_km": head_range_m / 1000.0,
        "model_doppler_hz": patch["model_doppler_hz"],
        "picked_doppler_hz": patch["picked_doppler_hz"],
        "rel_freqs": patch["rel_freqs"],
        "row_db": filtered_db[row_idx, :],
    }


def run_station(station: str, obs: list[dict], payload: dict, args: argparse.Namespace) -> dict:
    print(f"{station}: loading raw data", flush=True)
    init_station(payload, station, args)
    n_workers = min(args.workers, len(obs))
    print(f"{station}: processing {len(obs)} rows with {n_workers} workers", flush=True)
    if n_workers <= 1:
        rows = [process_observation(item) for item in enumerate(obs)]
    else:
        ctx = mp.get_context("fork")
        # On Linux/fork this lets workers share the already-read RF arrays via
        # copy-on-write instead of every worker rereading the station files.
        with ctx.Pool(n_workers) as pool:
            rows = []
            for row in pool.imap_unordered(process_observation, list(enumerate(obs)), chunksize=1):
                rows.append(row)
                done = len(rows)
                if done == 1 or done == len(obs) or done % max(1, len(obs) // 10) == 0:
                    print(f"{station}: {done}/{len(obs)} rows returned", flush=True)
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda row: row["time"])
    if not rows:
        raise ValueError(f"{station}: no rows generated")
    rel_ref = rows[0]["rel_freqs"]
    spec = []
    for row in rows:
        vals = row["row_db"]
        rel = row["rel_freqs"]
        if len(rel) != len(rel_ref) or np.nanmax(np.abs(rel - rel_ref)) > 0.51 * np.nanmedian(np.diff(rel_ref)):
            vals = np.interp(rel_ref, rel, vals, left=np.nan, right=np.nan)
        spec.append(vals)
    spec = np.asarray(spec)
    spec -= np.nanmedian(spec)
    return {
        "station": station,
        "times": np.asarray([row["time"] for row in rows]),
        "time_rel": np.asarray([row["time"] - EVENT_T0 for row in rows]),
        "ranges_km": np.asarray([row["range_km"] for row in rows]),
        "model_dopplers_hz": np.asarray([row["model_doppler_hz"] for row in rows]),
        "picked_dopplers_hz": np.asarray([row["picked_doppler_hz"] for row in rows]),
        "rel_freqs": rel_ref,
        "spec_db": spec,
    }


def plot_results(results: list[dict], out_fig: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    finite = np.concatenate([res["spec_db"][np.isfinite(res["spec_db"])] for res in results])
    vmin = 0.0
    vmax = max(8.0, float(np.nanpercentile(finite, 99.4))) if finite.size else 18.0
    mappable = None
    for ax, res in zip(axes, results):
        ax.set_facecolor("black")
        mappable = ax.pcolormesh(
            res["time_rel"],
            res["rel_freqs"],
            res["spec_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.set_ylabel("$f-f_{D,model}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo")
        ax.text(
            0.01,
            0.93,
            f"range {np.nanmedian(res['ranges_km']):.0f} km, 2D FFT notch",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, pad=0.012)
        cb.set_label("2D-FFT-notched matched-filter power (dB rel.)")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "2D FFT-notched head-echo micro-Doppler rows"
        h.attrs["event_t0_unix"] = EVENT_T0
        h["stations"] = np.asarray([res["station"].encode("utf-8") for res in results])
        for res in results:
            g = h.create_group(res["station"])
            g["time_rel"] = res["time_rel"]
            g["ranges_km"] = res["ranges_km"]
            g["model_dopplers_hz"] = res["model_dopplers_hz"]
            g["picked_dopplers_hz"] = res["picked_dopplers_hz"]
            g["rel_freqs"] = res["rel_freqs"]
            g["spec_db"] = res["spec_db"]


def load_results(path: Path) -> list[dict]:
    results = []
    with h5py.File(path, "r") as h:
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            results.append(
                {
                    "station": station,
                    "time_rel": g["time_rel"][()],
                    "ranges_km": g["ranges_km"][()],
                    "model_dopplers_hz": g["model_dopplers_hz"][()],
                    "picked_dopplers_hz": g["picked_dopplers_hz"][()],
                    "rel_freqs": g["rel_freqs"][()],
                    "spec_db": g["spec_db"][()],
                }
            )
    return results


def load_fit_summary(path: Path, stations: list[str]) -> dict[str, dict[int, dict]]:
    out = {station: {} for station in stations}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            station = row["station"]
            if station not in out:
                continue
            if row.get("success", "").lower() != "true":
                continue
            out[station][int(row["idx"])] = row
    return out


def run_station_from_patch_cache(h: h5py.File, station: str, fit_rows: dict[int, dict], args: argparse.Namespace) -> tuple[dict, list[dict]]:
    group = h[station]
    spectra = []
    rows = []
    rel_ref = None
    keys = sorted(key for key in group.keys() if key.startswith("patch_"))
    for key in keys:
        patch = group[key]
        idx = int(patch.attrs["idx"])
        fit = fit_rows.get(idx)
        if fit is None or not bool(patch.attrs.get("valid", False)):
            continue
        range_m = patch["range_m"][()].astype(float)
        rel_freqs = patch["rel_freqs"][()].astype(float)
        raw_db = patch["raw_db"][()].astype(float)
        n_coh = int(patch.attrs.get("n_coh", h.attrs.get("n_coh", args.n_coh)))
        filtered_db = notch_2d_doppler_repetition(
            raw_db,
            n_coh,
            args.notch_half_width,
            fft_domain=args.fft_filter_domain,
            inpaint_columns=args.inpaint_notched_columns,
        )
        fit_range_m = float(fit["fit_range_m"])
        fit_rel_hz = float(fit["fit_rel_doppler_hz"])
        row_i = int(np.argmin(np.abs(range_m - fit_range_m)))
        shifted_rel = rel_freqs - fit_rel_hz
        if rel_ref is None:
            df = float(np.nanmedian(np.diff(rel_freqs)))
            rel_ref = np.arange(-args.rel_half_width_hz, args.rel_half_width_hz + 0.5 * df, df)
        spectrum = np.interp(rel_ref, shifted_rel, filtered_db[row_i, :], left=np.nan, right=np.nan)
        spectra.append(spectrum)
        rows.append(
            {
                "station": station,
                "idx": idx,
                "time_unix": float(fit["time_unix"]),
                "time_rel_s": float(fit["time_rel_s"]),
                "patch_time_unix": float(patch.attrs["time_unix"]),
                "patch_time_rel_s": float(patch.attrs["time_rel_s"]),
                "fit_range_m": fit_range_m,
                "fit_doppler_hz": float(fit["fit_doppler_hz"]),
                "fit_rel_doppler_hz": fit_rel_hz,
                "seed_range_m": float(fit["seed_range_m"]),
                "seed_doppler_hz": float(fit["seed_doppler_hz"]),
                "model_doppler_hz": float(fit["model_doppler_hz"]),
                "range_row_m": float(range_m[row_i]),
                "n_coh": n_coh,
                "n_incoh": int(patch.attrs.get("n_incoh", h.attrs.get("n_incoh", args.n_incoh))),
                "step_codes": int(patch.attrs.get("step_codes", h.attrs.get("step_codes", args.step_codes))),
                "doppler_zero_pad_factor": int(patch.attrs.get("doppler_zero_pad_factor", h.attrs.get("doppler_zero_pad_factor", 1))),
                "head_echo_product_version": str(h.attrs.get("head_echo_product_version", args.product_version)),
            }
        )
    if not spectra:
        raise ValueError(f"{station}: no patch-cache spectra matched Gaussian fit summary")
    order = np.argsort([row["time_rel_s"] for row in rows])
    spectra = np.asarray(spectra, dtype=float)[order, :]
    rows = [rows[int(i)] for i in order]
    spec_db = spectra - np.nanmedian(spectra)
    return (
        {
            "station": station,
            "time_rel": np.asarray([row["time_rel_s"] for row in rows], dtype=float),
            "ranges_km": np.asarray([row["fit_range_m"] / 1000.0 for row in rows], dtype=float),
            "fit_dopplers_hz": np.asarray([row["fit_doppler_hz"] for row in rows], dtype=float),
            "rel_freqs": rel_ref,
            "spec_db": spec_db,
        },
        rows,
    )


def save_patch_cache_results(results: list[dict], rows: list[dict], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Gaussian-recentered, unsmoothed 2D-FFT-notched head-echo micro-Doppler rows from cached range-Doppler patches"
        h.attrs["head_echo_product_version"] = str(args.product_version)
        h.attrs["event_t0_unix"] = EVENT_T0
        h.attrs["patch_cache"] = str(args.patch_cache)
        h.attrs["fit_summary"] = str(args.fit_summary)
        h.attrs["notch_half_width"] = int(args.notch_half_width)
        h.attrs["fft_filter_domain"] = str(args.fft_filter_domain)
        h.attrs["inpaint_notched_columns"] = bool(args.inpaint_notched_columns)
        h.attrs["smoothing"] = "none"
        h["stations"] = np.asarray([res["station"].encode("utf-8") for res in results])
        for res in results:
            g = h.create_group(res["station"])
            for name in ["time_rel", "ranges_km", "fit_dopplers_hz", "rel_freqs", "spec_db"]:
                g[name] = res[name]
        rg = h.create_group("rows")
        for i, row in enumerate(rows):
            sg = rg.create_group(f"row_{i:04d}")
            for key, val in row.items():
                sg.attrs[key] = val


def write_patch_cache_summary(results: list[dict], rows: list[dict], args: argparse.Namespace) -> None:
    by_station = {res["station"]: [] for res in results}
    for row in rows:
        by_station.setdefault(str(row["station"]), []).append(row)
    n_coh = int(rows[0]["n_coh"]) if rows else int(args.n_coh)
    n_incoh = int(rows[0]["n_incoh"]) if rows else int(args.n_incoh)
    step_codes = int(rows[0]["step_codes"]) if rows else int(args.step_codes)
    lines = [
        "% Auto-generated by reports/generate_head_microdoppler_2dfft_notched.py\n",
        f"\\newcommand{{\\HeadMicroDopplerProductVersion}}{{{args.product_version}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerStationCount}}{{{len(results)}}}\n",
        "\\newcommand{\\HeadMicroDopplerCodeSeedCount}{5}\n",
        f"\\newcommand{{\\HeadMicroDopplerNCoherentCodes}}{{{n_coh}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerNIncoherentSpectra}}{{{n_incoh}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerStepCodes}}{{{step_codes}}}\n",
        "\\newcommand{\\HeadMicroDopplerCleanIterations}{0}\n",
        "\\newcommand{\\HeadMicroDopplerCleanGain}{0.0}\n",
    ]
    for res in results:
        station_rows = by_station.get(res["station"], [])
        name = res["station"].title()
        ranges_km = np.asarray([row["fit_range_m"] / 1000.0 for row in station_rows], dtype=float)
        dopplers_hz = np.asarray([row["fit_doppler_hz"] for row in station_rows], dtype=float)
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}N}}{{{len(station_rows)}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianRangeKm}}{{{np.nanmedian(ranges_km):.0f}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianDopplerHz}}{{{np.nanmedian(dopplers_hz):.0f}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianPickedDopplerHz}}{{{np.nanmedian(dopplers_hz):.0f}}}\n")
    OUT_SUMMARY.write_text("".join(lines), encoding="utf-8")


def plot_results_plain(results: list[dict], out_fig: Path, args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    finite = np.concatenate([res["spec_db"][np.isfinite(res["spec_db"])] for res in results])
    vmin = args.vmin_db if args.vmin_db is not None else 0.0
    vmax = args.vmax_db if args.vmax_db is not None else (max(8.0, float(np.nanpercentile(finite, 99.4))) if finite.size else 18.0)
    mappable = None
    for ax, res in zip(axes, results):
        ax.set_facecolor("black")
        mappable = ax.pcolormesh(
            res["time_rel"],
            res["rel_freqs"],
            res["spec_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0.0, color="white", alpha=0.5, linestyle="--", linewidth=0.8)
        ax.set_ylim(-500.0, 500.0)
        ax.set_ylabel("$f-f_{D,Gaussian}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo")
        dax = ax.twinx()
        fit_dopplers = np.asarray(res["fit_dopplers_hz"], dtype=float)
        dax.plot(res["time_rel"], fit_dopplers, color="white", alpha=0.5, linewidth=1.0)
        finite_dop = fit_dopplers[np.isfinite(fit_dopplers)]
        if finite_dop.size:
            pad = max(25.0, 0.05 * float(np.nanmax(finite_dop) - np.nanmin(finite_dop)))
            dax.set_ylim(float(np.nanmin(finite_dop) - pad), float(np.nanmax(finite_dop) + pad))
        dax.set_ylabel("$f_{D,Gaussian}$ (Hz)")
        dax.tick_params(axis="y")
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, pad=0.012)
        cb.set_label("2D-FFT-notched matched-filter power (dB rel.)")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_patch_cache_mode(args: argparse.Namespace) -> None:
    fit_rows = load_fit_summary(args.fit_summary, args.stations)
    results = []
    all_rows = []
    with h5py.File(args.patch_cache, "r") as h:
        for station in args.stations:
            print(f"{station}: micro-Doppler rows from cached unsmoothed 2D FFT-filtered patches", flush=True)
            result, rows = run_station_from_patch_cache(h, station, fit_rows[station], args)
            results.append(result)
            all_rows.extend(rows)
    save_patch_cache_results(results, all_rows, args.cache, args)
    write_patch_cache_summary(results, all_rows, args)
    plot_results_plain(results, args.out, args)
    print(args.cache, flush=True)
    print(args.out, flush=True)
    print(OUT_SUMMARY, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--fft-filter-domain", choices=("db", "linear"), default="linear")
    p.add_argument("--inpaint-notched-columns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dc-cancel", action="store_true", help="Apply range-dependent DC/modulo subtraction before the RD patch FFT.")
    p.add_argument("--dc-cancel-seconds", type=float, default=5.0)
    p.add_argument("--dc-cancel-start-second", type=float, default=0.0)
    p.add_argument("--sparse-voltage-subtract", action=argparse.BooleanOptionalAction, default=True, help="Subtract sparse range-dependent voltage estimates at 100 kHz before interpolation.")
    p.add_argument("--sparse-product-dir", type=Path, default=DEFAULT_SPARSE_PRODUCT_DIR)
    p.add_argument("--sparse-voltage-mode", choices=("full", "dc"), default="full")
    p.add_argument("--sparse-voltage-gain", type=float, default=1.0)
    p.add_argument("--polarization-combine", action="store_true")
    p.add_argument("--polarization-method", choices=("phase", "grid", "grid_image"), default="phase")
    p.add_argument("--polarization-equalize-noise", action="store_true")
    p.add_argument("--polarization-equalize-block-samples", type=int, default=100000)
    p.add_argument("--polarization-grid-theta", type=int, default=91)
    p.add_argument("--polarization-grid-phase", type=int, default=361)
    p.add_argument("--polarization-phase-chunk", type=int, default=32)
    p.add_argument("--out", type=Path, default=OUT_FIG)
    p.add_argument("--cache", type=Path, default=OUT_CACHE)
    p.add_argument("--plot-only", action="store_true", help="Plot from --cache without recomputing filtered RD patches.")
    p.add_argument("--from-patch-cache", action="store_true", help="Build micro-Doppler from cached RD patches and a Gaussian fit summary.")
    p.add_argument("--patch-cache", type=Path, default=DEFAULT_PATCH_CACHE)
    p.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    p.add_argument("--rel-half-width-hz", type=float, default=1800.0)
    p.add_argument("--product-version", default="0.2.6")
    p.add_argument("--vmin-db", type=float, default=None)
    p.add_argument("--vmax-db", type=float, default=None)
    args = p.parse_args()

    if args.from_patch_cache:
        run_patch_cache_mode(args)
        return

    if args.plot_only:
        results = load_results(args.cache)
    else:
        payload, observations = load_station_observations()
        results = [run_station(station, observations[station], payload, args) for station in args.stations]
        save_results(results, args.cache)
        print(args.cache, flush=True)
    plot_results(results, args.out)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
