"""Subtract sparse range-dependent voltage estimates from raw RF data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from range_doppler_mf_bolide import bpsk, interpolate_rect
from reports.generate_head_microdoppler import read_channel_files, rf_file_for_time


DEFAULT_SPARSE_PRODUCT_DIR = Path("products/trail_echo_sparse_deconv_full_3km_50code_rdmf_avx")


def _raw_codes(seeds: np.ndarray, code_len: int) -> np.ndarray:
    codes = []
    for seed in seeds:
        code = bpsk(int(seed), int(code_len))
        code = code / np.max(np.abs(code))
        codes.append(code.astype(np.complex64))
    return np.asarray(codes, dtype=np.complex64)


def subtract_sparse_voltage_from_raw(
    *,
    data_dir: Path,
    station: str,
    channels: np.ndarray,
    iso_hour: str,
    start_second: float,
    seconds: float,
    range_interp: int,
    sparse_product: Path,
    gain: float = 1.0,
    mode: str = "full",
) -> tuple[list[np.ndarray], float, int, dict]:
    """Read raw 100 kHz voltage, subtract sparse model, then interpolate.

    Parameters
    ----------
    mode:
        ``"full"`` subtracts the per-code sparse coefficient sequence.
        ``"dc"`` subtracts only the block-mean coefficient for each
        range/channel/transmitter.
    """
    raw_z = []
    sample_rate = None
    abs_index0 = None
    for ch in channels:
        z, sr, idx0 = read_channel_files(
            rf_file_for_time(data_dir, station, int(ch), iso_hour),
            start_second=start_second,
            seconds=seconds,
        )
        raw_z.append(z.astype(np.complex64, copy=False))
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
        if sample_rate != sr or abs_index0 != idx0:
            raise ValueError(f"{station}: channel {ch} time base mismatch")

    with h5py.File(sparse_product, "r") as h:
        product_station = h["station"][()].decode("utf-8") if isinstance(h["station"][()], bytes) else str(h["station"][()])
        if product_station != station:
            raise ValueError(f"Sparse product station {product_station!r} does not match {station!r}")
        seeds = h["seeds"][()]
        sparse_channels = h["channels"][()]
        code_len = int(h["code_len"][()])
        sparse_range_interp = int(h["range_interp"][()])
        coherent_codes = int(h["coherent_codes"][()])
        sparse_sample_rate = float(h["sample_rate"][()])
        sparse_t0_sample = int(h["t0_sample"][()])
        support_count = h["support_count"][()]
        support_range_idx = h["support_range_idx"][()]
        support_voltage = h["support_voltage"][()]
        dc_voltage = h["dc_voltage"][()] if "dc_voltage" in h else None

    if sparse_range_interp != 1:
        raise ValueError(f"Sparse subtraction expects sparse range_interp=1, got {sparse_range_interp}")
    if abs(float(sparse_sample_rate) - float(sample_rate)) > 1e-6:
        raise ValueError(f"Sparse sample rate {sparse_sample_rate} does not match raw sample rate {sample_rate}")
    if len(sparse_channels) != len(channels) or np.any(np.asarray(sparse_channels) != np.asarray(channels)):
        raise ValueError(f"Sparse channel order {sparse_channels} does not match requested channels {channels}")
    if mode not in {"full", "dc"}:
        raise ValueError(f"Unknown sparse subtraction mode {mode!r}")

    codes = _raw_codes(seeds, code_len)
    n_tx = codes.shape[0]
    offset = int(round(float(sparse_t0_sample) - float(abs_index0)))
    min_len = min(map(len, raw_z))
    model_power = np.zeros(len(raw_z), dtype=np.float64)
    blocks_used = 0
    terms = 0

    for bi in range(support_voltage.shape[0]):
        n_support = int(support_count[bi])
        has_dc = dc_voltage is not None and bi < dc_voltage.shape[0]
        if n_support <= 0 and not has_dc:
            continue
        used_block = False
        dc_coeff = None
        if mode == "dc" and n_support > 0:
            dc_coeff = np.mean(support_voltage[bi], axis=0)
        for ci in range(coherent_codes):
            start = offset + (bi * coherent_codes + ci) * code_len
            stop = start + code_len
            if stop <= 0 or start >= min_len:
                continue
            seg0 = max(start, 0)
            seg1 = min(stop, min_len)
            loc = np.arange(seg0 - start, seg1 - start, dtype=np.int64)
            for chi in range(len(raw_z)):
                model = np.zeros(seg1 - seg0, dtype=np.complex64)
                if has_dc:
                    model += dc_voltage[bi, ci, chi]
                    terms += 1
                for txi in range(n_tx):
                    code = codes[txi]
                    for si in range(n_support):
                        raw_ri = int(support_range_idx[bi, si])
                        if raw_ri < 0:
                            continue
                        if mode == "full":
                            coeff = support_voltage[bi, ci, chi, txi, si]
                        else:
                            coeff = dc_coeff[chi, txi, si]
                        if coeff == 0:
                            continue
                        model += coeff * code[(loc - raw_ri) % code_len]
                        terms += 1
                raw_z[chi][seg0:seg1] = (raw_z[chi][seg0:seg1] - gain * model).astype(np.complex64)
                model_power[chi] += float(np.sum(np.abs(model) ** 2))
            used_block = True
        if used_block:
            blocks_used += 1

    z_interp = [interpolate_rect(z, range_interp) for z in raw_z]
    meta = {
        "sparse_product": str(sparse_product),
        "gain": float(gain),
        "mode": mode,
        "blocks_used": int(blocks_used),
        "terms": int(terms),
        "model_power_per_channel": model_power,
    }
    return z_interp, float(sample_rate), int(abs_index0), meta
