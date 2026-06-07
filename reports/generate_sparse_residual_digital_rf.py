#!/usr/bin/env python3
"""Write Digital RF residual voltages after subtracting the sparse trail model."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from range_doppler_mf_bolide import bpsk  # noqa: E402
from station_config import STATION_ORDER  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "2026_05_30_bolide"
DEFAULT_OUT_DIR = ROOT / "2026_05_30_bolide_residual"
DEFAULT_SPARSE_DIR = ROOT / "products" / "trail_echo_sparse_deconv_full_3km_50code_rdmf_avx"


def raw_code(seed: int, code_len: int) -> np.ndarray:
    code = bpsk(int(seed), int(code_len))
    code = code / np.max(np.abs(code))
    return code.astype(np.complex64)


def load_rf_channel(channel_dir: Path) -> tuple[list[Path], list[int], np.ndarray, float]:
    files = sorted(channel_dir.glob("*/*.h5"))
    files = [path for path in files if path.name.startswith("rf@")]
    if not files:
        raise FileNotFoundError(f"No Digital RF files found in {channel_dir}")

    chunks = []
    starts = []
    sample_rate = None
    for path in files:
        with h5py.File(path, "r") as h:
            rf = h["rf_data"]
            if rf.dtype != np.dtype("complex64"):
                raise TypeError(f"{path} has dtype {rf.dtype}, expected complex64")
            sr = float(rf.attrs["sample_rate_numerator"]) / float(rf.attrs["sample_rate_denominator"])
            sample_rate = sr if sample_rate is None else sample_rate
            if abs(sr - sample_rate) > 1e-9:
                raise ValueError(f"Sample-rate mismatch in {path}: {sr} != {sample_rate}")
            starts.append(int(h["rf_data_index"][0, 0]))
            chunks.append(rf[:, 0].astype(np.complex64))
    return files, starts, np.concatenate(chunks).astype(np.complex64), float(sample_rate)


def write_rf_channel(files: list[Path], starts: list[int], residual: np.ndarray, data_dir: Path, out_dir: Path) -> None:
    pos = 0
    for in_path, start in zip(files, starts, strict=True):
        out_path = out_dir / in_path.relative_to(data_dir)
        n = None
        with h5py.File(in_path, "r") as hin:
            n = int(hin["rf_data"].shape[0])
        chunk = residual[pos : pos + n]
        pos += n
        with h5py.File(out_path, "r+") as h:
            if int(h["rf_data_index"][0, 0]) != int(start):
                raise ValueError(f"Output index mismatch in {out_path}")
            if h["rf_data"].dtype != np.dtype("complex64"):
                raise TypeError(f"{out_path} has dtype {h['rf_data'].dtype}, expected complex64")
            h["rf_data"][:, 0] = chunk.astype(np.complex64, copy=False)
    if pos != len(residual):
        raise ValueError("Internal write length mismatch")


def subtract_sparse_model(raw_by_channel: list[np.ndarray], abs_index0: int, sparse_path: Path, gain: float, mode: str) -> dict:
    with h5py.File(sparse_path, "r") as h:
        station = h["station"][()].decode("utf-8") if isinstance(h["station"][()], bytes) else str(h["station"][()])
        seeds = h["seeds"][()]
        sparse_channels = h["channels"][()]
        code_len = int(h["code_len"][()])
        coherent_codes = int(h["coherent_codes"][()])
        range_interp = int(h["range_interp"][()])
        sample_rate = float(h["sample_rate"][()])
        sparse_t0_sample = int(h["t0_sample"][()])
        support_count = h["support_count"][()]
        support_range_idx = h["support_range_idx"][()]
        support_voltage = h["support_voltage"][()]
        dc_voltage = h["dc_voltage"][()] if "dc_voltage" in h else None

    if range_interp != 1:
        raise ValueError(f"{sparse_path}: expected raw range_interp=1, got {range_interp}")
    if mode not in {"full", "dc"}:
        raise ValueError(f"Unknown subtraction mode {mode!r}")
    if len(sparse_channels) != len(raw_by_channel):
        raise ValueError(f"{station}: sparse channel count does not match raw channel count")

    codes = np.asarray([raw_code(int(seed), code_len) for seed in seeds], dtype=np.complex64)
    sample_idx = np.arange(code_len, dtype=np.int64)
    n_tx = codes.shape[0]
    min_len = min(len(z) for z in raw_by_channel)
    offset = int(round(float(sparse_t0_sample) - float(abs_index0)))
    model_power = np.zeros(len(raw_by_channel), dtype=np.float64)
    blocks_used = 0
    terms = 0

    for bi in range(support_voltage.shape[0]):
        n_support = int(support_count[bi])
        has_dc = dc_voltage is not None and bi < dc_voltage.shape[0]
        if n_support <= 0 and not has_dc:
            continue
        support = support_range_idx[bi, :n_support].astype(np.int64) if n_support > 0 else np.asarray([], dtype=np.int64)
        good = support >= 0
        support = support[good]
        if support.size == 0 and not has_dc:
            continue
        shifted = np.empty((len(support), n_tx, code_len), dtype=np.complex64)
        for si, raw_ri in enumerate(support):
            shifted[si] = codes[:, (sample_idx - int(raw_ri)) % code_len]

        used_block = False
        dc_coeff = None
        if mode == "dc":
            dc_coeff = np.mean(support_voltage[bi, :, :, :, :n_support], axis=0)

        for ci in range(coherent_codes):
            start = offset + (bi * coherent_codes + ci) * code_len
            stop = start + code_len
            if stop <= 0 or start >= min_len:
                continue
            seg0 = max(start, 0)
            seg1 = min(stop, min_len)
            loc0 = seg0 - start
            loc1 = seg1 - start
            shifted_part = shifted[:, :, loc0:loc1]
            for chi, raw in enumerate(raw_by_channel):
                if n_support == 0:
                    coeff = np.empty((0, n_tx), dtype=np.complex64)
                elif mode == "full":
                    coeff = support_voltage[bi, ci, chi, :, :n_support].T
                else:
                    coeff = dc_coeff[chi, :, :n_support].T
                model = np.sum(coeff[:, :, None] * shifted_part, axis=(0, 1), dtype=np.complex64)
                if has_dc:
                    if dc_voltage.ndim == 1:
                        model += dc_voltage[bi]
                    else:
                        model += dc_voltage[bi, ci, chi]
                    terms += 1
                raw[seg0:seg1] = (raw[seg0:seg1] - gain * model).astype(np.complex64)
                model_power[chi] += float(np.sum(np.abs(model) ** 2))
                terms += int(np.count_nonzero(coeff))
            used_block = True
        if used_block:
            blocks_used += 1

    return {
        "station": station,
        "sample_rate": sample_rate,
        "blocks_used": blocks_used,
        "terms": terms,
        "model_power_per_channel": model_power,
    }


def copy_input_tree(data_dir: Path, out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    shutil.copytree(data_dir, out_dir)


def process_station(data_dir: Path, out_dir: Path, sparse_dir: Path, station: str, gain: float, mode: str) -> dict:
    sparse_path = sparse_dir / f"trail_sparse_deconv_{station}.h5"
    if not sparse_path.exists():
        raise FileNotFoundError(f"Missing sparse product {sparse_path}")

    files_by_channel = []
    starts_by_channel = []
    raw_by_channel = []
    sample_rates = []
    for ch in (0, 1):
        files, starts, raw, sr = load_rf_channel(data_dir / station / f"ch{ch:03d}")
        files_by_channel.append(files)
        starts_by_channel.append(starts)
        raw_by_channel.append(raw)
        sample_rates.append(sr)
    if len(set(sample_rates)) != 1:
        raise ValueError(f"{station}: sample-rate mismatch between channels")
    if starts_by_channel[0][0] != starts_by_channel[1][0]:
        raise ValueError(f"{station}: channel start-index mismatch")

    meta = subtract_sparse_model(raw_by_channel, starts_by_channel[0][0], sparse_path, gain, mode)
    for files, starts, raw in zip(files_by_channel, starts_by_channel, raw_by_channel, strict=True):
        write_rf_channel(files, starts, raw, data_dir, out_dir)
    return meta


def write_manifest(out_dir: Path, args: argparse.Namespace, station_meta: list[dict]) -> None:
    manifest = out_dir / "sparse_residual_manifest.txt"
    lines = [
        "Sparse residual Digital RF product",
        f"input_data_dir: {args.data_dir}",
        f"sparse_product_dir: {args.sparse_dir}",
        f"gain: {args.gain}",
        f"mode: {args.mode}",
        "dtype: complex64",
        "residual: raw voltage minus reconstructed sparse model",
        "",
    ]
    for meta in station_meta:
        lines.append(
            f"{meta['station']}: blocks_used={meta['blocks_used']} terms={meta['terms']} "
            f"model_power_per_channel={np.asarray(meta['model_power_per_channel']).tolist()}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sparse-dir", type=Path, default=DEFAULT_SPARSE_DIR)
    parser.add_argument("--stations", nargs="+", default=STATION_ORDER)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--mode", choices=("full", "dc"), default="full")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    copy_input_tree(args.data_dir, args.out_dir, args.overwrite)
    station_meta = []
    for station in args.stations:
        print(f"{station}: subtracting sparse model", flush=True)
        meta = process_station(args.data_dir, args.out_dir, args.sparse_dir, station, args.gain, args.mode)
        station_meta.append(meta)
        print(f"{station}: blocks={meta['blocks_used']} terms={meta['terms']}", flush=True)
    write_manifest(args.out_dir, args, station_meta)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
