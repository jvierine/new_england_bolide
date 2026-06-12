#!/usr/bin/env python3
"""Estimate 1 s residual 100 kHz voltage power after sparse-model subtraction."""

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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from range_doppler_mf_bolide import bpsk  # noqa: E402
from reports.generate_head_microdoppler import read_channel_files, rf_file_for_time  # noqa: E402

DATA_DIR = ROOT / "2026_05_30_bolide"
SPARSE_PRODUCT_DIR = ROOT / "products" / "trail_echo_sparse_deconv_full_3km_50code_rdmf_avx"
OUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_noise_power_1s.h5"
OUT_CSV = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_noise_power_1s.csv"
OUT_FIG = ROOT / "reports" / "figures" / "sparse_residual_noise_power_1s.png"
STATIONS = ("beverly", "clinton", "pomfret", "siena", "amherstnh")
STATION_LABELS = {
    "amherstnh": "Amherst",
    "beverly": "Beverly",
    "clinton": "Clinton",
    "pomfret": "Pomfret",
    "siena": "Siena",
}


def raw_codes(seeds: np.ndarray, code_len: int) -> np.ndarray:
    codes = []
    for seed in seeds:
        code = bpsk(int(seed), int(code_len))
        code = code / np.max(np.abs(code))
        codes.append(code.astype(np.complex64))
    return np.asarray(codes, dtype=np.complex64)


def load_station_raw(data_dir: Path, station: str, channels: np.ndarray, iso_hour: str) -> tuple[list[np.ndarray], float, int]:
    raw = []
    sample_rate = None
    abs_index0 = None
    for channel in channels:
        z, sr, idx0 = read_channel_files(
            rf_file_for_time(data_dir, station, int(channel), iso_hour),
            start_second=0.0,
            seconds=None,
            dc_subtract="none",
        )
        raw.append(z.astype(np.complex64, copy=False))
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
        if sample_rate != sr or abs_index0 != idx0:
            raise ValueError(f"{station}: channel {channel} time base mismatch")
    n = min(len(z) for z in raw)
    return [z[:n] for z in raw], float(sample_rate), int(abs_index0)


def subtract_sparse_model(raw: list[np.ndarray], sample_rate: float, abs_index0: int, sparse_product: Path) -> dict:
    if not sparse_product.exists():
        return {"sparse_product": str(sparse_product), "sparse_product_found": False}

    with h5py.File(sparse_product, "r") as h:
        station = h["station"][()].decode("utf-8") if isinstance(h["station"][()], bytes) else str(h["station"][()])
        seeds = h["seeds"][()]
        sparse_channels = h["channels"][()]
        code_len = int(h["raw_code_len"][()] if "raw_code_len" in h else h["code_len"][()])
        coherent_codes = int(h["coherent_codes"][()])
        sparse_sample_rate = float(h["sample_rate"][()])
        sparse_t0_sample = int(h["t0_sample"][()])
        support_count = h["support_count"][()]
        support_range_idx = h["support_range_idx"][()]
        support_voltage = h["support_voltage"][()]
        dc_voltage = h["dc_voltage"][()] if "dc_voltage" in h else None

    if abs(sparse_sample_rate - sample_rate) > 1e-6:
        raise ValueError(f"{station}: sparse sample rate {sparse_sample_rate} != raw sample rate {sample_rate}")
    if len(sparse_channels) != len(raw):
        raise ValueError(f"{station}: sparse channel count {len(sparse_channels)} != raw channel count {len(raw)}")

    codes = raw_codes(seeds, code_len)
    n_tx = codes.shape[0]
    min_len = min(len(z) for z in raw)
    offset = int(round(float(sparse_t0_sample) - float(abs_index0)))
    blocks_used = 0
    terms = 0
    model_power = np.zeros(len(raw), dtype=np.float64)

    for bi in range(support_voltage.shape[0]):
        n_support = int(support_count[bi])
        has_dc = dc_voltage is not None and bi < dc_voltage.shape[0]
        if n_support <= 0 and not has_dc:
            continue
        used_block = False
        for ci in range(coherent_codes):
            start = offset + (bi * coherent_codes + ci) * code_len
            stop = start + code_len
            if stop <= 0 or start >= min_len:
                continue
            seg0 = max(start, 0)
            seg1 = min(stop, min_len)
            loc = np.arange(seg0 - start, seg1 - start, dtype=np.int64)
            for chi, z in enumerate(raw):
                model = np.zeros(seg1 - seg0, dtype=np.complex64)
                if has_dc:
                    if np.ndim(dc_voltage) == 1:
                        model += dc_voltage[bi]
                    else:
                        model += dc_voltage[bi, ci, chi]
                    terms += 1
                for txi in range(n_tx):
                    code = codes[txi]
                    for si in range(n_support):
                        raw_ri = int(support_range_idx[bi, si])
                        if raw_ri < 0:
                            continue
                        coeff = support_voltage[bi, ci, chi, txi, si]
                        if coeff == 0:
                            continue
                        model += coeff * code[(loc - raw_ri) % code_len]
                        terms += 1
                z[seg0:seg1] = (z[seg0:seg1] - model).astype(np.complex64)
                model_power[chi] += float(np.sum(np.abs(model) ** 2))
            used_block = True
        if used_block:
            blocks_used += 1

    return {
        "sparse_product": str(sparse_product),
        "sparse_product_found": True,
        "blocks_used": int(blocks_used),
        "terms": int(terms),
        "model_power_per_channel": model_power.tolist(),
    }


def one_second_powers(raw: list[np.ndarray], sample_rate: float, abs_index0: int) -> dict[str, np.ndarray]:
    n = min(len(z) for z in raw)
    samples_per_bin = int(round(sample_rate))
    n_bins = n // samples_per_bin
    if n_bins <= 0:
        raise ValueError("not enough samples for a 1 s power estimate")
    n_used = n_bins * samples_per_bin
    arr = np.stack([z[:n_used].reshape(n_bins, samples_per_bin) for z in raw], axis=1)
    raw_power = np.mean(np.abs(arr) ** 2, axis=(1, 2), dtype=np.float64)
    dc = np.mean(arr, axis=2, keepdims=True, dtype=np.complex128)
    residual = arr - dc
    dc_subtracted_power = np.mean(np.abs(residual) ** 2, axis=(1, 2), dtype=np.float64)
    dc_power = np.mean(np.abs(dc[:, :, 0]) ** 2, axis=1, dtype=np.float64)
    time_unix = (abs_index0 + (np.arange(n_bins, dtype=np.float64) + 0.5) * samples_per_bin) / sample_rate
    return {
        "time_unix": time_unix,
        "raw_power": raw_power,
        "dc_subtracted_power": dc_subtracted_power,
        "dc_power": dc_power,
    }


def compute(args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    station_payload = {}
    rows = []
    metadata = {
        "source": "reports/generate_sparse_residual_noise_power.py",
        "iso_hour": args.iso_hour,
        "data_dir": str(args.data_dir),
        "sparse_product_dir": str(args.sparse_product_dir),
        "stations": {},
        "power_definition": "mean channel-averaged |z - mean_1s(z)|^2 after subtracting the sparse CS voltage model",
        "time_resolution_s": 1.0,
    }

    for station in args.stations:
        sparse_product = args.sparse_product_dir / f"trail_sparse_deconv_{station}.h5"
        if sparse_product.exists():
            with h5py.File(sparse_product, "r") as h:
                channels = np.asarray(h["channels"][()], dtype=np.int32)
        else:
            channels = np.asarray(args.channels, dtype=np.int32)
        print(f"{station}: reading raw channels {channels.tolist()}", flush=True)
        raw, sample_rate, abs_index0 = load_station_raw(args.data_dir, station, channels, args.iso_hour)
        print(f"{station}: subtracting sparse model", flush=True)
        sparse_meta = subtract_sparse_model(raw, sample_rate, abs_index0, sparse_product)
        powers = one_second_powers(raw, sample_rate, abs_index0)
        station_payload[station] = powers
        metadata["stations"][station] = {
            "channels": channels.tolist(),
            "sample_rate_hz": sample_rate,
            "abs_index0": abs_index0,
            "n_seconds": int(len(powers["time_unix"])),
            **sparse_meta,
        }
        for i, t in enumerate(powers["time_unix"]):
            rows.append(
                {
                    "station": station,
                    "time_unix": f"{t:.6f}",
                    "time_utc": datetime.fromtimestamp(float(t), UTC).isoformat(timespec="milliseconds"),
                    "raw_power": f"{powers['raw_power'][i]:.9e}",
                    "dc_subtracted_power": f"{powers['dc_subtracted_power'][i]:.9e}",
                    "dc_power": f"{powers['dc_power'][i]:.9e}",
                }
            )
        print(f"{station}: {len(powers['time_unix'])} one-second bins", flush=True)

    tmp_h5 = args.output_h5.with_suffix(args.output_h5.suffix + ".tmp")
    with h5py.File(tmp_h5, "w") as h:
        h.attrs["metadata_json"] = json.dumps(metadata, indent=2)
        h.attrs["power_definition"] = metadata["power_definition"]
        h.attrs["time_resolution_s"] = metadata["time_resolution_s"]
        h.attrs["source"] = metadata["source"]
        h.attrs["iso_hour"] = metadata["iso_hour"]
        h.attrs["data_dir"] = metadata["data_dir"]
        h.attrs["sparse_product_dir"] = metadata["sparse_product_dir"]
        h.create_dataset("stations", data=np.asarray(list(station_payload), dtype="S"))
        for station, powers in station_payload.items():
            group = h.create_group(station)
            for key, value in powers.items():
                group.create_dataset(key, data=np.asarray(value))
            for key, value in metadata["stations"][station].items():
                if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
                    group.attrs[key] = value
                else:
                    group.attrs[key] = json.dumps(value)
    tmp_h5.replace(args.output_h5)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "time_unix", "time_utc", "raw_power", "dc_subtracted_power", "dc_power"])
        writer.writeheader()
        writer.writerows(rows)

    print(args.output_h5, flush=True)
    print(args.output_csv, flush=True)


def load_h5(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as h:
        metadata = json.loads(str(h.attrs["metadata_json"]))
        data = {}
        for station in metadata["stations"]:
            group = h[station]
            data[station] = {
                "time_unix": group["time_unix"][()],
                "raw_power": group["raw_power"][()],
                "dc_subtracted_power": group["dc_subtracted_power"][()],
                "dc_power": group["dc_power"][()],
            }
    return data, metadata


def plot(args: argparse.Namespace) -> None:
    data, metadata = load_h5(args.output_h5)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, ax = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    for station in args.stations:
        if station not in data:
            continue
        t = data[station]["time_unix"]
        p = data[station]["dc_subtracted_power"]
        ax.plot(
            mdates.date2num([datetime.fromtimestamp(float(x), UTC) for x in t]),
            10.0 * np.log10(np.maximum(p, 1e-30)),
            lw=1.2,
            label=STATION_LABELS.get(station, station),
        )
    ax.set_xlabel("UTC on 2026-05-30")
    ax.set_ylabel("1 s residual 100 kHz power (dB, arbitrary)")
    ax.set_title("Sparse-model-subtracted 100 kHz residual voltage power", fontsize=15, pad=10)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.legend(ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.text(
        0.0,
        -0.18,
        metadata.get("power_definition", ""),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_fig, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output_fig, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--stations", nargs="+", default=list(STATIONS))
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sparse-product-dir", type=Path, default=SPARSE_PRODUCT_DIR)
    parser.add_argument("--output-h5", type=Path, default=OUT_H5)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--output-fig", type=Path, default=OUT_FIG)
    args = parser.parse_args()

    if args.mode in {"compute", "both"}:
        compute(args)
    if args.mode in {"plot", "both"}:
        plot(args)


if __name__ == "__main__":
    main()
