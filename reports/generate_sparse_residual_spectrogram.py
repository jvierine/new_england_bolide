#!/usr/bin/env python3
"""Generate sparse-model-subtracted residual-voltage spectrograms."""

from __future__ import annotations

import argparse
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

from reports.generate_sparse_residual_noise_power import (  # noqa: E402
    DATA_DIR,
    SPARSE_PRODUCT_DIR,
    STATION_LABELS,
    STATIONS,
    load_station_raw,
    subtract_sparse_model,
)

OUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_spectrogram_1s_100khz.h5"
OUT_FIG = ROOT / "reports" / "figures" / "sparse_residual_spectrogram_1s_100khz.png"


def station_channels(sparse_product: Path, fallback_channels: list[int]) -> np.ndarray:
    if sparse_product.exists():
        with h5py.File(sparse_product, "r") as h:
            return np.asarray(h["channels"][()], dtype=np.int32)
    return np.asarray(fallback_channels, dtype=np.int32)


def one_second_spectrogram(
    raw: list[np.ndarray],
    sample_rate: float,
    abs_index0: int,
    fft_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(z) for z in raw)
    samples_per_bin = int(round(sample_rate))
    n_bins = n // samples_per_bin
    if n_bins <= 0:
        raise ValueError("not enough samples for a 1 s spectrogram")
    n_used = n_bins * samples_per_bin
    n_segments = samples_per_bin // fft_len
    if n_segments <= 0:
        raise ValueError(f"FFT length {fft_len} exceeds 1 s sample count {samples_per_bin}")
    keep = n_segments * fft_len
    window = np.hanning(fft_len).astype(np.float32)
    window_norm = float(np.sum(window**2))
    spec = np.empty((n_bins, fft_len), dtype=np.float32)
    for ti in range(n_bins):
        p = np.zeros(fft_len, dtype=np.float64)
        i0 = ti * samples_per_bin
        for z in raw:
            block = z[i0 : i0 + keep].reshape(n_segments, fft_len)
            block = block - np.mean(block, axis=1, keepdims=True)
            fft = np.fft.fftshift(np.fft.fft(block * window[None, :], axis=1), axes=1)
            p += np.mean(np.abs(fft) ** 2, axis=0) / window_norm
        spec[ti] = (p / float(len(raw))).astype(np.float32)
    time_unix = (abs_index0 + (np.arange(n_bins, dtype=np.float64) + 0.5) * samples_per_bin) / sample_rate
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / sample_rate)).astype(np.float32)
    return time_unix, frequency_hz, spec


def compute(args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_h5.with_suffix(args.output_h5.suffix + ".tmp")
    metadata = {
        "source": "reports/generate_sparse_residual_spectrogram.py",
        "iso_hour": args.iso_hour,
        "data_dir": str(args.data_dir),
        "sparse_product_dir": str(args.sparse_product_dir),
        "stations": {},
        "time_resolution_s": 1.0,
        "fft_len": int(args.fft_len),
        "spectrum_definition": "channel-averaged mean Hann-windowed periodogram of sparse-model-subtracted raw 100 kHz voltage; per-FFT-segment complex mean removed",
    }
    station_payload = {}
    frequency_hz = None
    for station in args.stations:
        sparse_product = args.sparse_product_dir / f"trail_sparse_deconv_{station}.h5"
        channels = station_channels(sparse_product, args.channels)
        print(f"{station}: reading raw channels {channels.tolist()}", flush=True)
        raw, sample_rate, abs_index0 = load_station_raw(args.data_dir, station, channels, args.iso_hour)
        print(f"{station}: subtracting sparse model", flush=True)
        sparse_meta = subtract_sparse_model(raw, sample_rate, abs_index0, sparse_product)
        print(f"{station}: estimating spectrogram", flush=True)
        time_unix, f_hz, power = one_second_spectrogram(raw, sample_rate, abs_index0, args.fft_len)
        if frequency_hz is None:
            frequency_hz = f_hz
        elif frequency_hz.shape != f_hz.shape or np.max(np.abs(frequency_hz - f_hz)) > 1e-6:
            raise ValueError(f"{station}: frequency grid mismatch")
        station_payload[station] = {"time_unix": time_unix, "power": power}
        metadata["stations"][station] = {
            "channels": channels.tolist(),
            "sample_rate_hz": float(sample_rate),
            "abs_index0": int(abs_index0),
            "n_seconds": int(len(time_unix)),
            **sparse_meta,
        }
        print(f"{station}: {len(time_unix)} one-second spectra", flush=True)

    with h5py.File(tmp, "w") as h:
        h.attrs["metadata_json"] = json.dumps(metadata, indent=2)
        h.attrs["source"] = metadata["source"]
        h.attrs["spectrum_definition"] = metadata["spectrum_definition"]
        h.attrs["time_resolution_s"] = metadata["time_resolution_s"]
        h.attrs["fft_len"] = metadata["fft_len"]
        h.attrs["iso_hour"] = metadata["iso_hour"]
        h.attrs["data_dir"] = metadata["data_dir"]
        h.attrs["sparse_product_dir"] = metadata["sparse_product_dir"]
        h.create_dataset("frequency_hz", data=np.asarray(frequency_hz, dtype=np.float32))
        h.create_dataset("stations", data=np.asarray(list(station_payload), dtype="S"))
        for station, payload in station_payload.items():
            group = h.create_group(station)
            group.create_dataset("time_unix", data=payload["time_unix"])
            group.create_dataset("power_spectrum", data=payload["power"], compression="gzip", shuffle=True)
            for key, value in metadata["stations"][station].items():
                if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
                    group.attrs[key] = value
                else:
                    group.attrs[key] = json.dumps(value)
    tmp.replace(args.output_h5)
    print(args.output_h5, flush=True)


def load_h5(path: Path) -> tuple[dict, dict, np.ndarray]:
    with h5py.File(path, "r") as h:
        metadata = json.loads(str(h.attrs["metadata_json"]))
        frequency_hz = h["frequency_hz"][()]
        data = {}
        for station in metadata["stations"]:
            group = h[station]
            data[station] = {
                "time_unix": group["time_unix"][()],
                "power_spectrum": group["power_spectrum"][()],
            }
    return data, metadata, frequency_hz


def bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.r_[first, mids, last]


def plot(args: argparse.Namespace) -> None:
    data, metadata, frequency_hz = load_h5(args.output_h5)
    vals = np.concatenate(
        [10.0 * np.log10(np.maximum(d["power_spectrum"].ravel(), 1e-30)) for d in data.values()]
    )
    vmin = float(np.nanpercentile(vals, 5.0))
    vmax = float(np.nanpercentile(vals, 99.5))
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})
    stations = [s for s in args.stations if s in data]
    fig, axes = plt.subplots(len(stations), 1, figsize=(11.5, 9.0), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, station in zip(axes, stations, strict=True):
        t = data[station]["time_unix"]
        p_db = 10.0 * np.log10(np.maximum(data[station]["power_spectrum"], 1e-30))
        mesh = ax.pcolormesh(
            bin_edges(mdates.date2num([datetime.fromtimestamp(float(x), UTC) for x in t])),
            bin_edges(frequency_hz / 1000.0),
            p_db.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{STATION_LABELS.get(station, station)}\n$ f $ (kHz)")
        ax.grid(False)
    axes[-1].set_xlabel("UTC on 2026-05-30")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.suptitle("Sparse-model-subtracted 100 kHz residual-voltage spectra", fontsize=15)
    fig.colorbar(mesh, ax=axes, label="Power spectrum (dB, arbitrary)", shrink=0.93)
    axes[0].text(
        0.01,
        0.96,
        f"1 s bins, FFT length {metadata['fft_len']}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
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
    parser.add_argument("--fft-len", type=int, default=2048)
    parser.add_argument("--output-h5", type=Path, default=OUT_H5)
    parser.add_argument("--output-fig", type=Path, default=OUT_FIG)
    args = parser.parse_args()
    if args.mode in {"compute", "both"}:
        compute(args)
    if args.mode in {"plot", "both"}:
        plot(args)


if __name__ == "__main__":
    main()
