#!/usr/bin/env python3
"""Generate raw-voltage spectrograms for RFI inspection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
EVENT_T0 = 1780164380.0
DEFAULT_DATA_DIR = ROOT / "2026_05_30_bolide_residual"
DEFAULT_OUT_DIR = ROOT / "reports" / "figures" / "raw_voltage_spectrograms"
ISO_HOUR = "2026-05-30T18-00-00"


def rf_timestamp(path: Path) -> float:
    match = re.search(r"rf@([0-9]+(?:\.[0-9]+)?)\.h5$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse RF timestamp from {path}")
    return float(match.group(1))


def station_channels(data_dir: Path, stations: list[str] | None, channels: list[int] | None) -> list[tuple[str, int]]:
    out = []
    station_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir()) if stations is None else [data_dir / s for s in stations]
    for station_dir in station_dirs:
        if not station_dir.is_dir():
            continue
        ch_dirs = sorted(station_dir.glob("ch[0-9][0-9][0-9]"))
        for ch_dir in ch_dirs:
            ch = int(ch_dir.name[2:])
            if channels is not None and ch not in channels:
                continue
            if (ch_dir / ISO_HOUR).is_dir():
                out.append((station_dir.name, ch))
    return out


def read_rf_span(data_dir: Path, station: str, channel: int, start_unix: float, stop_unix: float) -> tuple[np.ndarray, float]:
    files = sorted((data_dir / station / f"ch{channel:03d}" / ISO_HOUR).glob("rf@*.h5"), key=rf_timestamp)
    chunks = []
    sample_rate = None
    for fn in files:
        file_start = rf_timestamp(fn)
        with h5py.File(fn, "r") as h:
            ds = h["rf_data"]
            sr = float(ds.attrs["sample_rate_numerator"]) / float(ds.attrs["sample_rate_denominator"])
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample-rate mismatch in {fn}: {sr} != {sample_rate}")
            file_stop = file_start + ds.shape[0] / sr
            if file_stop <= start_unix or file_start >= stop_unix:
                continue
            i0 = max(0, int(np.floor((start_unix - file_start) * sr)))
            i1 = min(ds.shape[0], int(np.ceil((stop_unix - file_start) * sr)))
            chunks.append(ds[i0:i1, 0].astype(np.complex64))
    if sample_rate is None or not chunks:
        raise FileNotFoundError(f"No RF samples for {station} ch{channel:03d} in requested interval")
    z = np.concatenate(chunks)
    target_n = int(round((stop_unix - start_unix) * sample_rate))
    if len(z) > target_n:
        z = z[:target_n]
    return z, sample_rate


def power_db_limits(sxx_db: np.ndarray, dynamic_range_db: float) -> tuple[float, float]:
    finite = sxx_db[np.isfinite(sxx_db)]
    vmax = float(np.nanpercentile(finite, 99.7)) if finite.size else 0.0
    return vmax - dynamic_range_db, vmax


def estimate_dc_voltage(z: np.ndarray, method: str) -> complex:
    if method == "none":
        return 0.0 + 0.0j
    if method == "mean":
        return complex(float(np.mean(z.real)), float(np.mean(z.imag)))
    if method == "median":
        return complex(float(np.median(z.real)), float(np.median(z.imag)))
    raise ValueError(f"Unknown DC subtraction method {method!r}")


def plot_one(
    z: np.ndarray,
    sample_rate: float,
    station: str,
    channel: int,
    start_unix: float,
    stop_unix: float,
    out: Path,
    *,
    nperseg: int,
    noverlap: int,
    zoom_hz: float,
    dynamic_range_db: float,
    dc_method: str,
) -> dict:
    dc_voltage = estimate_dc_voltage(z, dc_method)
    z = z - np.complex64(dc_voltage)
    freqs, times, sxx = spectrogram(
        z,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        scaling="spectrum",
        mode="psd",
    )
    order = np.argsort(np.fft.fftshift(freqs))
    freqs = np.fft.fftshift(freqs)[order]
    sxx = np.fft.fftshift(sxx, axes=0)[order, :]
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-30))
    t_rel = times + start_unix - EVENT_T0
    vmin, vmax = power_db_limits(sxx_db, dynamic_range_db)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.2), sharex=True, constrained_layout=True)
    for ax, limit_hz, title in [
        (axes[0], sample_rate / 2.0, "Full 100 kHz complex spectrum"),
        (axes[1], zoom_hz, f"+/- {zoom_hz / 1000.0:g} kHz zoom"),
    ]:
        keep = np.abs(freqs) <= limit_hz
        pcm = ax.pcolormesh(t_rel, freqs[keep] / 1000.0, sxx_db[keep, :], shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(title)
        ax.grid(color="white", alpha=0.12, lw=0.4)
        fig.colorbar(pcm, ax=ax, pad=0.012, label="Power (dB)")
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    fig.suptitle(
        f"{station} ch{channel:03d}, 18:06:15-18:06:25 UTC, "
        f"{sample_rate / nperseg:.0f} Hz bins, DC {dc_method}",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    median_spec = np.nanmedian(sxx_db, axis=1)
    floor = float(np.nanmedian(median_spec))
    peak_idx = np.argsort(median_spec)[-20:][::-1]
    return {
        "station": station,
        "channel": channel,
        "sample_rate": sample_rate,
        "nperseg": nperseg,
        "seconds": stop_unix - start_unix,
        "dc_real": float(np.real(dc_voltage)),
        "dc_imag": float(np.imag(dc_voltage)),
        "dc_abs": float(abs(dc_voltage)),
        "median_floor_db": floor,
        "top_lines": [(float(freqs[i]), float(median_spec[i] - floor)) for i in peak_idx],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stations", nargs="+")
    p.add_argument("--channels", nargs="+", type=int)
    p.add_argument("--start-unix", type=float, default=1780164375.0)
    p.add_argument("--stop-unix", type=float, default=1780164385.0)
    p.add_argument("--resolution-hz", type=float, default=100.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--zoom-hz", type=float, default=10_000.0)
    p.add_argument("--dynamic-range-db", type=float, default=70.0)
    p.add_argument("--dc-subtract", choices=("none", "mean", "median"), default="mean")
    args = p.parse_args()

    pairs = station_channels(args.data_dir, args.stations, args.channels)
    if not pairs:
        raise FileNotFoundError(f"No station/channel RF data found under {args.data_dir}")
    summaries = []
    for station, channel in pairs:
        z, sample_rate = read_rf_span(args.data_dir, station, channel, args.start_unix, args.stop_unix)
        nperseg = int(round(sample_rate / args.resolution_hz))
        noverlap = int(round(args.overlap * nperseg))
        dc_suffix = f"dc{args.dc_subtract}"
        out = args.out_dir / f"{args.data_dir.name}_{station}_ch{channel:03d}_180615_180625_spectrogram_{dc_suffix}.png"
        summary = plot_one(
            z,
            sample_rate,
            station,
            channel,
            args.start_unix,
            args.stop_unix,
            out,
            nperseg=nperseg,
            noverlap=noverlap,
            zoom_hz=args.zoom_hz,
            dynamic_range_db=args.dynamic_range_db,
            dc_method=args.dc_subtract,
        )
        summaries.append(summary)
        print(out, flush=True)

    summary_path = args.out_dir / f"{args.data_dir.name}_180615_180625_spectrogram_{dc_suffix}_rfi_lines.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("# Median-spectrum line candidates relative to median spectral floor\n")
        fh.write("# DC subtraction uses one complex voltage estimate over the requested interval\n")
        fh.write("# station channel frequency_hz excess_db dc_real dc_imag dc_abs\n")
        for summary in summaries:
            for freq_hz, excess_db in summary["top_lines"]:
                fh.write(
                    f"{summary['station']} ch{summary['channel']:03d} {freq_hz:.1f} {excess_db:.2f} "
                    f"{summary['dc_real']:.8g} {summary['dc_imag']:.8g} {summary['dc_abs']:.8g}\n"
                )
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
