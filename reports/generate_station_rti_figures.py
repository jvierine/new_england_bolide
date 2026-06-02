#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MERGED_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
FIG_DIR = ROOT / "reports" / "figures"
STATION_RANGES_KM = [
    ("beverly", 150.0, 300.0),
    ("clinton", 200.0, 300.0),
    ("pomfret", 300.0, 450.0),
    ("siena", 400.0, 500.0),
    ("amherstnh", 200.0, 350.0),
]
SNR_THRESHOLD_DB = 5.0
DOPPLER_VMIN_HZ = -3000.0
DOPPLER_VMAX_HZ = 0.0
PLOT_START_UNIX = 1780164378.0


def unix_to_mpl(t: np.ndarray) -> np.ndarray:
    ns = np.asarray(np.asarray(t, dtype=float) * 1e9, dtype="datetime64[ns]")
    return mdates.date2num(ns.astype("datetime64[us]").astype(object))


def load_station(station: str, rmin_km: float, rmax_km: float) -> dict:
    path = MERGED_DIR / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(path, "r") as h:
        p = h["P"][()]
        n = h["N"][()]
        doppler = h["D"][()]
        t = h["t_sample"][()] / float(h["sample_rate"][()])
        if "range_km" in h:
            ranges = h["range_km"][()]
        else:
            ranges = h["range_idx"][()] * 3.0
    range_keep = (ranges >= rmin_km) & (ranges <= rmax_km)
    time_keep = t >= PLOT_START_UNIX
    snr_db = 10.0 * np.log10(
        np.maximum(p[time_keep][:, range_keep] / np.maximum(n[time_keep][:, range_keep], 1e-12), 1e-12)
    )
    return {
        "station": station,
        "t": t[time_keep],
        "range_km": ranges[range_keep],
        "snr_db": snr_db,
        "doppler_hz": doppler[time_keep][:, range_keep],
        "rmin_km": rmin_km,
        "rmax_km": rmax_km,
    }


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.titlesize": 17,
            "axes.linewidth": 1.0,
        }
    )


def plot_snr(datasets: list[dict]) -> Path:
    all_snr = np.concatenate([d["snr_db"].ravel() for d in datasets])
    vmin = max(-5.0, float(np.nanpercentile(all_snr, 2.0)))
    vmax = min(28.0, float(np.nanpercentile(all_snr, 99.7)))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    mesh = None
    for ax, data in zip(axes, datasets):
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            data["snr_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{data['station']}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle("Range-time intensity by station")
    fig.colorbar(mesh, ax=axes, label="SNR (dB)", shrink=0.93)
    out = FIG_DIR / "station_range_time_snr.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_doppler(datasets: list[dict]) -> Path:
    threshold_masked = []
    for data in datasets:
        threshold_masked.append(np.where(data["snr_db"] >= SNR_THRESHOLD_DB, data["doppler_hz"], np.nan))
    cmap = plt.cm.turbo.copy()
    cmap.set_bad("0.78")

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    mesh = None
    for ax, data, doppler in zip(axes, datasets, threshold_masked):
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            doppler.T,
            shading="auto",
            cmap=cmap,
            vmin=DOPPLER_VMIN_HZ,
            vmax=DOPPLER_VMAX_HZ,
        )
        ax.set_ylabel(f"{data['station']}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle(f"Peak Doppler by station, masked below {SNR_THRESHOLD_DB:g} dB SNR")
    fig.colorbar(mesh, ax=axes, label="Peak Doppler (Hz)", shrink=0.93)
    out = FIG_DIR / "station_range_time_doppler.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    set_plot_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [load_station(station, rmin, rmax) for station, rmin, rmax in STATION_RANGES_KM]
    print(plot_snr(datasets))
    print(plot_doppler(datasets))


if __name__ == "__main__":
    main()
