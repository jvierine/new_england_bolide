#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_aod_chunks import unix_to_mpl


DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def read_merged(fname):
    with h5py.File(fname, "r") as h:
        sr = float(h["sample_rate"][()])
        return {
            "station": h["station"][()].decode(),
            "t": h["t_sample"][()] / sr,
            "range_km": h["range_km"][()],
            "P": h["P"][()],
            "N": h["N"][()],
            "D": h["D"][()],
        }


def clip_time(data, t0, t1):
    keep = (data["t"] >= t0) & (data["t"] <= t1)
    return {
        "station": data["station"],
        "t": data["t"][keep],
        "range_km": data["range_km"],
        "P": data["P"][keep, :],
        "N": data["N"][keep, :],
        "D": data["D"][keep, :],
    }


def draw(datasets, out_png, title, snr_vmax=None):
    fig, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(13.0, max(2.0 * len(datasets), 8.0)),
        sharex="col",
        constrained_layout=True,
    )
    if len(datasets) == 1:
        axes = axes[None, :]

    snr_all = []
    dop_all = []
    for d in datasets:
        snr = d["P"] / np.maximum(d["N"], 1e-12)
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
        snr_all.append(snr_db.ravel())
        dop_all.append(np.abs(d["D"]).ravel())
    snr_all = np.concatenate(snr_all)
    dop_all = np.concatenate(dop_all)
    snr_vmin = max(-5.0, float(np.nanpercentile(snr_all, 2)))
    snr_vmax = snr_vmax if snr_vmax is not None else min(45.0, float(np.nanpercentile(snr_all, 99.7)))
    dop_vmax = max(1000.0, float(np.nanpercentile(dop_all, 98)))

    snr_mesh = dop_mesh = None
    for row, d in enumerate(datasets):
        tplot = unix_to_mpl(d["t"])
        r = d["range_km"]
        snr = d["P"] / np.maximum(d["N"], 1e-12)
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
        ax = axes[row, 0]
        snr_mesh = ax.pcolormesh(tplot, r, snr_db.T, shading="auto", cmap="inferno", vmin=snr_vmin, vmax=snr_vmax)
        ax.set_ylabel(f"{d['station']}\nrange (km)")
        if row == 0:
            ax.set_title("Matched-filter RTI")

        ax = axes[row, 1]
        dop_mesh = ax.pcolormesh(tplot, r, d["D"].T, shading="auto", cmap="seismic", vmin=-dop_vmax, vmax=dop_vmax)
        if row == 0:
            ax.set_title("Peak Doppler")

        for ax in axes[row, :]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True, color="0.9", linewidth=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel("UTC")
    fig.suptitle(title, fontsize=14)
    fig.colorbar(snr_mesh, ax=axes[:, 0], label="SNR (dB)", shrink=0.9)
    fig.colorbar(dop_mesh, ax=axes[:, 1], label="Hz", shrink=0.9)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot merged simple RDMF products.")
    p.add_argument("merged_dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    p.add_argument("--head-seconds", type=float, default=2.0)
    args = p.parse_args()

    datasets = []
    for station in args.stations:
        fname = Path(args.merged_dir) / f"rdmf_peak_{station}_merged.h5"
        if fname.exists():
            datasets.append(read_merged(fname))
    if not datasets:
        raise SystemExit("No merged RDMF files found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = min(float(np.min(d["t"])) for d in datasets)
    head = [clip_time(d, t0, t0 + args.head_seconds) for d in datasets]
    draw(head, out_dir / "rdmf_head_180620_180622.png", "Simple RDMF head window, 18:06:20-18:06:22 UTC")
    draw(datasets, out_dir / "rdmf_full_180620_180800.png", "Simple RDMF full duration, 18:06:20-18:08:00 UTC")
    print(out_dir / "rdmf_head_180620_180622.png")
    print(out_dir / "rdmf_full_180620_180800.png")


if __name__ == "__main__":
    main()
