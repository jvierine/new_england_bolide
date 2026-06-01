#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_aod_chunks import load_station, unix_to_mpl


DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def clip_time(data, t0, t1):
    keep = (data["t"] >= t0) & (data["t"] <= t1)
    out = {"station": data["station"], "range_km": data["range_km"], "t": data["t"][keep]}
    for key in ("snr", "doppler", "az", "el", "score"):
        out[key] = data[key][keep, :]
    return out


def draw_panel_grid(datasets, out_png, title, snr_threshold_db):
    threshold_linear = 10.0 ** (snr_threshold_db / 10.0)
    n = len(datasets)
    fig, axes = plt.subplots(
        n,
        4,
        figsize=(17.5, max(2.0 * n, 8.0)),
        sharex="col",
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[None, :]

    all_snr_db = [
        10.0 * np.log10(np.maximum(d["snr"], 1e-12)).ravel()
        for d in datasets
        if d["snr"].size
    ]
    if all_snr_db:
        all_snr_db = np.concatenate(all_snr_db)
        snr_vmin = max(-5.0, float(np.nanpercentile(all_snr_db, 5)))
        snr_vmax = min(45.0, float(np.nanpercentile(all_snr_db, 99.5)))
    else:
        snr_vmin, snr_vmax = -5.0, 35.0

    all_dop = []
    for d in datasets:
        good = d["snr"] >= threshold_linear
        vals = np.abs(np.where(good, d["doppler"], np.nan))
        if np.any(np.isfinite(vals)):
            all_dop.append(vals[np.isfinite(vals)])
    dop_vmax = max(1000.0, float(np.nanpercentile(np.concatenate(all_dop), 98))) if all_dop else 1000.0

    snr_mesh = dop_mesh = None
    az_sc = el_sc = None
    for row, d in enumerate(datasets):
        tplot = unix_to_mpl(d["t"])
        r = d["range_km"]
        station = d["station"]
        snr_db = 10.0 * np.log10(np.maximum(d["snr"], 1e-12))
        good = d["snr"] >= threshold_linear

        ax = axes[row, 0]
        if d["snr"].size:
            snr_mesh = ax.pcolormesh(
                tplot, r, snr_db.T, shading="auto", cmap="inferno", vmin=snr_vmin, vmax=snr_vmax
            )
        ax.set_ylabel(f"{station}\nrange (km)")
        if row == 0:
            ax.set_title("RTI / SNR")

        ax = axes[row, 1]
        dop = np.where(good, d["doppler"], np.nan)
        cmap = plt.cm.seismic.copy()
        cmap.set_bad("0.8")
        if d["doppler"].size:
            dop_mesh = ax.pcolormesh(
                tplot, r, dop.T, shading="auto", cmap=cmap, vmin=-dop_vmax, vmax=dop_vmax
            )
        if row == 0:
            ax.set_title(f"Peak Doppler, SNR >= {snr_threshold_db:g} dB")

        ti, ri = np.where(good & np.isfinite(d["az"]) & np.isfinite(d["el"]))

        ax = axes[row, 2]
        if len(ti):
            az_sc = ax.scatter(tplot[ti], r[ri], c=d["az"][ti, ri], s=3.0, cmap="twilight", vmin=0, vmax=360)
        if row == 0:
            ax.set_title("AoD azimuth")

        ax = axes[row, 3]
        if len(ti):
            el_sc = ax.scatter(tplot[ti], r[ri], c=d["el"][ti, ri], s=3.0, cmap="viridis", vmin=0, vmax=90)
        if row == 0:
            ax.set_title("AoD elevation")

        for ax in axes[row, :]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True, color="0.9", linewidth=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel("UTC")
    fig.suptitle(title, fontsize=14)
    if snr_mesh is not None:
        fig.colorbar(snr_mesh, ax=axes[:, 0], label="SNR (dB)", shrink=0.9)
    if dop_mesh is not None:
        fig.colorbar(dop_mesh, ax=axes[:, 1], label="Hz", shrink=0.9)
    if az_sc is not None:
        fig.colorbar(az_sc, ax=axes[:, 2], label="deg", shrink=0.9)
    if el_sc is not None:
        fig.colorbar(el_sc, ax=axes[:, 3], label="deg", shrink=0.9)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Make head/trail overview plots from chunked AoD products.")
    p.add_argument("indir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--snr-threshold-db", type=float, default=6.0)
    p.add_argument("--head-seconds", type=float, default=2.0)
    p.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for station in args.stations:
        data = load_station(args.indir, station)
        if data is not None and len(data["t"]):
            datasets.append(data)
    if not datasets:
        raise SystemExit("No complete chunks found")

    t0 = min(float(np.min(d["t"])) for d in datasets)
    head = [clip_time(d, t0, t0 + args.head_seconds) for d in datasets]
    head = [d for d in head if len(d["t"])]
    draw_panel_grid(
        head,
        outdir / "head_echo_180620_180622.png",
        "Head echo, 18:06:20-18:06:22 UTC",
        args.snr_threshold_db,
    )
    draw_panel_grid(
        datasets,
        outdir / "trail_echo_full_duration.png",
        "Trail echo / full duration, 18:06:20-18:08:00 UTC",
        args.snr_threshold_db,
    )
    print(outdir / "head_echo_180620_180622.png")
    print(outdir / "trail_echo_full_duration.png")


if __name__ == "__main__":
    main()
