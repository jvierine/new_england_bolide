#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def read_chunk(fname):
    with h5py.File(fname, "r") as h:
        if int(h.attrs.get("complete", 0)) != 1:
            return None
        station = h["station"][()].decode()
        sample_rate = float(h["sample_rate"][()])
        return {
            "station": station,
            "t": h["t_sample"][()] / sample_rate,
            "range_km": h["range_km"][()],
            "snr": h["snr"][()],
            "doppler": h["peak_doppler_hz"][()],
            "az": h["aod_az_deg"][()],
            "el": h["aod_el_deg"][()],
            "score": h["aod_score"][()],
        }


def load_station(indir, station):
    rows = []
    for fname in sorted(Path(indir).glob(f"aod_peak_{station}_*.h5")):
        row = read_chunk(fname)
        if row is not None:
            rows.append(row)
    if not rows:
        return None

    range_km = rows[0]["range_km"]
    t = np.concatenate([r["t"] for r in rows])
    order = np.argsort(t)
    out = {"station": station, "range_km": range_km, "t": t[order]}
    for key in ("snr", "doppler", "az", "el", "score"):
        out[key] = np.concatenate([r[key] for r in rows], axis=0)[order, :]
    return out


def unix_to_mpl(t):
    return mdates.epoch2num(np.asarray(t, dtype=float))


def plot_station(data, outdir, snr_threshold_db):
    station = data["station"]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tplot = unix_to_mpl(data["t"])
    r = data["range_km"]
    snr_db = 10.0 * np.log10(np.maximum(data["snr"], 1e-12))
    threshold_linear = 10.0 ** (snr_threshold_db / 10.0)
    good = data["snr"] >= threshold_linear

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    pcm = ax.pcolormesh(tplot, r, snr_db.T, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="SNR (dB)")
    ax.set_title(f"{station}: range-time intensity")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Bistatic group range (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)
    rti_png = outdir / f"{station}_rti.png"
    fig.savefig(rti_png, bbox_inches="tight")
    plt.close(fig)

    dop = np.where(good, data["doppler"], np.nan)
    vmax = np.nanpercentile(np.abs(dop), 98) if np.any(np.isfinite(dop)) else 1000.0
    vmax = max(float(vmax), 1000.0)
    cmap = plt.cm.seismic.copy()
    cmap.set_bad("0.75")
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    pcm = ax.pcolormesh(tplot, r, dop.T, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="Peak Doppler (Hz)")
    ax.set_title(f"{station}: peak Doppler, SNR >= {snr_threshold_db:g} dB")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Bistatic group range (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)
    dop_png = outdir / f"{station}_doppler.png"
    fig.savefig(dop_png, bbox_inches="tight")
    plt.close(fig)

    ti, ri = np.where(good & np.isfinite(data["az"]) & np.isfinite(data["el"]))
    if len(ti):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, constrained_layout=True)
        sc = ax0.scatter(tplot[ti], r[ri], c=data["az"][ti, ri], s=7, cmap="twilight", vmin=0, vmax=360)
        fig.colorbar(sc, ax=ax0, label="Azimuth (deg)")
        ax0.set_title(f"{station}: AoD azimuth, SNR >= {snr_threshold_db:g} dB")
        ax0.set_ylabel("Bistatic group range (km)")
        sc = ax1.scatter(tplot[ti], r[ri], c=data["el"][ti, ri], s=7, cmap="viridis", vmin=0, vmax=90)
        fig.colorbar(sc, ax=ax1, label="Elevation (deg)")
        ax1.set_title(f"{station}: AoD elevation")
        ax1.set_xlabel("UTC")
        ax1.set_ylabel("Bistatic group range (km)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate(rotation=30)
        aod_png = outdir / f"{station}_aod.png"
        fig.savefig(aod_png, bbox_inches="tight")
        plt.close(fig)
    else:
        aod_png = None

    peak = np.nanmax(snr_db)
    return rti_png, dop_png, aod_png, len(data["t"]), int(np.sum(good)), float(peak)


def main():
    p = argparse.ArgumentParser(description="Plot chunked AoD/RDMF products.")
    p.add_argument("indir", help="Directory containing aod_peak_<station>_*.h5 files")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--snr-threshold-db", type=float, default=6.0)
    p.add_argument("--stations", nargs="+", default=["amherstnh", "beverly", "clinton", "pomfret", "siena"])
    args = p.parse_args()

    outdir = args.out_dir or str(Path(args.indir) / "plots")
    for station in args.stations:
        data = load_station(args.indir, station)
        if data is None:
            print(f"{station}: no complete chunks")
            continue
        paths = plot_station(data, outdir, args.snr_threshold_db)
        rti, dop, aod, n_time, n_det, peak_db = paths
        print(
            f"{station}: {n_time} time rows, {n_det} detections, "
            f"peak_snr={peak_db:.1f} dB"
        )
        print(f"  {rti}")
        print(f"  {dop}")
        if aod is not None:
            print(f"  {aod}")


if __name__ == "__main__":
    main()
