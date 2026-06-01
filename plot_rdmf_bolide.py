#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def read_rdmf(fname):
    with h5py.File(fname, "r") as h:
        P = h["P"][()]
        N = h["N"][()]
        D = h["D"][()]
        t_sample = h["t_sample"][()]
        range_idx = h["range_idx"][()]
        if "range_km" in h:
            rvec_km = h["range_km"][()]
        else:
            rvec_km = range_idx * 3.0
        sample_rate = float(h["sample_rate"][()])
        station = h["station"][()].decode()
        frequency_hz = float(h["frequency_hz"][()])
    unix_seconds = t_sample / sample_rate
    tvec = np.array(unix_seconds * 1e9, dtype="datetime64[ns]")
    return P, N, D, tvec, rvec_km, station, frequency_hz


def datetime64_to_mpl(tvec):
    return tvec.astype("datetime64[us]").astype(object)


def main():
    p = argparse.ArgumentParser(description="Plot bolide range-Doppler matched-filter products.")
    p.add_argument("input")
    p.add_argument("--snr-threshold", type=float, default=5.0)
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    P, N, D, tvec, rvec_km, station, frequency_hz = read_rdmf(args.input)
    tplot = datetime64_to_mpl(tvec)
    snr = P / np.maximum(N, 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-6))
    D_plot = D.copy()
    D_plot[snr < args.snr_threshold] = np.nan

    prefix = args.out_prefix
    if prefix is None:
        prefix = str(Path(args.input).with_suffix(""))

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    pcm = ax.pcolormesh(tplot, rvec_km, snr_db.T, shading="auto", cmap="inferno")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("SNR (dB)")
    ax.set_title(f"{station} range-time intensity")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Bistatic group range (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=35)
    snr_png = f"{prefix}_rti.png"
    fig.savefig(snr_png, bbox_inches="tight")

    cmap = plt.cm.seismic.copy()
    cmap.set_bad("0.5")
    vmax = np.nanpercentile(np.abs(D_plot), 98) if np.any(np.isfinite(D_plot)) else 1000.0
    vmax = max(1000.0, float(vmax))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    pcm = ax.pcolormesh(
        tplot,
        rvec_km,
        D_plot.T,
        shading="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Doppler shift (Hz)")
    ax.set_title(f"{station} peak Doppler, SNR >= {args.snr_threshold:g}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Bistatic group range (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=35)
    dop_png = f"{prefix}_doppler.png"
    fig.savefig(dop_png, bbox_inches="tight")

    peak = np.nanmax(snr)
    ti, ri = np.unravel_index(np.nanargmax(snr), snr.shape)
    lam = 299792458.0 / frequency_hz
    print(f"Wrote {snr_png}")
    print(f"Wrote {dop_png}")
    print(
        f"Peak SNR={peak:.1f} at {tvec[ti]} UTC, "
        f"range={rvec_km[ri]:.1f} km, doppler={D[ti, ri]:.1f} Hz, "
        f"monostatic-equivalent speed={abs(D[ti, ri]) * lam / 2.0:.1f} m/s"
    )

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
