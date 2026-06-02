#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from plot_aod_chunks import load_station, unix_to_mpl
from station_config import short_station_name


DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def per_time_snr_db(power):
    noise = np.nanmedian(power, axis=1)
    noise = np.maximum(noise, 1e-12)
    return 10.0 * np.log10(np.maximum(power / noise[:, None], 1e-12))


def robust_limits(x, default=(0.0, 12.0)):
    good = x[np.isfinite(x)]
    if good.size == 0:
        return default
    lo = max(default[0], float(np.nanpercentile(good, 5)))
    hi = max(default[1], float(np.nanpercentile(good, 99.5)))
    return lo, hi


def plot_station_head(data, outdir, seconds, dynamic_doppler_hz):
    t0 = float(np.min(data["t"]))
    keep = (data["t"] >= t0) & (data["t"] <= t0 + seconds)
    t = data["t"][keep]
    r = data["range_km"]
    power = data["peak_power"][keep, :]
    doppler = data["doppler"][keep, :]
    if len(t) == 0:
        return None

    snr_db = per_time_snr_db(power)
    dynamic_snr_db = np.where(np.abs(doppler) >= dynamic_doppler_hz, snr_db, np.nan)
    pwr_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    tplot = unix_to_mpl(t)

    station = data["station"]
    station_name = short_station_name(station)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snr_vmin, snr_vmax = robust_limits(snr_db, default=(-2.0, 12.0))
    dyn_vmin, dyn_vmax = robust_limits(dynamic_snr_db, default=(-2.0, 8.0))
    pwr_vmin, pwr_vmax = robust_limits(pwr_db, default=(float(np.nanmin(pwr_db)), float(np.nanmax(pwr_db))))

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.5), sharex=True, constrained_layout=True)

    pcm = axes[0].pcolormesh(tplot, r, snr_db.T, shading="auto", cmap="inferno", vmin=snr_vmin, vmax=snr_vmax)
    fig.colorbar(pcm, ax=axes[0], label="dB")
    axes[0].set_title(f"{station_name}: matched-filter RTI, incoherent average over 2 pols and all codes")
    axes[0].set_ylabel("Range (km)")

    pcm = axes[1].pcolormesh(tplot, r, pwr_db.T, shading="auto", cmap="magma", vmin=pwr_vmin, vmax=pwr_vmax)
    fig.colorbar(pcm, ax=axes[1], label="Power (dB)")
    axes[1].set_title("Matched-filter peak power")
    axes[1].set_ylabel("Range (km)")

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("0.82")
    pcm = axes[2].pcolormesh(tplot, r, dynamic_snr_db.T, shading="auto", cmap=cmap, vmin=dyn_vmin, vmax=dyn_vmax)
    fig.colorbar(pcm, ax=axes[2], label="dB")
    axes[2].set_title(f"Same RTI, masked to |peak Doppler| >= {dynamic_doppler_hz:g} Hz")
    axes[2].set_ylabel("Range (km)")

    dop = np.where(np.isfinite(dynamic_snr_db), doppler, np.nan)
    dop_good = np.abs(dop[np.isfinite(dop)])
    dop_vmax = max(1000.0, float(np.nanpercentile(dop_good, 98))) if dop_good.size else 1000.0
    dcmap = plt.cm.seismic.copy()
    dcmap.set_bad("0.82")
    pcm = axes[3].pcolormesh(tplot, r, dop.T, shading="auto", cmap=dcmap, vmin=-dop_vmax, vmax=dop_vmax)
    fig.colorbar(pcm, ax=axes[3], label="Hz")
    axes[3].set_title("Peak Doppler for dynamic bins")
    axes[3].set_ylabel("Range (km)")
    axes[3].set_xlabel("UTC")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
        ax.grid(True, color="0.9", linewidth=0.4)
    fig.autofmt_xdate(rotation=25)

    out = outdir / f"{station}_head_matched_filter.png"
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)

    imax = int(np.nanargmax(snr_db))
    ti, ri = np.unravel_index(imax, snr_db.shape)
    return out, float(np.nanmax(snr_db)), float(r[ri]), float(t[ti] - t0), float(doppler[ti, ri])


def main():
    p = argparse.ArgumentParser(description="Plot first seconds of matched-filter head echo products.")
    p.add_argument("indir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--dynamic-doppler-hz", type=float, default=200.0)
    p.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    args = p.parse_args()

    for station in args.stations:
        data = load_station(args.indir, station)
        if data is None:
            print(f"{station}: no complete chunks")
            continue
        result = plot_station_head(data, args.out_dir, args.seconds, args.dynamic_doppler_hz)
        if result is None:
            print(f"{station}: no samples in requested window")
            continue
        out, peak_db, peak_range, peak_dt, peak_dop = result
        print(
            f"{station}: wrote {out}; all-doppler peak={peak_db:.1f} dB "
            f"range={peak_range:.1f} km dt={peak_dt:.3f}s dop={peak_dop:.1f} Hz"
        )


if __name__ == "__main__":
    main()
