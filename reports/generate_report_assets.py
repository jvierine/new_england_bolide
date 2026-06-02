#!/usr/bin/env python3
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import load_station_yaml
from range_doppler_mf_bolide import (
    detrend_window,
    get_codes,
    interpolate_rect,
    read_channel_files,
    rf_file_for_time,
)


DATA_DIR = ROOT / "2026_05_30_bolide"
FIG_DIR = ROOT / "reports" / "figures"
STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def read_raw(station, channel, start_second, seconds):
    files = rf_file_for_time(DATA_DIR / station, channel, "2026-05-30T18-00-00")
    return read_channel_files(files, start_second=start_second, seconds=seconds)


def spike_scan(start_second=75.0, seconds=10.0, spike_sigma=8.0):
    rows = []
    total_samples = 0
    total_spikes = 0
    worst = None
    for station in STATIONS:
        for channel in (0, 1):
            z, sr, _ = read_raw(station, channel, start_second, seconds)
            y = z - np.median(z)
            med_abs = float(np.median(np.abs(y)))
            if med_abs > 0:
                score = np.abs(y) / med_abs
                mask = score > spike_sigma
            else:
                score = np.zeros(len(y))
                mask = np.zeros(len(y), dtype=bool)
            count = int(np.sum(mask))
            rows.append((station, channel, len(y), count, med_abs))
            total_samples += len(y)
            total_spikes += count
            if count:
                imax = int(np.argmax(score))
                candidate = {
                    "score": float(score[imax]),
                    "station": station,
                    "channel": channel,
                    "index": imax,
                    "time_second": imax / sr,
                    "sample_rate": sr,
                    "median_abs": med_abs,
                    "z": z,
                }
                if worst is None or candidate["score"] > worst["score"]:
                    worst = candidate
    return rows, total_samples, total_spikes, worst


def plot_detrend_example(spike_sigma=8.0):
    rows, total_samples, total_spikes, worst = spike_scan(spike_sigma=spike_sigma)
    if worst is None:
        raise RuntimeError("No spike example found")

    z = worst["z"]
    imax = worst["index"]
    sr = worst["sample_rate"]
    med = np.median(z)
    y = z - med
    threshold = spike_sigma * worst["median_abs"]
    y_filtered = detrend_window(z, spike_sigma=spike_sigma)

    half = 180
    sl = slice(max(0, imax - half), min(len(z), imax + half + 1))
    t_ms = (np.arange(sl.start, sl.stop) - imax) / sr * 1e3

    fig = plt.figure(figsize=(11.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.0])
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    boxes = [
        ("raw complex samples", 0.10),
        ("subtract complex median", 0.38),
        (f"if |sample| > {spike_sigma:g} x median(|sample|)", 0.67),
        ("set spike sample to 0", 0.90),
    ]
    for text, x in boxes:
        ax0.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.25"),
            fontsize=10,
        )
    for x0, x1 in zip([b[1] for b in boxes[:-1]], [b[1] for b in boxes[1:]]):
        ax0.annotate("", xy=(x1 - 0.09, 0.55), xytext=(x0 + 0.09, 0.55), arrowprops=dict(arrowstyle="->"))

    ax = fig.add_subplot(gs[1])
    ax.plot(t_ms, np.abs(y[sl]), color="tab:red", lw=1.3, label="after median subtraction")
    ax.plot(t_ms, np.abs(y_filtered[sl]), color="tab:blue", lw=1.1, label="after spike filter")
    ax.axhline(threshold, color="0.25", ls="--", lw=1.0, label=f"{spike_sigma:g} x median(|sample|)")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_title(
        f"Example spike removed by detrend_window: {worst['station']} ch{worst['channel']}, "
        f"18:06:15+{worst['time_second']:.5f} s"
    )
    ax.set_xlabel("Time relative to spike (ms)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.legend(loc="upper right")

    out = FIG_DIR / "detrend_window_spike_example.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, rows, total_samples, total_spikes, worst


def beverly_range_doppler_snapshot(start_second=80.6):
    site = load_station_yaml(DATA_DIR, "beverly")
    range_interp = 4
    n_coh = 8
    codes_per_step = 1
    fmin, fmax = -50000.0, 50000.0
    code_len_i = site["code_len"] * range_interp
    sample_rate = 100000.0
    sample_rate_i = sample_rate * range_interp
    win_len = n_coh * code_len_i
    gate_km = site["range_resolution_m"] / range_interp / 1000.0
    rmax = int(round(600.0 / gate_km))
    range_idx = np.arange(0, rmax + 1, dtype=np.int64)
    read_len = win_len + rmax + 1
    read_seconds = read_len / sample_rate_i + 0.01

    zc = []
    for ch in (0, 1):
        z, sr, _ = read_raw("beverly", ch, start_second, read_seconds)
        sample_rate = sr
        zc.append(interpolate_rect(z, range_interp))
    sample_rate_i = sample_rate * range_interp
    fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    fidx = np.where((fvec >= fmin) & (fvec <= fmax))[0]
    fvec0 = fvec[fidx]
    codes = get_codes(site["seeds"], site["code_len"], n_coh, range_interp=range_interp)

    S = np.zeros((len(range_idx), len(fidx)), dtype=np.float32)
    for z in zc:
        z0 = detrend_window(z[:read_len], spike_sigma=0.0)
        windows = np.lib.stride_tricks.sliding_window_view(z0, win_len)[range_idx, :]
        for code in codes:
            Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
            S += np.abs(Z) ** 2
    S /= float(len(zc) * len(codes))

    dop_floor = np.median(S, axis=0)
    S_white = S / np.maximum(dop_floor[None, :], 1e-12)
    raw_noise_by_range = np.median(S, axis=1)
    white_noise_by_range = np.median(S_white, axis=1)
    ranges_km = range_idx * gate_km
    # Select the meteor-head example from the requested region: range 200-300 km
    # and Doppler -3000..-1000 Hz. The metric is whitened SNR because this is
    # the detector used by the high-resolution RTI product.
    white_snr = S_white / np.maximum(white_noise_by_range[:, None], 1e-12)
    range_sel = (ranges_km >= 200.0) & (ranges_km <= 300.0)
    dop_sel = (fvec0 > -3000.0) & (fvec0 < -1000.0)
    search = np.where(range_sel[:, None] & dop_sel[None, :], white_snr, -np.inf)
    ri, white_peak_i = np.unravel_index(int(np.argmax(search)), search.shape)
    target_range_km = float(ranges_km[ri])
    raw_spec = S[ri, :]
    white_spec = S_white[ri, :]
    raw_peak_i = int(np.argmax(raw_spec))

    return {
        "time_second": start_second,
        "ranges_km": ranges_km,
        "fvec": fvec0,
        "S": S,
        "S_white": S_white,
        "dop_floor": dop_floor,
        "raw_noise_by_range": raw_noise_by_range,
        "white_noise_by_range": white_noise_by_range,
        "target_range_km": float(ranges_km[ri]),
        "target_ri": ri,
        "raw_peak_hz": float(fvec0[raw_peak_i]),
        "white_peak_hz": float(fvec0[white_peak_i]),
        "raw_peak_snr_db": float(10.0 * np.log10(raw_spec[raw_peak_i] / max(raw_noise_by_range[ri], 1e-12))),
        "raw_selected_snr_db": float(
            10.0
            * np.log10(raw_spec[white_peak_i] / max(raw_noise_by_range[ri], 1e-12))
        ),
        "white_peak_snr_db": float(
            10.0 * np.log10(white_spec[white_peak_i] / max(white_noise_by_range[ri], 1e-12))
        ),
        "selected_doppler_snr_db": float(
            10.0
            * np.log10(
                white_spec[white_peak_i] / max(white_noise_by_range[ri], 1e-12)
            )
        ),
    }


def plot_doppler_whiten_example():
    snap = beverly_range_doppler_snapshot()
    f = snap["fvec"]
    r = snap["ranges_km"]
    ri = snap["target_ri"]
    target_range = snap["target_range_km"]
    raw_db = 10.0 * np.log10(np.maximum(snap["S"] / snap["raw_noise_by_range"][:, None], 1e-12))
    white_db = 10.0 * np.log10(np.maximum(snap["S_white"] / snap["white_noise_by_range"][:, None], 1e-12))
    spec_raw_db = raw_db[ri, :]
    spec_white_db = white_db[ri, :]
    raw_lim = np.nanpercentile(raw_db, [5, 99.7])
    white_lim = np.nanpercentile(white_db, [5, 99.7])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)
    p = axes[0, 0].pcolormesh(f / 1e3, r, raw_db, shading="auto", cmap="inferno", vmin=raw_lim[0], vmax=raw_lim[1])
    fig.colorbar(p, ax=axes[0, 0], label="dB above range median")
    axes[0, 0].axhline(target_range, color="cyan", lw=1.0)
    axes[0, 0].set_title("Before Doppler whitening")
    axes[0, 0].set_ylabel("Range (km)")

    p = axes[0, 1].pcolormesh(
        f / 1e3, r, white_db, shading="auto", cmap="inferno", vmin=white_lim[0], vmax=white_lim[1]
    )
    fig.colorbar(p, ax=axes[0, 1], label="dB above whitened range median")
    axes[0, 1].axhline(target_range, color="cyan", lw=1.0)
    axes[0, 1].set_title("After Doppler whitening")

    axes[1, 0].plot(f / 1e3, spec_raw_db, color="tab:red", lw=1.0)
    axes[1, 0].axhline(0, color="0.3", ls="--", lw=0.8, label="range-gate noise floor")
    axes[1, 0].axvline(snap["white_peak_hz"] / 1e3, color="tab:blue", ls=":", lw=1.0, label="whitened peak")
    axes[1, 0].set_title(f"Range gate {target_range:.1f} km before whitening")
    axes[1, 0].set_ylabel("Power / range-gate median (dB)")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(f / 1e3, spec_white_db, color="tab:blue", lw=1.0)
    axes[1, 1].axhline(0, color="0.3", ls="--", lw=0.8, label="whitened range-gate noise floor")
    axes[1, 1].axvline(snap["white_peak_hz"] / 1e3, color="tab:blue", ls=":", lw=1.0, label="whitened peak")
    axes[1, 1].set_title(f"Range gate {target_range:.1f} km after whitening")
    axes[1, 1].legend(loc="upper right")

    for ax in axes.ravel():
        ax.set_xlabel("Doppler shift (kHz)")
        ax.grid(True, color="0.9", linewidth=0.5)

    fig.suptitle("Beverly matched-filter snapshot at 18:06:20.600 UTC")
    out = FIG_DIR / "beverly_doppler_whiten_example.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, snap


def write_stats_tex(rows, total_samples, total_spikes, worst, snap, spike_sigma=8.0):
    lines = []
    lines.append(f"\\newcommand{{\\SpikeSigma}}{{{spike_sigma:g}}}\n")
    lines.append(f"\\newcommand{{\\SpikeTotalSamples}}{{{total_samples:,}}}\n")
    lines.append(f"\\newcommand{{\\SpikeTotalRemoved}}{{{total_spikes:,}}}\n")
    lines.append(f"\\newcommand{{\\SpikeFractionPercent}}{{{100.0 * total_spikes / total_samples:.5f}\\%}}\n")
    lines.append(f"\\newcommand{{\\WorstSpikeStation}}{{{worst['station']}}}\n")
    lines.append(f"\\newcommand{{\\WorstSpikeChannel}}{{{worst['channel']}}}\n")
    lines.append(f"\\newcommand{{\\WorstSpikeScore}}{{{worst['score']:.1f}}}\n")
    lines.append(f"\\newcommand{{\\WorstSpikeOffset}}{{{worst['time_second']:.5f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyExampleRange}}{{{snap['target_range_km']:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyRawPeakDoppler}}{{{snap['raw_peak_hz'] / 1e3:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyWhitePeakDoppler}}{{{snap['white_peak_hz'] / 1e3:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyRawPeakSnr}}{{{snap['raw_peak_snr_db']:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyRawSelectedSnr}}{{{snap['raw_selected_snr_db']:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlyWhitePeakSnr}}{{{snap['white_peak_snr_db']:.1f}}}\n")
    lines.append(f"\\newcommand{{\\BeverlySelectedSnr}}{{{snap['selected_doppler_snr_db']:.1f}}}\n")
    table = ["\\begin{tabular}{lrrrr}\n", "\\toprule\n", "Station & Channel & Samples & Removed & Fraction \\\\\n", "\\midrule\n"]
    for station, channel, n, count, _ in rows:
        table.append(f"{station} & {channel} & {n:,} & {count:,} & {100.0 * count / n:.5f}\\% \\\\\n")
    table.append("\\bottomrule\n\\end{tabular}\n")
    (ROOT / "reports" / "spike_stats_table.tex").write_text("".join(table), encoding="utf-8")
    (ROOT / "reports" / "report_stats.tex").write_text("".join(lines), encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    spike_fig, rows, total_samples, total_spikes, worst = plot_detrend_example(spike_sigma=8.0)
    whiten_fig, snap = plot_doppler_whiten_example()
    write_stats_tex(rows, total_samples, total_spikes, worst, snap, spike_sigma=8.0)
    print(spike_fig)
    print(whiten_fig)
    print(ROOT / "reports" / "report_stats.tex")
    print(ROOT / "reports" / "spike_stats_table.tex")


if __name__ == "__main__":
    main()
