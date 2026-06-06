#!/usr/bin/env python3
"""Plot saved one-code unwhitened head-echo search products."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import correlated_ellipse_points, gaussian2d, smooth_power_db  # noqa: E402
from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402

DEFAULT_H5 = ROOT / "reports" / "coh1_raw_head_echo_test" / "cache" / "coh1_raw_head_echo_search.h5"
DEFAULT_CSV = ROOT / "reports" / "coh1_raw_head_echo_test" / "coh1_raw_head_echo_search_summary.csv"
DEFAULT_DEBUG_DIR = ROOT / "reports" / "coh1_raw_head_echo_test" / "figures" / "raw_match_debug"
DEFAULT_TIMESERIES = ROOT / "reports" / "coh1_raw_head_echo_test" / "figures" / "coh1_raw_range_doppler_timeseries.png"


def db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-30))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict, key: str) -> float:
    val = row.get(key, "")
    return np.nan if val == "" else float(val)


def fit_from_group(group: h5py.Group) -> dict | None:
    if "gaussian_fit" not in group:
        return None
    return dict(group["gaussian_fit"].attrs.items())


def plot_debug_patch(station: str, group: h5py.Group, out: Path, smooth_range_px: float, smooth_freq_px: float) -> None:
    range_km = group["range_m"][()] / 1000.0
    rel_hz = group["rel_freqs"][()]
    raw_db = db(group["raw_power"][()])
    smoothed_db = smooth_power_db(raw_db, smooth_range_px, smooth_freq_px)

    seed_range_km = float(group.attrs["seed_range_m"]) / 1000.0
    seed_rel_hz = float(group.attrs["seed_doppler_hz"]) - float(group.attrs["model_doppler_hz"])
    time_rel = float(group.attrs["time_rel"])
    idx = int(group.attrs["idx"])
    fit = fit_from_group(group)

    if fit is None:
        fit_range_km = seed_range_km
        fit_rel_hz = seed_rel_hz
        fit_model = None
        ell_x = ell_y = np.array([])
    else:
        fit_range_km = float(fit["range_km"])
        fit_rel_hz = float(fit["rel_doppler_hz"])
        sig_r = float(fit["sigma_range_km"])
        sig_f = float(fit["sigma_doppler_hz"])
        rho = float(fit["range_doppler_correlation"])
        rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
        fit_model = gaussian2d(
            np.array(
                [
                    float(fit["offset_db"]),
                    float(fit["amplitude_db"]),
                    fit_range_km,
                    fit_rel_hz,
                    sig_r,
                    sig_f,
                    rho,
                ]
            ),
            rr,
            ff,
        )
        ell_x, ell_y = correlated_ellipse_points(fit_range_km, fit_rel_hz, sig_r, sig_f, rho)

    x_pad = 650.0
    y_pad = 14.0
    x_min = min(seed_rel_hz, fit_rel_hz) - x_pad
    x_max = max(seed_rel_hz, fit_rel_hz) + x_pad
    y_min = min(seed_range_km, fit_range_km) - y_pad
    y_max = max(seed_range_km, fit_range_km) + y_pad

    images = [
        ("Raw one-code matched filter", raw_db),
        ("Raw + smoothed", smoothed_db),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, img) in zip(axes, images, strict=True):
        show = (
            (range_km[:, None] >= y_min)
            & (range_km[:, None] <= y_max)
            & (rel_hz[None, :] >= x_min)
            & (rel_hz[None, :] <= x_max)
        )
        finite = img[show & np.isfinite(img)]
        vmin = float(np.nanpercentile(finite, 3.0))
        vmax = float(np.nanpercentile(finite, 99.6))
        mappable = ax.pcolormesh(rel_hz, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot([seed_rel_hz, fit_rel_hz], [seed_range_km, fit_range_km], color="white", lw=0.9, alpha=0.9)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=9, mew=1.4, color="#9be7ff", linestyle="none")
        ax.plot(fit_rel_hz, fit_range_km, marker="x", ms=8, mew=1.4, color="#42ff9f", linestyle="none")
        if fit_model is not None:
            ax.plot(ell_x, ell_y, color="#42ff9f", lw=1.2)
            ax.contour(
                rel_hz,
                range_km,
                fit_model,
                levels=[
                    float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-2.0),
                    float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-0.5),
                ],
                colors=["#42ff9f", "#d8ffe8"],
                linewidths=[0.8, 1.0],
            )
        ax.set_title(title)
        ax.set_xlabel(r"$f-f_{D,model}$ (Hz)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(color="white", alpha=0.12, lw=0.5)
        cb = fig.colorbar(mappable, ax=ax, pad=0.012)
        cb.set_label("dB")
    axes[0].set_ylabel("Bistatic range (km)")

    if fit is None:
        subtitle = "no Gaussian fit"
    else:
        dr_m = (fit_range_km - seed_range_km) * 1000.0
        df_hz = fit_rel_hz - seed_rel_hz
        subtitle = f"Gaussian correction {dr_m:+.0f} m, {df_hz:+.1f} Hz"
    fig.suptitle(f"{STATION_LABELS.get(station, station)} idx={idx} t={time_rel:.3f} s: {subtitle}", y=1.03, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(rows: list[dict], out: Path) -> None:
    stations = []
    for row in rows:
        st = row["station"]
        if st not in stations:
            stations.append(st)
    colors = {"beverly": "#2f74c0", "clinton": "#d95f02", "pomfret": "#2ca25f"}

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True, constrained_layout=True)
    for station in stations:
        srows = [row for row in rows if row["station"] == station and row.get("success") == "True"]
        srows.sort(key=lambda row: as_float(row, "time_rel_s"))
        t = np.array([as_float(row, "time_rel_s") for row in srows])
        seed_range = np.array([as_float(row, "seed_range_m") for row in srows]) / 1000.0
        fit_range = np.array([as_float(row, "fit_range_m") for row in srows]) / 1000.0
        seed_dop = np.array([as_float(row, "seed_doppler_hz") for row in srows]) / 1000.0
        fit_dop = np.array([as_float(row, "fit_doppler_hz") for row in srows]) / 1000.0
        color = colors.get(station)
        label = STATION_LABELS.get(station, station)
        axes[0].plot(t, seed_range, color=color, lw=0.8, alpha=0.25)
        axes[0].plot(t, fit_range, color=color, lw=1.6, marker=".", ms=3.0, label=label)
        axes[1].plot(t, seed_dop, color=color, lw=0.8, alpha=0.25)
        axes[1].plot(t, fit_dop, color=color, lw=1.6, marker=".", ms=3.0, label=label)

    axes[0].set_ylabel("Bistatic range (km)")
    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].set_xlabel("Seconds after 18:06:20 UTC")
    axes[0].set_title("One-code unwhitened Gaussian head-echo measurements")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(loc="best", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--summary", type=Path, default=DEFAULT_CSV)
    p.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    p.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES)
    p.add_argument("--stations", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=1.0)
    p.add_argument("--skip-debug", action="store_true")
    args = p.parse_args()

    rows = read_rows(args.summary)
    if args.stations is not None:
        keep = set(args.stations)
        rows = [row for row in rows if row["station"] in keep]
    plot_time_series(rows, args.timeseries)
    print(args.timeseries, flush=True)

    if args.skip_debug:
        return

    n_plotted = 0
    with h5py.File(args.h5, "r") as h:
        stations = args.stations
        if stations is None:
            stations = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]
        for station in stations:
            group = h[station]
            keys = sorted([key for key in group.keys() if key.startswith("patch_")])
            if args.limit is not None:
                keys = keys[: args.limit]
            print(f"{station}: plotting {len(keys)} debug patches", flush=True)
            for key in keys:
                patch = group[key]
                idx = int(patch.attrs["idx"])
                time_rel = float(patch.attrs["time_rel"])
                out = args.debug_dir / station / f"{station}_idx{idx:04d}_t{time_rel:06.3f}.png"
                plot_debug_patch(station, patch, out, args.smooth_range_pixels, args.smooth_frequency_pixels)
                n_plotted += 1
    print(f"Wrote {n_plotted} debug figures under {args.debug_dir}", flush=True)


if __name__ == "__main__":
    main()
