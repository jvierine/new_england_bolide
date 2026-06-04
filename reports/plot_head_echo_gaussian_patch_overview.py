#!/usr/bin/env python3
"""Plot station-level overviews of cached head-echo matched-filter patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0, notch_2d_doppler_repetition  # noqa: E402

DEFAULT_CACHE = ROOT / "products" / "head_echo_gaussian_patches" / "head_echo_gaussian_patches_16code.h5"
DEFAULT_OUT_DIR = ROOT / "reports" / "figures" / "head_echo_gaussian_overviews"
ROI_FILE = ROOT / "annotations" / "head_echo_rois.json"


def load_rois() -> dict:
    if not ROI_FILE.exists():
        return {}
    with ROI_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("stations", {})


def station_keys(h: h5py.File, stations: list[str] | None) -> list[str]:
    if stations is not None:
        return stations
    if "stations" not in h:
        return [key for key in h.keys() if isinstance(h[key], h5py.Group)]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]


def cached_station_overview(group: h5py.Group, n_coh: int, notch_half_width: int) -> dict:
    keys = sorted(key for key in group.keys() if key.startswith("patch_"))
    if not keys:
        raise ValueError("no patch_* groups")

    first = group[keys[0]]
    rel_freqs = first["rel_freqs"][()].astype(float)
    range_residual_grid_km = np.linspace(-40.0, 40.0, 161)

    time_rel_s = []
    raw_doppler = []
    filtered_doppler = []
    raw_range = []
    filtered_range = []

    for key in keys:
        patch = group[key]
        raw = patch["raw_db"][()].astype(float)
        filtered = notch_2d_doppler_repetition(raw, n_coh, notch_half_width)
        seed_range_m = float(patch.attrs["seed_range_m"])
        range_residual_km = (patch["range_m"][()].astype(float) - seed_range_m) / 1000.0

        time_rel_s.append(float(patch.attrs["time_unix"]) - EVENT_T0)
        raw_doppler.append(np.nanmax(raw, axis=0))
        filtered_doppler.append(np.nanmax(filtered, axis=0))
        raw_range.append(np.interp(range_residual_grid_km, range_residual_km, np.nanmax(raw, axis=1), left=np.nan, right=np.nan))
        filtered_range.append(
            np.interp(range_residual_grid_km, range_residual_km, np.nanmax(filtered, axis=1), left=np.nan, right=np.nan)
        )

    return {
        "time_rel_s": np.asarray(time_rel_s),
        "rel_freqs": rel_freqs,
        "range_residual_grid_km": range_residual_grid_km,
        "raw_doppler": np.asarray(raw_doppler),
        "filtered_doppler": np.asarray(filtered_doppler),
        "raw_range": np.asarray(raw_range),
        "filtered_range": np.asarray(filtered_range),
    }


def image_scale(arrays: list[np.ndarray], low_pct: float = 5.0, high_pct: float = 99.0) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    return float(np.percentile(vals, low_pct)), float(np.percentile(vals, high_pct))


def add_roi_shading(ax: plt.Axes, station: str, rois: dict) -> None:
    roi = rois.get(station, {})
    tr = roi.get("time_range")
    if not tr:
        return
    start = float(tr["start_unix"]) - EVENT_T0
    end = float(tr["end_unix"]) - EVENT_T0
    ax.axvspan(start, end, color="cyan", alpha=0.15, lw=0)


def plot_station(path: Path, station: str, data: dict, rois: dict) -> None:
    time = data["time_rel_s"]
    freq = data["rel_freqs"]
    rr = data["range_residual_grid_km"]
    t_extent = (float(time.min()), float(time.max()))
    doppler_extent = (*t_extent, float(freq.min()), float(freq.max()))
    range_extent = (*t_extent, float(rr.min()), float(rr.max()))
    dop_vmin, dop_vmax = image_scale([data["raw_doppler"], data["filtered_doppler"]])
    rng_vmin, rng_vmax = image_scale([data["raw_range"], data["filtered_range"]])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    panels = [
        (axes[0, 0], data["raw_doppler"].T, doppler_extent, dop_vmin, dop_vmax, "Raw max over range", r"$f-f_{D,\mathrm{model}}$ (Hz)"),
        (
            axes[0, 1],
            data["filtered_doppler"].T,
            doppler_extent,
            dop_vmin,
            dop_vmax,
            "2D FFT notch filtered max over range",
            r"$f-f_{D,\mathrm{model}}$ (Hz)",
        ),
        (axes[1, 0], data["raw_range"].T, range_extent, rng_vmin, rng_vmax, "Raw max over Doppler", "Range residual (km)"),
        (
            axes[1, 1],
            data["filtered_range"].T,
            range_extent,
            rng_vmin,
            rng_vmax,
            "2D FFT notch filtered max over Doppler",
            "Range residual (km)",
        ),
    ]
    for ax, arr, extent, vmin, vmax, title, ylabel in panels:
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="inferno")
        add_roi_shading(ax, station, rois)
        ax.axhline(0.0, color="cyan", lw=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Time from 18:06:20 UTC (s)")
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label="dB")
    label = STATION_LABELS.get(station, station)
    fig.suptitle(f"{label}: 16-code matched-filter overview on common a priori time grid", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stations", nargs="+", default=None)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--notch-half-width", type=int, default=1)
    args = p.parse_args()

    rois = load_rois()
    with h5py.File(args.cache, "r") as h:
        for station in station_keys(h, args.stations):
            print(f"{station}: plotting overview", flush=True)
            data = cached_station_overview(h[station], args.n_coh, args.notch_half_width)
            plot_station(args.out_dir / f"{station}_overview.png", station, data, rois)
    print(f"Wrote overview figures under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
