#!/usr/bin/env python3
"""Plot manually reviewed head-echo range/Doppler measurements for the appendix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "reports" / "head_echo_manual_selection_ext0p5" / "head_echo_manual_selection_all.csv"
DEFAULT_OUT = ROOT / "reports" / "figures" / "head_echo_manual_selection_range_doppler.png"

STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, CT",
}

STATION_COLORS = {
    "beverly": "tab:blue",
    "clinton": "tab:red",
    "pomfret": "tab:green",
}

SELECTION_MARKERS = {
    "real": "o",
    "unknown": ".",
    "reject": "x",
}

SELECTION_ALPHA = {
    "real": 0.95,
    "unknown": 0.55,
    "reject": 0.30,
}


def load_selection(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_rel_s"] = pd.to_numeric(df["time_rel_s"], errors="coerce")
    df["fit_range_m"] = pd.to_numeric(df["fit_range_m"], errors="coerce")
    df["seed_range_m"] = pd.to_numeric(df["seed_range_m"], errors="coerce")
    df["fit_doppler_hz"] = pd.to_numeric(df["fit_doppler_hz"], errors="coerce")
    df["seed_doppler_hz"] = pd.to_numeric(df["seed_doppler_hz"], errors="coerce")
    df["selection"] = df["selection"].fillna("unknown").astype(str).str.lower()
    df = df.dropna(subset=["time_rel_s", "fit_range_m", "fit_doppler_hz"]).copy()
    if mode == "real":
        df = df[df["selection"] == "real"].copy()
    elif mode == "accepted":
        df = df[df["selection"] != "reject"].copy()
    elif mode != "all":
        raise ValueError(f"Unknown mode {mode!r}")
    df["fit_range_km"] = df["fit_range_m"] / 1000.0
    df["seed_range_km"] = df["seed_range_m"] / 1000.0
    return df.sort_values(["station", "time_rel_s", "idx"])


def plot_selection(df: pd.DataFrame, out: Path, mode: str, overlay_seed: bool) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True, constrained_layout=True)

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, f"No {mode} measurements", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    else:
        seed_label_written = False
        for station, sdf in df.groupby("station", sort=False):
            color = STATION_COLORS.get(station, "0.3")
            label = STATION_LABELS.get(station, station)
            for selection, ssdf in sdf.groupby("selection", sort=False):
                marker = SELECTION_MARKERS.get(selection, ".")
                alpha = SELECTION_ALPHA.get(selection, 0.55)
                suffix = "" if mode == "real" else f" ({selection})"
                axes[0].scatter(
                    ssdf["time_rel_s"],
                    ssdf["fit_range_km"],
                    s=30 if marker != "." else 22,
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    linewidths=0.8,
                    label=f"{label}{suffix}",
                )
                axes[1].scatter(
                    ssdf["time_rel_s"],
                    ssdf["fit_doppler_hz"],
                    s=30 if marker != "." else 22,
                    marker=marker,
                    color=color,
                    alpha=alpha,
                    linewidths=0.8,
                )
            if overlay_seed:
                seed_label = "Original range/Doppler picks" if not seed_label_written else None
                axes[0].scatter(
                    sdf["time_rel_s"],
                    sdf["seed_range_km"],
                    s=34,
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.9,
                    alpha=0.65,
                    label=seed_label,
                    zorder=5,
                )
                axes[1].scatter(
                    sdf["time_rel_s"],
                    sdf["seed_doppler_hz"],
                    s=34,
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.9,
                    alpha=0.65,
                    zorder=5,
                )
                seed_label_written = True

    axes[0].set_ylabel("Bistatic range (km)")
    axes[1].set_ylabel("Doppler (Hz)")
    axes[1].set_xlabel("Seconds after 18:06:20 UTC")
    axes[0].set_title("Head-echo range measurements")
    axes[1].set_title("Head-echo Doppler measurements")
    for ax in axes:
        ax.grid(True, color="0.85", linewidth=0.8)
    axes[0].legend(loc="best", ncols=1, framealpha=0.88)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mode", choices=["all", "accepted", "real"], default="all")
    p.add_argument("--no-seed-overlay", action="store_true")
    args = p.parse_args()

    df = load_selection(args.selection, args.mode)
    plot_selection(df, args.out, args.mode, overlay_seed=not args.no_seed_overlay)
    print(args.out)
    print(f"{len(df)} plotted measurements")


if __name__ == "__main__":
    main()
