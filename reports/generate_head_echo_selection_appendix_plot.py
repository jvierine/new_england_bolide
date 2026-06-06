#!/usr/bin/env python3
"""Plot manually reviewed head-echo range/Doppler measurements for the appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "reports" / "head_echo_python_selection_workflow_ext0p5" / "head_echo_manual_selection_all.csv"
)
DEFAULT_OUT = ROOT / "reports" / "figures" / "head_echo_python_selection_range_doppler.png"
DEFAULT_TRAJECTORY_FIT = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
EVENT_SECOND_UNIX = 1780164380.0

STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, VT",
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
    df["fit_doppler_hz"] = pd.to_numeric(df["fit_doppler_hz"], errors="coerce")
    df["selection"] = df["selection"].fillna("unknown").astype(str).str.lower()
    df = df.dropna(subset=["time_rel_s", "fit_range_m", "fit_doppler_hz"]).copy()
    if mode == "real":
        df = df[df["selection"] == "real"].copy()
    elif mode == "accepted":
        df = df[df["selection"] != "reject"].copy()
    elif mode != "all":
        raise ValueError(f"Unknown mode {mode!r}")
    df["fit_range_km"] = df["fit_range_m"] / 1000.0
    return df.sort_values(["station", "time_rel_s", "idx"])


def load_trajectory_observations(path: Path) -> pd.DataFrame:
    fit = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(fit["observations"])
    df["time_rel_s"] = pd.to_numeric(df["unix_time"], errors="coerce") - EVENT_SECOND_UNIX
    df["range_km"] = pd.to_numeric(df["range_m"], errors="coerce") / 1000.0
    df["doppler_hz"] = pd.to_numeric(df["doppler_hz"], errors="coerce")
    df = df.dropna(subset=["time_rel_s", "range_km", "doppler_hz"]).copy()
    return df.sort_values(["station", "time_rel_s"])


def plot_selection(df: pd.DataFrame, roi_df: pd.DataFrame, out: Path, mode: str, overlay_roi: bool) -> None:
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

    if overlay_roi and not roi_df.empty:
        roi_label_written = False
        for station, rdf in roi_df.groupby("station", sort=False):
            roi_label = "ROI trajectory-fit measurements" if not roi_label_written else None
            axes[0].scatter(
                rdf["time_rel_s"],
                rdf["range_km"],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
                alpha=0.65,
                label=roi_label,
                zorder=5,
            )
            axes[1].scatter(
                rdf["time_rel_s"],
                rdf["doppler_hz"],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
                alpha=0.65,
                zorder=5,
            )
            roi_label_written = True

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
    p.add_argument("--trajectory-fit", type=Path, default=DEFAULT_TRAJECTORY_FIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mode", choices=["all", "accepted", "real"], default="all")
    p.add_argument("--no-roi-overlay", action="store_true")
    args = p.parse_args()

    df = load_selection(args.selection, args.mode)
    roi_df = load_trajectory_observations(args.trajectory_fit)
    plot_selection(df, roi_df, args.out, args.mode, overlay_roi=not args.no_roi_overlay)
    print(args.out)
    print(f"{len(df)} plotted measurements")
    print(f"{len(roi_df)} ROI trajectory-fit measurements")


if __name__ == "__main__":
    main()
