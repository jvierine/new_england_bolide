#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from station_config import short_station_name


DEFAULT_SUMMARY = "products/specular_trail_positions/specular_trail_position_summary.json"
DEFAULT_OUT = "reports/figures/trail_aod_sky_scatter.png"


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
        }
    )


def plot_trail_aod(summary_path: Path, out_path: Path) -> Path:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["positions"]
    if not rows:
        raise SystemExit(f"No trail AoD positions in {summary_path}")

    stations = sorted({r["station"] for r in rows})
    colors = dict(zip(stations, plt.cm.tab10(np.linspace(0.0, 1.0, len(stations)))))

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    for station in stations:
        rr = [r for r in rows if r["station"] == station]
        az = np.array([r["aod_az_deg"] for r in rr], dtype=float)
        el = np.array([r["aod_el_deg"] for r in rr], dtype=float)
        snr = np.array([r["snr_db"] for r in rr], dtype=float)
        sizes = 22.0 + 3.0 * np.clip(snr - 8.0, 0.0, 10.0)
        ax.scatter(
            az,
            el,
            s=sizes,
            color=colors[station],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=f"{short_station_name(station)} ({len(rr)})",
        )

    ax.set_xlabel("Transmit AoD azimuth (deg)")
    ax.set_ylabel("Transmit AoD elevation (deg)")
    ax.set_title("Trail Echo Transmit AoD Sky Map")
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True, color="0.82", linewidth=0.55)
    ax.legend(loc="best", framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trail AoD azimuth/elevation sky scatter by receive station.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    set_plot_style()
    out = plot_trail_aod(Path(args.summary), Path(args.out))
    print(out)


if __name__ == "__main__":
    main()
