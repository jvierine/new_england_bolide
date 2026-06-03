#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from station_config import STATION_ORDER, short_station_name


DEFAULT_HEAD_SUMMARY = "products/aod_position_from_merged/aod_position_summary.json"
DEFAULT_TRAIL_SUMMARY = "products/specular_trail_positions/specular_trail_position_summary.json"
DEFAULT_GEO_HEIGHT_OUT = "reports/figures/trail_echo_geographic_height.png"
DEFAULT_AOD_OUT = "reports/figures/head_trail_aod_launch_angles.png"
STATION_COLORS = {
    "beverly": "#1f77b4",
    "clinton": "#d62728",
    "pomfret": "#2ca02c",
    "siena": "#9467bd",
    "amherstnh": "#ff7f0e",
}
STATION_MARKERS = {
    "beverly": "o",
    "clinton": "s",
    "pomfret": "^",
    "siena": "D",
    "amherstnh": "P",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
        }
    )


def load_positions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("positions", [])
    if not rows:
        raise SystemExit(f"No positions found in {path}")
    return rows


def station_colors(rows: list[dict]) -> dict[str, tuple[float, float, float, float]]:
    stations = [station for station in STATION_ORDER if any(r["station"] == station for r in rows)]
    stations.extend(sorted({r["station"] for r in rows} - set(stations)))
    fallback = plt.cm.Dark2(np.linspace(0.0, 1.0, max(len(stations), 1)))
    return {station: STATION_COLORS.get(station, fallback[i]) for i, station in enumerate(stations)}


def station_marker(station: str, index: int = 0) -> str:
    fallback = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    return STATION_MARKERS.get(station, fallback[index % len(fallback)])


def snr_sizes(rows: list[dict]) -> np.ndarray:
    snr = np.array([float(r.get("snr_db", 0.0)) for r in rows], dtype=float)
    return 22.0 + 3.0 * np.clip(snr - 8.0, 0.0, 12.0)


def plot_trail_axis_height(
    ax,
    rows: list[dict],
    coord_key: str,
    xlabel: str,
    colors: dict[str, tuple[float, float, float, float]],
    head_rows: list[dict] | None = None,
) -> None:
    if head_rows:
        head_coord = np.array([float(r[coord_key]) for r in head_rows], dtype=float)
        head_alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in head_rows], dtype=float)
        ax.scatter(
            head_coord,
            head_alt_km,
            s=18.0,
            marker=".",
            color="0.32",
            alpha=0.55,
            linewidth=0.0,
            zorder=1,
            label="Head echo",
        )

    for si, station in enumerate(colors):
        rr = [r for r in rows if r["station"] == station]
        if not rr:
            continue
        coord = np.array([float(r[coord_key]) for r in rr], dtype=float)
        alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in rr], dtype=float)
        ax.scatter(
            coord,
            alt_km,
            s=snr_sizes(rr),
            marker=station_marker(station, si),
            color=colors[station],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
            label=f"{short_station_name(station)} ({len(rr)})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Height (km)")
    ax.grid(True, color="0.84", linewidth=0.55)


def plot_trail_geographic_height(rows: list[dict], out_path: Path, head_rows: list[dict] | None = None) -> Path:
    colors = station_colors(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True, sharey=True)
    plot_trail_axis_height(axes[0], rows, "lon_deg", "Longitude (deg)", colors, head_rows=head_rows)
    plot_trail_axis_height(axes[1], rows, "lat_deg", "Latitude (deg)", colors, head_rows=head_rows)
    axes[0].set_title("Longitude vs. Height")
    axes[1].set_title("Latitude vs. Height")
    axes[0].set_xlim(-70.8, -70.4)
    axes[1].set_xlim(right=42.1)
    axes[1].legend(loc="best", framealpha=0.92, ncols=1)
    fig.suptitle("Interferometric Non-Specular Trail Echo Positions", fontsize=17)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def scatter_aod_panel(
    ax,
    rows: list[dict],
    title: str,
    colors: dict[str, tuple[float, float, float, float]],
    background_rows: list[dict] | None = None,
    background_label: str | None = None,
) -> None:
    if background_rows:
        bg_az = np.array([float(r["aod_az_deg"]) for r in background_rows], dtype=float)
        bg_el = np.array([float(r["aod_el_deg"]) for r in background_rows], dtype=float)
        ax.scatter(
            bg_az,
            bg_el,
            s=15.0,
            marker=".",
            color="0.32",
            alpha=0.55,
            linewidth=0.0,
            zorder=1,
            label=background_label,
        )

    for si, station in enumerate(colors):
        rr = [r for r in rows if r["station"] == station]
        if not rr:
            continue
        az = np.array([float(r["aod_az_deg"]) for r in rr], dtype=float)
        el = np.array([float(r["aod_el_deg"]) for r in rr], dtype=float)
        ax.scatter(
            az,
            el,
            s=snr_sizes(rr),
            marker=station_marker(station, si),
            color=colors[station],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
            label=f"{short_station_name(station)} ({len(rr)})",
        )
    ax.set_title(title)
    ax.set_xlabel("Transmit AoD azimuth (deg)")
    ax.set_ylabel("Transmit AoD elevation (deg)")
    ax.set_xlim(125.0, 145.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_xticks(np.arange(125, 146, 5))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True, color="0.84", linewidth=0.55)


def plot_head_trail_aod(head_rows: list[dict], trail_rows: list[dict], out_path: Path) -> Path:
    colors = station_colors(head_rows + trail_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True, sharex=True, sharey=True)
    scatter_aod_panel(
        axes[0],
        head_rows,
        "Head Echo Launch Directions",
        colors,
        background_rows=trail_rows,
        background_label="Trail echo",
    )
    scatter_aod_panel(
        axes[1],
        trail_rows,
        "Non-Specular Trail Echo Launch Directions",
        colors,
        background_rows=head_rows,
        background_label="Head echo",
    )
    axes[1].legend(loc="best", framealpha=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot interferometric head/trail AoD launch directions and trail longitude-height positions."
    )
    parser.add_argument("--head-summary", default=DEFAULT_HEAD_SUMMARY)
    parser.add_argument("--trail-summary", default=DEFAULT_TRAIL_SUMMARY)
    parser.add_argument("--geo-height-out", default=DEFAULT_GEO_HEIGHT_OUT)
    parser.add_argument("--aod-out", default=DEFAULT_AOD_OUT)
    args = parser.parse_args()

    set_plot_style()
    head_rows = load_positions(Path(args.head_summary))
    trail_rows = load_positions(Path(args.trail_summary))
    geo_height = plot_trail_geographic_height(trail_rows, Path(args.geo_height_out), head_rows=head_rows)
    aod = plot_head_trail_aod(head_rows, trail_rows, Path(args.aod_out))
    print(geo_height)
    print(aod)


if __name__ == "__main__":
    main()
