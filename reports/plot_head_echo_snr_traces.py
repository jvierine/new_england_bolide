#!/usr/bin/env python3
"""Plot selected head-echo matched-filter SNR traces for the three links."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVENT_T0 = 1780164380.0
DEFAULT_TRAJECTORY_JSON = Path("products/head_echo_trajectory_roi_fit/head_echo_trajectory.json")
FALLBACK_TRAJECTORY_JSON = Path("/Users/j/src/new_england_bolide/products/head_echo_trajectory_roi_fit/head_echo_trajectory.json")
DEFAULT_SUMMARY_CSV = Path("reports/head_echo_gaussian_fit_summary_v0p2p7_1coh_2incoh_zp16_sparse_dcmean.csv")
DEFAULT_OUT = Path("reports/figures/head_echo_snr_traces.png")
STATION_LABELS = {
    "beverly": "Beverly",
    "clinton": "Clinton",
    "pomfret": "Pomfret",
}
STATION_COLORS = {
    "beverly": "#2f70b7",
    "clinton": "#d95f02",
    "pomfret": "#1b9e77",
}


def resolve_default_input(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_TRAJECTORY_JSON and FALLBACK_TRAJECTORY_JSON.exists():
        return FALLBACK_TRAJECTORY_JSON
    return path


def load_observations(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations = payload.get("observations", [])
    by_station: dict[str, list[dict]] = {}
    for obs in observations:
        station = str(obs.get("station", ""))
        if station not in STATION_LABELS:
            continue
        if "unix_time" not in obs or "snr_db" not in obs:
            continue
        by_station.setdefault(station, []).append(obs)
    for rows in by_station.values():
        rows.sort(key=lambda row: float(row["unix_time"]))
    return by_station


def load_gaussian_summary(path: Path, event_t0: float) -> dict[str, list[dict]]:
    by_station: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            station = str(row.get("station", ""))
            if station not in STATION_LABELS:
                continue
            if row.get("success", "").lower() != "true":
                continue
            if not row.get("snr_db"):
                continue
            time_unix = row.get("time_unix")
            if time_unix:
                unix_time = float(time_unix)
            elif row.get("time_rel_s"):
                unix_time = event_t0 + float(row["time_rel_s"])
            else:
                continue
            by_station.setdefault(station, []).append(
                {
                    "unix_time": unix_time,
                    "snr_db": float(row["snr_db"]),
                }
            )
    for rows in by_station.values():
        rows.sort(key=lambda row: float(row["unix_time"]))
    return by_station


def plot_snr_traces(observations: dict[str, list[dict]], out: Path, event_t0: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    for station in ("beverly", "clinton", "pomfret"):
        rows = observations.get(station, [])
        if not rows:
            continue
        t_rel = np.array([float(row["unix_time"]) - event_t0 for row in rows], dtype=float)
        snr_db = np.array([float(row["snr_db"]) for row in rows], dtype=float)
        ax.plot(
            t_rel,
            snr_db,
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            color=STATION_COLORS[station],
            label=f"{STATION_LABELS[station]} ({len(rows)})",
        )
    ax.set_xlabel("Seconds after 18:06:20 UTC")
    ax.set_ylabel("Matched-filter SNR (dB)")
    ax.set_title("Selected head-echo SNR by receiving link")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(loc="best", frameon=True, framealpha=0.94)
    ax.set_ylim(bottom=0.0)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY_JSON)
    parser.add_argument("--event-t0", type=float, default=EVENT_T0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_csv.exists():
        observations = load_gaussian_summary(args.summary_csv, args.event_t0)
        source = args.summary_csv
    else:
        trajectory_json = resolve_default_input(args.trajectory_json)
        observations = load_observations(trajectory_json)
        source = trajectory_json
    if not observations:
        raise SystemExit(f"No head-echo SNR observations found in {source}")
    plot_snr_traces(observations, args.out, args.event_t0)
    for station, rows in observations.items():
        print(f"{station}: {len(rows)} samples")
    print(f"Read {source}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
