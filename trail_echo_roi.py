#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml
from estimate_head_echo_trajectory import model_state_series, predict_ranges, station_geometry
from station_config import short_station_name


DATA_DIR = ROOT / "2026_05_30_bolide"
DEFAULT_MERGED_DIR = "products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_TRAJECTORY_JSON = "products/head_echo_trajectory_roi_fit/head_echo_trajectory.json"
DEFAULT_OUT = "annotations/trail_echo_rois.json"
DEFAULT_PLOT = "reports/figures/trail_echo_roi.png"
DEFAULT_POINTS = "products/trail_echo_roi/trail_echo_roi_points.json"
ALTITUDE_TICKS_KM = np.array([85.0, 80.0, 75.0])
REPORT_RANGE_WINDOWS_KM = {
    "beverly": (180.0, 275.0),
    "clinton": (220.0, 280.0),
    "pomfret": (350.0, 420.0),
    "siena": (400.0, 500.0),
    "amherstnh": (200.0, 350.0),
}


def iso_utc(t):
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


def load_merged(merged_dir, station):
    path = Path(merged_dir) / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(path, "r") as h:
        sr = float(h["sample_rate"][()])
        return {
            "source_file": str(path),
            "station": h["station"][()].decode(),
            "t_unix": h["t_sample"][()] / sr,
            "range_km": h["range_km"][()],
            "P_rti": h["P_rti"][()] if "P_rti" in h else h["P"][()],
            "D_rti": h["D_rti"][()] if "D_rti" in h else h["D"][()],
            "N": h["N"][()],
        }


def trail_mask(data, snr_threshold_db, max_abs_doppler_hz, range_window_km):
    snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
    r = data["range_km"]
    r0, r1 = range_window_km
    keep_r = (r >= min(r0, r1)) & (r <= max(r0, r1))
    mask = (snr_db >= snr_threshold_db) & (np.abs(data["D_rti"]) < max_abs_doppler_hz) & keep_r[None, :]
    return mask, snr_db


def build_station_roi(data, snr_threshold_db, max_abs_doppler_hz, range_window_km):
    mask, snr_db = trail_mask(data, snr_threshold_db, max_abs_doppler_hz, range_window_km)
    ti, ri = np.where(mask)
    if len(ti) == 0:
        return None, []
    t = data["t_unix"]
    r = data["range_km"]
    points = []
    for i, j in zip(ti, ri):
        points.append(
            {
                "station": data["station"],
                "unix_time": float(t[i]),
                "utc": iso_utc(t[i]),
                "range_km": float(r[j]),
                "time_index": int(i),
                "range_index": int(j),
                "snr_db": float(snr_db[i, j]),
                "doppler_hz": float(data["D_rti"][i, j]),
            }
        )

    time_counts = {}
    for i in ti:
        time_counts[str(int(i))] = int(np.sum(ti == i))

    roi = {
        "has_trail_echo": True,
        "source_file": data["source_file"],
        "selection": {
            "snr_threshold_db": float(snr_threshold_db),
            "max_abs_doppler_hz": float(max_abs_doppler_hz),
            "range_window_km": [float(range_window_km[0]), float(range_window_km[1])],
            "multiple_range_gates_per_time": True,
        },
        "time_range": {
            "start_unix": float(np.min(t[ti])),
            "end_unix": float(np.max(t[ti])),
            "start_utc": iso_utc(np.min(t[ti])),
            "end_utc": iso_utc(np.max(t[ti])),
        },
        "range_km": {
            "min": float(np.min(r[ri])),
            "max": float(np.max(r[ri])),
        },
        "summary": {
            "n_cells": int(len(ti)),
            "n_time_bins": int(len(np.unique(ti))),
            "n_range_bins": int(len(np.unique(ri))),
            "range_gates_per_time": {
                "min": int(min(time_counts.values())),
                "median": float(np.median(list(time_counts.values()))),
                "max": int(max(time_counts.values())),
            },
            "snr_db": {
                "min": float(np.min(snr_db[ti, ri])),
                "median": float(np.median(snr_db[ti, ri])),
                "p95": float(np.percentile(snr_db[ti, ri], 95.0)),
                "max": float(np.max(snr_db[ti, ri])),
            },
            "abs_doppler_hz": {
                "median": float(np.median(np.abs(data["D_rti"][ti, ri]))),
                "max": float(np.max(np.abs(data["D_rti"][ti, ri]))),
            },
        },
    }
    return roi, points


def load_trajectory_altitude_ticks(stations, trajectory_json):
    trajectory_json = Path(trajectory_json)
    if not trajectory_json.is_absolute():
        trajectory_json = ROOT / trajectory_json
    if not trajectory_json.exists():
        return {}
    fit = json.loads(trajectory_json.read_text(encoding="utf-8"))
    obs = fit.get("observations", [])
    if not obs:
        return {}

    geom = station_geometry(DATA_DIR, stations)
    tx_site = load_station_yaml(DATA_DIR, fit["used_stations"][0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    t_grid = np.linspace(
        min(float(o["unix_time"]) for o in obs),
        max(float(o["unix_time"]) for o in obs),
        400,
    )
    state = model_state_series(params, t_grid, fit["t_ref_unix"], tx0, rot)
    alt_km = state["lat_lon_alt"][:, 2] / 1000.0
    order = np.argsort(alt_km)
    alt_sorted = alt_km[order]
    _, unique_idx = np.unique(np.round(alt_sorted, 6), return_index=True)
    unique_idx = np.sort(unique_idx)
    alt_unique = alt_sorted[unique_idx]

    ticks_by_station = {}
    for station in stations:
        model_obs = [{"station": station, "unix_time": float(t)} for t in t_grid]
        range_km = predict_ranges(params, model_obs, geom, fit["t_ref_unix"], tx0, rot) / 1000.0
        range_unique = range_km[order][unique_idx]
        station_ticks = []
        for alt_tick in ALTITUDE_TICKS_KM:
            if alt_tick < alt_unique[0] or alt_tick > alt_unique[-1]:
                continue
            station_ticks.append((float(np.interp(alt_tick, alt_unique, range_unique)), float(alt_tick)))
        ticks_by_station[station] = station_ticks
    return ticks_by_station


def add_manual_altitude_axis(ax, station, ticks_by_station):
    r0, r1 = ax.get_ylim()
    ticks = [
        (range_km, alt_km)
        for range_km, alt_km in ticks_by_station.get(station, [])
        if min(r0, r1) <= range_km <= max(r0, r1)
    ]
    if not ticks:
        return
    alt_ax = ax.twinx()
    alt_ax.set_ylim(ax.get_ylim())
    alt_ax.set_yticks([range_km for range_km, _ in ticks])
    alt_ax.set_yticklabels([])
    alt_ax.tick_params(axis="y", colors="0.25", labelsize=10, length=3, width=0.8, pad=3)
    alt_ax.spines["right"].set_color("0.45")
    alt_ax.grid(False)

    fig = ax.figure
    fig.canvas.draw()
    ticks_sorted = sorted(ticks)
    y_pixels = [ax.transData.transform((0.0, range_km))[1] for range_km, _ in ticks_sorted]
    adjusted = []
    min_sep_px = 15.0
    for y in y_pixels:
        adjusted.append(max(y, adjusted[-1] + min_sep_px) if adjusted else y)
    overflow = adjusted[-1] - ax.bbox.y1
    if overflow > 0:
        adjusted = [y - overflow for y in adjusted]
    underflow = ax.bbox.y0 - adjusted[0]
    if underflow > 0:
        adjusted = [y + underflow for y in adjusted]
    for (range_km, alt_km), y0, y1 in zip(ticks_sorted, y_pixels, adjusted, strict=True):
        dy_points = (y1 - y0) * 72.0 / fig.dpi
        ax.annotate(
            f"{alt_km:.0f}",
            xy=(1.0, range_km),
            xycoords=ax.get_yaxis_transform(),
            xytext=(10, dy_points),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="0.25",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=0),
        )


def draw_roi_figure(datasets, rois, out_png, snr_threshold_db, max_abs_doppler_hz, trajectory_json):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, max(2.15 * len(datasets), 8.5)), sharex=True, constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([axes])
    ticks_by_station = load_trajectory_altitude_ticks([data["station"] for data in datasets], trajectory_json)
    t_ref = min(float(data["t_unix"][0]) for data in datasets)
    t_max = max(float(data["t_unix"][-1]) for data in datasets) - t_ref
    snr_all = []
    for data in datasets:
        snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
        snr_all.append(10.0 * np.log10(np.maximum(snr, 1e-12)).ravel())
    snr_all = np.concatenate(snr_all)
    vmin = max(-5.0, float(np.nanpercentile(snr_all, 5)))
    vmax = min(30.0, float(np.nanpercentile(snr_all, 99.7)))

    for row, data in enumerate(datasets):
        station = data["station"]
        t_rel = data["t_unix"] - t_ref
        r = data["range_km"]
        snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
        mask, _ = trail_mask(data, snr_threshold_db, max_abs_doppler_hz, REPORT_RANGE_WINDOWS_KM[station])

        ax = axes[row]
        ax.pcolormesh(t_rel, r, snr_db.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.contour(t_rel, r, mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_ylabel(f"{short_station_name(station)}\nrange (km)")

        roi = rois.get(station)
        if roi:
            y0 = roi["range_km"]["min"]
            y1 = roi["range_km"]["max"]
            ax.set_ylim(max(float(r[0]), y0 - 8.0), min(float(r[-1]), y1 + 8.0))
        add_manual_altitude_axis(ax, station, ticks_by_station)
        ax.grid(True, color="0.85", lw=0.4)
        ax.set_xlim(0.0, t_max)
    axes[-1].set_xlabel(f"Seconds after {iso_utc(t_ref)}")

    fig.suptitle(f"Trail Echo ROI: SNR >= {snr_threshold_db:g} dB And |Doppler| < {max_abs_doppler_hz:g} Hz")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Build near-zero-Doppler trail echo ROI boxes from merged RDMF products.")
    p.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--stations", nargs="+", default=["amherstnh", "siena", "pomfret", "clinton", "beverly"])
    p.add_argument("--snr-threshold-db", type=float, default=6.0)
    p.add_argument("--max-abs-doppler-hz", type=float, default=100.0)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--points-out", default=DEFAULT_POINTS)
    p.add_argument("--plot", default=DEFAULT_PLOT)
    p.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY_JSON)
    args = p.parse_args()

    annotations = {
        "schema": "new_england_bolide.trail_echo_roi.v1",
        "description": "Near-zero-Doppler trail echo ROI labels from RTI products; all cells in the ROI may be used.",
        "stations": {},
    }
    all_points = []
    datasets = []
    for station in args.stations:
        data = load_merged(args.merged_dir, station)
        datasets.append(data)
        roi, points = build_station_roi(
            data,
            snr_threshold_db=args.snr_threshold_db,
            max_abs_doppler_hz=args.max_abs_doppler_hz,
            range_window_km=REPORT_RANGE_WINDOWS_KM[station],
        )
        if roi is None:
            annotations["stations"][station] = {
                "has_trail_echo": False,
                "source_file": data["source_file"],
                "selection": {
                    "snr_threshold_db": args.snr_threshold_db,
                    "max_abs_doppler_hz": args.max_abs_doppler_hz,
                    "range_window_km": list(REPORT_RANGE_WINDOWS_KM[station]),
                },
            }
            print(f"{short_station_name(station)}: no trail ROI cells")
            continue
        annotations["stations"][station] = roi
        all_points.extend(points)
        s = roi["summary"]
        print(
            f"{short_station_name(station)}: ROI {roi['time_range']['start_utc']} to {roi['time_range']['end_utc']}, "
            f"{roi['range_km']['min']:.1f}-{roi['range_km']['max']:.1f} km, "
            f"{s['n_cells']} cells, {s['range_gates_per_time']['median']:.1f} median gates/time"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(annotations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    points_out = Path(args.points_out)
    points_out.parent.mkdir(parents=True, exist_ok=True)
    points_payload = {
        "schema": "new_england_bolide.trail_echo_roi_points.v1",
        "source_annotations": str(out),
        "n_points": len(all_points),
        "points": all_points,
    }
    points_out.write_text(json.dumps(points_payload, indent=2) + "\n", encoding="utf-8")
    draw_roi_figure(
        datasets,
        annotations["stations"],
        args.plot,
        args.snr_threshold_db,
        args.max_abs_doppler_hz,
        args.trajectory_json,
    )
    print(f"Wrote {out}")
    print(f"Wrote {points_out}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
