#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
DEFAULT_ALTITUDE_OVERLAY_PLOT = "reports/figures/trail_echo_roi_trail_altitude.png"
DEFAULT_POINTS = "products/trail_echo_roi/trail_echo_roi_points.json"
DEFAULT_TRAIL_SUMMARY = "products/specular_trail_positions/specular_trail_position_summary.json"
DEFAULT_HEIGHT_LABELS = "annotations/trail_height_annotations.json"
DEFAULT_HEAD_ROIS = "annotations/head_echo_rois.json"
ALTITUDE_TICKS_KM = np.array([85.0, 80.0, 75.0])
HEIGHT_LABELS_KM = (90.0, 72.0, 35.0, 10.0)
REPORT_RANGE_WINDOWS_KM = {
    "beverly": (180.0, 275.0),
    "clinton": (220.0, 280.0),
    "pomfret": (350.0, 420.0),
    "siena": (400.0, 500.0),
    "amherstnh": (200.0, 350.0),
}
REPORT_PLOT_START_REL_S = 4.0
REPORT_FONT_SCALE = 1.5


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


def load_head_echo_range_tracks(datasets, trajectory_json):
    trajectory_json = Path(trajectory_json)
    if not trajectory_json.is_absolute():
        trajectory_json = ROOT / trajectory_json
    if not trajectory_json.exists():
        return {}
    fit = json.loads(trajectory_json.read_text(encoding="utf-8"))
    if not fit.get("observations"):
        return {}

    stations = [data["station"] for data in datasets]
    geom = station_geometry(DATA_DIR, stations)
    tx_site = load_station_yaml(DATA_DIR, fit["used_stations"][0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]

    t_all = np.unique(np.concatenate([data["t_unix"] for data in datasets]))
    state = model_state_series(params, t_all, fit["t_ref_unix"], tx0, rot)
    alt_all_km = state["lat_lon_alt"][:, 2] / 1000.0

    tracks = {}
    for data in datasets:
        station = data["station"]
        model_obs = [{"station": station, "unix_time": float(t)} for t in data["t_unix"]]
        range_km = predict_ranges(params, model_obs, geom, fit["t_ref_unix"], tx0, rot) / 1000.0
        tracks[station] = {
            "range_km": range_km,
            "alt_km": np.interp(data["t_unix"], t_all, alt_all_km),
        }
    return tracks


def height_label_positions(t_rel, track):
    alt_km = np.asarray(track["alt_km"], dtype=float)
    range_km = np.asarray(track["range_km"], dtype=float)
    order = np.argsort(alt_km)
    alt_sorted = alt_km[order]
    unique_alt, unique_idx = np.unique(np.round(alt_sorted, 6), return_index=True)
    unique_order = order[np.sort(unique_idx)]
    alt_unique = alt_km[unique_order]
    t_unique = np.asarray(t_rel, dtype=float)[unique_order]
    range_unique = range_km[unique_order]

    positions = []
    for height_km in HEIGHT_LABELS_KM:
        if height_km < np.nanmin(alt_unique) or height_km > np.nanmax(alt_unique):
            continue
        positions.append(
            {
                "height_km": height_km,
                "t_rel": float(np.interp(height_km, alt_unique, t_unique)),
                "range_km": float(np.interp(height_km, alt_unique, range_unique)),
            }
        )
    return positions


def local_snr_score(snr_db, t_rel, range_km, x, y, dx, dy, text_dx=0.0, text_dy=0.0):
    t_center = x + text_dx
    r_center = y + text_dy
    keep_t = (t_rel >= t_center - dx) & (t_rel <= t_center + dx)
    keep_r = (range_km >= r_center - dy) & (range_km <= r_center + dy)
    if not np.any(keep_t) or not np.any(keep_r):
        return 0.0
    return float(np.nanmax(snr_db[np.ix_(keep_t, keep_r)]))


def annotate_height_labels(ax, t_rel, range_km, snr_db, track, label_size):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    x_span = x_max - x_min
    x_margin = 0.55
    y_margin = 0.08 * y_span
    offsets = [
        (0.08 * x_span, 0.12 * y_span, "left"),
        (0.08 * x_span, -0.12 * y_span, "left"),
        (-0.08 * x_span, 0.12 * y_span, "right"),
        (-0.08 * x_span, -0.12 * y_span, "right"),
        (0.12 * x_span, 0.0, "left"),
        (-0.12 * x_span, 0.0, "right"),
    ]
    for label in height_label_positions(t_rel, track):
        x = label["t_rel"]
        y = label["range_km"]
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        best = None
        for dx_text, dy_text, ha in offsets:
            text_x = x + dx_text
            text_y = y + dy_text
            if not (x_min + x_margin <= text_x <= x_max - x_margin):
                continue
            if not (y_min + y_margin <= text_y <= y_max - y_margin):
                continue
            score = local_snr_score(snr_db, t_rel, range_km, x, y, 0.12, 4.0, dx_text, dy_text)
            candidate = (score, dx_text, dy_text, ha)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            continue
        _, dx_text, dy_text, ha = best
        height_km = label["height_km"]
        label_text = f"{height_km:.0f} km height"
        ax.annotate(
            label_text,
            xy=(x, y),
            xytext=(x + dx_text, y + dy_text),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=label_size,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="none", alpha=0.35),
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.45, shrinkA=0, shrinkB=0),
            zorder=7,
        )


def load_height_label_annotations(path):
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    for station, rows in payload.get("stations", {}).items():
        labels[station] = {float(row["height_km"]): row for row in rows}
    return labels


def load_head_roi_annotations(path):
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("stations", {})


def draw_head_roi_box(ax, head_roi, t_ref):
    if not head_roi or not head_roi.get("has_head_echo"):
        return
    t0 = float(head_roi["time_range"]["start_unix"]) - t_ref
    t1 = float(head_roi["time_range"]["end_unix"]) - t_ref
    r0 = float(head_roi["range_km"]["min"])
    r1 = float(head_roi["range_km"]["max"])
    rect = Rectangle(
        (min(t0, t1), min(r0, r1)),
        abs(t1 - t0),
        abs(r1 - r0),
        facecolor="cyan",
        edgecolor="cyan",
        linewidth=1.1,
        alpha=0.5,
        zorder=4,
    )
    ax.add_patch(rect)


def annotate_manual_height_labels(ax, track, station_labels, label_size):
    for label in height_label_positions(np.asarray(track["t_rel"], dtype=float), track):
        height_km = label["height_km"]
        placement = station_labels.get(height_km)
        if placement is None:
            continue
        text_x = float(placement["text_t_rel"])
        text_y = float(placement["text_range_km"])
        ax.annotate(
            f"{height_km:.0f} km height",
            xy=(label["t_rel"], label["range_km"]),
            xytext=(text_x, text_y),
            textcoords="data",
            ha=placement.get("ha", "center"),
            va=placement.get("va", "center"),
            fontsize=label_size,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="none", alpha=0.35),
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.45, shrinkA=0, shrinkB=0),
            zorder=7,
        )


def load_trail_altitude_points(summary_json):
    path = Path(summary_json)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        row
        for row in payload.get("positions", [])
        if {"station", "unix_time", "range_km", "alt_m"}.issubset(row)
    ]


def draw_roi_figure(
    datasets,
    rois,
    out_png,
    snr_threshold_db,
    max_abs_doppler_hz,
    trajectory_json,
    altitude_points=None,
    height_labels=None,
    head_rois=None,
):
    axis_label_size = 10 * REPORT_FONT_SCALE
    tick_label_size = 10 * REPORT_FONT_SCALE
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, max(2.15 * len(datasets), 8.5)), sharex=True, constrained_layout=True)
    if len(datasets) == 1:
        axes = np.array([axes])
    t_ref = min(float(data["t_unix"][0]) for data in datasets)
    head_echo_range_tracks = load_head_echo_range_tracks(datasets, trajectory_json)
    for data in datasets:
        station = data["station"]
        if station in head_echo_range_tracks:
            head_echo_range_tracks[station]["t_rel"] = data["t_unix"] - t_ref
    height_labels = height_labels or {}
    head_rois = head_rois or {}
    t_max = max(float(data["t_unix"][-1]) for data in datasets) - t_ref
    snr_all = []
    for data in datasets:
        snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
        snr_all.append(10.0 * np.log10(np.maximum(snr, 1e-12)).ravel())
    snr_all = np.concatenate(snr_all)
    vmin = max(-5.0, float(np.nanpercentile(snr_all, 5)))
    vmax = min(30.0, float(np.nanpercentile(snr_all, 99.7)))
    altitude_points = altitude_points or []
    altitude_km = np.array([row["alt_m"] / 1000.0 for row in altitude_points], dtype=float)
    altitude_vmin = float(np.nanpercentile(altitude_km, 2.0)) if altitude_km.size else 0.0
    altitude_vmax = float(np.nanpercentile(altitude_km, 98.0)) if altitude_km.size else 1.0
    mesh = None
    scatter = None

    for row, data in enumerate(datasets):
        station = data["station"]
        t_rel = data["t_unix"] - t_ref
        r = data["range_km"]
        snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))

        ax = axes[row]
        mesh = ax.pcolormesh(t_rel, r, snr_db.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        if station in head_echo_range_tracks:
            track = head_echo_range_tracks[station]
            plotted_range = np.asarray(track["range_km"], dtype=float).copy()
            plotted_range[np.asarray(track["alt_km"], dtype=float) < 0.0] = np.nan
            ax.plot(
                t_rel,
                plotted_range,
                color="white",
                linestyle="--",
                linewidth=1.2,
                alpha=0.3,
                zorder=5,
            )
        draw_head_roi_box(ax, head_rois.get(station), t_ref)
        ax.set_ylabel(f"{short_station_name(station)}\nrange (km)", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)

        roi = rois.get(station)
        if roi:
            y0 = roi["range_km"]["min"]
            y1 = roi["range_km"]["max"]
            ax.set_ylim(max(float(r[0]), y0 - 8.0), min(float(r[-1]), y1 + 8.0))
        station_points = [
            point
            for point in altitude_points
            if point["station"] == station
            and float(data["t_unix"][0]) <= float(point["unix_time"]) <= float(data["t_unix"][-1])
        ]
        if station_points:
            scatter = ax.scatter(
                [float(point["unix_time"]) - t_ref for point in station_points],
                [float(point["range_km"]) for point in station_points],
                c=[float(point["alt_m"]) / 1000.0 for point in station_points],
                cmap="turbo",
                vmin=altitude_vmin,
                vmax=altitude_vmax,
                s=34,
                marker="o",
                edgecolors="black",
                linewidths=0.45,
                zorder=6,
            )
        ax.grid(True, color="0.85", lw=0.4)
        ax.set_xlim(REPORT_PLOT_START_REL_S, t_max)
        if station in head_echo_range_tracks:
            if station in height_labels:
                annotate_manual_height_labels(
                    ax,
                    head_echo_range_tracks[station],
                    height_labels[station],
                    8.0 * REPORT_FONT_SCALE,
                )
            else:
                annotate_height_labels(ax, t_rel, r, snr_db, head_echo_range_tracks[station], 8.0 * REPORT_FONT_SCALE)
    axes[-1].set_xlabel(f"Seconds after {iso_utc(t_ref)}", fontsize=axis_label_size)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, label="SNR (dB)", shrink=0.93)
        cbar.ax.tick_params(labelsize=tick_label_size)
        cbar.set_label("SNR (dB)", fontsize=axis_label_size)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, label="Trail echo altitude (km)", shrink=0.93)
        cbar.ax.tick_params(labelsize=tick_label_size)
        cbar.set_label("Trail echo altitude (km)", fontsize=axis_label_size)
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
    p.add_argument("--altitude-overlay-plot", default=None)
    p.add_argument("--trail-summary", default=DEFAULT_TRAIL_SUMMARY)
    p.add_argument("--height-labels", default=DEFAULT_HEIGHT_LABELS)
    p.add_argument("--head-rois", default=DEFAULT_HEAD_ROIS)
    p.add_argument("--overlay-only", action="store_true")
    p.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY_JSON)
    args = p.parse_args()

    if args.overlay_only:
        if not args.altitude_overlay_plot:
            raise SystemExit("--overlay-only requires --altitude-overlay-plot")
        annotations = json.loads(Path(args.out).read_text(encoding="utf-8"))
        datasets = [load_merged(args.merged_dir, station) for station in args.stations]
        draw_roi_figure(
            datasets,
            annotations["stations"],
            args.altitude_overlay_plot,
            args.snr_threshold_db,
            args.max_abs_doppler_hz,
            args.trajectory_json,
            altitude_points=load_trail_altitude_points(args.trail_summary),
            height_labels=load_height_label_annotations(args.height_labels),
            head_rois=load_head_roi_annotations(args.head_rois),
        )
        print(f"Wrote {args.altitude_overlay_plot}")
        return

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
        height_labels=load_height_label_annotations(args.height_labels),
        head_rois=load_head_roi_annotations(args.head_rois),
    )
    if args.altitude_overlay_plot:
        draw_roi_figure(
            datasets,
            annotations["stations"],
            args.altitude_overlay_plot,
            args.snr_threshold_db,
            args.max_abs_doppler_hz,
            args.trajectory_json,
            altitude_points=load_trail_altitude_points(args.trail_summary),
            height_labels=load_height_label_annotations(args.height_labels),
            head_rois=load_head_roi_annotations(args.head_rois),
        )
    print(f"Wrote {out}")
    print(f"Wrote {points_out}")
    print(f"Wrote {args.plot}")
    if args.altitude_overlay_plot:
        print(f"Wrote {args.altitude_overlay_plot}")


if __name__ == "__main__":
    main()
