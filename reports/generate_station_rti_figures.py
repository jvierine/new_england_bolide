#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml
from estimate_head_echo_trajectory import model_state_series, predict_ranges, station_geometry
from station_config import short_station_name

DATA_DIR = ROOT / "2026_05_30_bolide"
MERGED_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
TRAJECTORY_JSON = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
FIG_DIR = ROOT / "reports" / "figures"
STATION_RANGES_KM = [
    ("beverly", 180.0, 275.0),
    ("clinton", 220.0, 280.0),
    ("pomfret", 350.0, 420.0),
    ("siena", 400.0, 500.0),
    ("amherstnh", 200.0, 350.0),
]
PLOT_START_UNIX = 1780164378.0
HEAD_PLOT_START_UNIX = 1780164379.0
HEAD_PLOT_END_UNIX = 1780164382.0
HEAD_DOPPLER_SNR_MASK_DB = 7.5
ALTITUDE_TICKS_KM = np.array([90.0, 80.0, 70.0, 60.0, 50.0, 40.0])


def unix_to_mpl(t: np.ndarray) -> np.ndarray:
    ns = np.asarray(np.asarray(t, dtype=float) * 1e9, dtype="datetime64[ns]")
    return mdates.date2num(ns.astype("datetime64[us]").astype(object))


def load_station(station: str, rmin_km: float, rmax_km: float) -> dict:
    path = MERGED_DIR / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(path, "r") as h:
        p_rti = h["P_rti"][()] if "P_rti" in h else h["P"][()]
        p_head = h["P_head"][()] if "P_head" in h else h["P"][()]
        d_head = h["D_head"][()] if "D_head" in h else h["D"][()]
        n = h["N"][()]
        t = h["t_sample"][()] / float(h["sample_rate"][()])
        if "range_km" in h:
            ranges = h["range_km"][()]
        else:
            ranges = h["range_idx"][()] * 3.0
    range_keep = (ranges >= rmin_km) & (ranges <= rmax_km)
    time_keep = t >= PLOT_START_UNIX
    n_sel = np.maximum(n[time_keep][:, range_keep], 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(p_rti[time_keep][:, range_keep] / n_sel, 1e-12))
    head_snr_db = 10.0 * np.log10(np.maximum(p_head[time_keep][:, range_keep] / n_sel, 1e-12))
    head_doppler_hz = d_head[time_keep][:, range_keep]
    return {
        "station": station,
        "t": t[time_keep],
        "range_km": ranges[range_keep],
        "snr_db": snr_db,
        "head_snr_db": head_snr_db,
        "head_doppler_hz": head_doppler_hz,
        "rmin_km": rmin_km,
        "rmax_km": rmax_km,
    }


def load_trajectory_altitude_ticks(stations: list[str], t_min_unix: float, t_max_unix: float) -> dict[str, list[tuple[float, float]]]:
    if not TRAJECTORY_JSON.exists():
        return {}
    fit = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8"))
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
    t_grid = np.linspace(float(t_min_unix), float(t_max_unix), 800)
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


def add_manual_altitude_axis(ax, data: dict, ticks_by_station: dict[str, list[tuple[float, float]]]) -> None:
    ticks = [
        (range_km, alt_km)
        for range_km, alt_km in ticks_by_station.get(data["station"], [])
        if data["rmin_km"] <= range_km <= data["rmax_km"]
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


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 18,
            "axes.linewidth": 1.0,
        }
    )


def head_window(data: dict) -> dict:
    keep = (data["t"] >= HEAD_PLOT_START_UNIX) & (data["t"] <= HEAD_PLOT_END_UNIX)
    out = data.copy()
    out["t"] = data["t"][keep]
    out["head_snr_db"] = data["head_snr_db"][keep]
    out["head_doppler_hz"] = data["head_doppler_hz"][keep]
    return out


def format_head_time_axis(axes: np.ndarray) -> None:
    xlim = unix_to_mpl(np.array([HEAD_PLOT_START_UNIX, HEAD_PLOT_END_UNIX]))
    for ax in axes:
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def plot_snr(datasets: list[dict]) -> Path:
    ticks_by_station = load_trajectory_altitude_ticks(
        [d["station"] for d in datasets],
        min(float(d["t"][0]) for d in datasets),
        max(float(d["t"][-1]) for d in datasets),
    )
    all_snr = np.concatenate([d["snr_db"].ravel() for d in datasets])
    vmin = max(-5.0, float(np.nanpercentile(all_snr, 2.0)))
    vmax = min(28.0, float(np.nanpercentile(all_snr, 99.7)))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    mesh = None
    for ax, data in zip(axes, datasets):
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            data["snr_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{short_station_name(data['station'])}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        add_manual_altitude_axis(ax, data, ticks_by_station)
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle("Range-time intensity by station")
    fig.colorbar(mesh, ax=axes, label="SNR (dB)", shrink=0.93)
    out = FIG_DIR / "station_range_time_snr.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_head_snr(datasets: list[dict]) -> Path:
    datasets = [head_window(d) for d in datasets]
    all_snr = np.concatenate([d["head_snr_db"].ravel() for d in datasets])
    vmin = max(3.5, float(np.nanpercentile(all_snr, 2.0)))
    vmax = min(18.0, max(12.0, float(np.nanpercentile(all_snr, 99.99))))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    mesh = None
    for ax, data in zip(axes, datasets):
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            data["head_snr_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{short_station_name(data['station'])}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
    format_head_time_axis(axes)
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle("Head-echo Doppler-gated range-time intensity by station")
    fig.colorbar(mesh, ax=axes, label="Head-gated SNR (dB)", shrink=0.93)
    out = FIG_DIR / "station_range_time_head_snr.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_head_doppler(datasets: list[dict]) -> Path:
    datasets = [head_window(d) for d in datasets]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("0.88")
    mesh = None
    for ax, data in zip(axes, datasets):
        doppler = data["head_doppler_hz"].astype(float).copy()
        doppler[data["head_snr_db"] < HEAD_DOPPLER_SNR_MASK_DB] = np.nan
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            doppler.T,
            shading="auto",
            cmap=cmap,
            vmin=-4000.0,
            vmax=0.0,
        )
        ax.set_ylabel(f"{short_station_name(data['station'])}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
    format_head_time_axis(axes)
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle("Head-echo Doppler shift by station")
    fig.colorbar(mesh, ax=axes, label="Head-gated Doppler shift (Hz)", shrink=0.93)
    out = FIG_DIR / "station_range_time_head_doppler.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    set_plot_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [load_station(station, rmin, rmax) for station, rmin, rmax in STATION_RANGES_KM]
    print(plot_snr(datasets))
    print(plot_head_snr(datasets))
    print(plot_head_doppler(datasets))


if __name__ == "__main__":
    main()
