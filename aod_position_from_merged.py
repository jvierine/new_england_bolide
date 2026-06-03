#!/usr/bin/env python3
import argparse
import json
import sys
import types
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml, regular_direction_grid
from constrained_aod_from_rdmf import (
    angular_distance_deg,
    estimate_one,
    pick_voltage,
    trajectory_position_enu,
    unit_to_az_el,
)
from estimate_head_echo_trajectory import ecef_to_geodetic_wgs84, sample_fit_parameters
from station_config import short_station_name as station_label


DEFAULT_MERGED_DIR = "products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_TRAJECTORY = "products/head_echo_trajectory_roi_fit/head_echo_trajectory.json"
DEFAULT_OUT_DIR = "products/aod_position_from_merged"
DEFAULT_REPORT_FIG = "reports/figures/head_echo_aod_tristatic_comparison.png"
DEFAULT_REPORT_TEX = "reports/aod_tristatic_comparison_summary.tex"


def load_rdmf_station(path):
    h = h5py.File(path, "r")
    sample_rate = float(h["sample_rate"][()])
    range_gate_width_m = float(h["range_resolution_m"][()]) / float(h["range_interp"][()])
    return {
        "handle": h,
        "time_unix": h["t_sample"][()] / sample_rate,
        "range_km": h["range_km"][()],
        "snr_db": 10.0 * np.log10(np.maximum(h["P"][()] / np.maximum(h["N"][()], 1e-12), 1e-12)),
        "range_gate_width_m": range_gate_width_m,
    }


def aod_delay_position(tx_ecef, rx_ecef, unit_enu, rot_enu_to_ecef, bistatic_range_m):
    unit_ecef = rot_enu_to_ecef @ np.asarray(unit_enu, dtype=float)
    unit_ecef /= np.linalg.norm(unit_ecef)
    baseline = np.asarray(rx_ecef, dtype=float) - np.asarray(tx_ecef, dtype=float)
    path = float(bistatic_range_m)
    denom = 2.0 * (path - float(np.dot(unit_ecef, baseline)))
    if denom <= 0.0:
        return None
    rho = (path * path - float(np.dot(baseline, baseline))) / denom
    if not np.isfinite(rho) or rho <= 0.0:
        return None
    pos_ecef = np.asarray(tx_ecef, dtype=float) + rho * unit_ecef
    residual = rho + np.linalg.norm(pos_ecef - rx_ecef) - path
    return pos_ecef, rho, float(residual)


def tx_angle_sigma_rad(site, phase_error_deg):
    ant = np.asarray(site["antenna_pos_tx"], dtype=float)
    ant = ant - np.mean(ant, axis=0, keepdims=True)
    aperture_rms_m = float(np.sqrt(np.mean(np.sum(ant[:, :2] ** 2, axis=1))))
    wavelength_m = 299792458.0 / float(site["frequency_hz"])
    k = 2.0 * np.pi / wavelength_m
    if aperture_rms_m <= 0.0:
        return 0.0
    return float(np.deg2rad(phase_error_deg) / (k * aperture_rms_m))


def perturb_unit_enu(unit, sigma_rad, rng):
    unit = np.asarray(unit, dtype=float)
    unit /= np.linalg.norm(unit)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(unit, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(unit, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    delta = rng.normal(0.0, sigma_rad, size=2)
    out = unit + delta[0] * e1 + delta[1] * e2
    return out / np.linalg.norm(out)


def position_uncertainty_mc(
    tx_ecef,
    rx_ecef,
    rot,
    unit_enu,
    bistatic_range_m,
    range_gate_width_m,
    angle_sigma_rad,
    n_mc,
    rng,
):
    if n_mc <= 0:
        return None
    sigma_range_m = float(range_gate_width_m) / np.sqrt(12.0)
    positions = []
    for _ in range(n_mc):
        u = perturb_unit_enu(unit_enu, angle_sigma_rad, rng)
        br = float(bistatic_range_m) + rng.normal(0.0, sigma_range_m)
        solved = aod_delay_position(tx_ecef, rx_ecef, u, rot, br)
        if solved is None:
            continue
        pos_ecef, _rho, _resid = solved
        positions.append(rot.T @ (pos_ecef - tx_ecef))
    if len(positions) < max(10, n_mc // 10):
        return None
    positions = np.array(positions, dtype=float)
    cov = np.cov(positions.T)
    sigma_axis = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    center = np.mean(positions, axis=0)
    return {
        "n_mc": int(len(positions)),
        "range_gate_width_m": float(range_gate_width_m),
        "range_sigma_m": sigma_range_m,
        "aod_phase_error_deg": None,
        "aod_angle_sigma_deg": float(np.rad2deg(angle_sigma_rad)),
        "position_mean_enu_m": center.tolist(),
        "position_cov_enu_m2": cov.tolist(),
        "position_sigma_enu_m": np.sqrt(np.maximum(np.diag(cov), 0.0)).tolist(),
        "position_sigma_axes_m": sigma_axis.tolist(),
        "position_sigma_3d_m": float(np.sqrt(np.trace(cov))),
        "position_sigma_horizontal_m": float(np.sqrt(cov[0, 0] + cov[1, 1])),
    }


def fit_line_enu(rows, t_ref):
    t = np.array([r["unix_time"] - t_ref for r in rows], dtype=float)
    y = np.array([r["position_enu_m"] for r in rows], dtype=float)
    weights = np.array([max(r["aod_score"], 1e-6) * 10.0 ** (r["snr_db"] / 10.0) for r in rows], dtype=float)
    weights /= np.median(weights)
    a = np.column_stack((np.ones_like(t), t))
    aw = a * np.sqrt(weights)[:, None]
    params = []
    pred = np.zeros_like(y)
    for dim in range(3):
        yw = y[:, dim] * np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        params.append(coef)
        pred[:, dim] = a @ coef
    params = np.array(params)
    residual_m = np.linalg.norm(y - pred, axis=1)
    return {
        "position_ref_enu_m": params[:, 0],
        "velocity_enu_mps": params[:, 1],
        "predicted_enu_m": pred,
        "residual_m": residual_m,
        "rms_residual_m": float(np.sqrt(np.mean(residual_m**2))),
        "median_residual_m": float(np.median(residual_m)),
    }


def model_fit_enu(fit, t_unix):
    return trajectory_position_enu(fit, t_unix)


def trajectory_position_enu_from_params(params, fit, t_unix):
    params = np.asarray(params, dtype=float)
    t_unix = np.asarray(t_unix, dtype=float)
    return params[:3] + (t_unix - float(fit["t_ref_unix"]))[:, None] * params[3:6]


def trajectory_covariance_samples(fit, n_samples=200):
    covariance = fit.get("linearized_fit_error", {}).get("covariance")
    if covariance is None or n_samples <= 0:
        return np.empty((0, 6), dtype=float)
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    return sample_fit_parameters(params, np.asarray(covariance, dtype=float), n_samples=n_samples, seed=20260530)


def enu_series_to_geodetic(tx_ecef, rot, enu):
    ecef = tx_ecef[None, :] + np.asarray(enu, dtype=float) @ rot.T
    llh = np.array([ecef_to_geodetic_wgs84(x) for x in ecef])
    return ecef, llh


def plot_map(rows, fit, sites, tx_ecef, rot, line_fit, out_png, uncertainty_samples=None):
    try:
        if "readline" not in sys.modules:
            readline_stub = types.ModuleType("readline")
            readline_stub.get_completer = lambda: None
            readline_stub.set_completer = lambda func: None
            readline_stub.parse_and_bind = lambda spec: None
            sys.modules["readline"] = readline_stub
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        projection = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15.0, 7.2),
            subplot_kw={"projection": projection},
            constrained_layout=True,
        )
        transform = projection
    except Exception:
        cfeature = None
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2), constrained_layout=True)
        transform = None
    axes = np.atleast_1d(axes)

    stations = sorted(set(r["station"] for r in rows))
    colors = dict(zip(stations, plt.cm.tab10(np.linspace(0, 1, len(stations)))))
    t = np.linspace(min(r["unix_time"] for r in rows), max(r["unix_time"] for r in rows), 220)
    tri_enu = np.array([model_fit_enu(fit, ti) for ti in t])
    _, tri_llh = enu_series_to_geodetic(tx_ecef, rot, tri_enu)
    tx_lat, tx_lon, _ = sites[stations[0]]["gps_tx"]
    sample_llh = []
    if uncertainty_samples is not None and len(uncertainty_samples) > 0:
        for params in uncertainty_samples:
            sample_enu = trajectory_position_enu_from_params(params, fit, t)
            _, llh = enu_series_to_geodetic(tx_ecef, rot, sample_enu)
            sample_llh.append(llh)

    echo_lons = [r["lon_deg"] for r in rows] + tri_llh[:, 1].tolist()
    echo_lats = [r["lat_deg"] for r in rows] + tri_llh[:, 0].tolist()
    for llh in sample_llh:
        echo_lons.extend(llh[:, 1].tolist())
        echo_lats.extend(llh[:, 0].tolist())
    context_lons = list(echo_lons) + [tx_lon]
    context_lats = list(echo_lats) + [tx_lat]
    for station in stations:
        rx_lat, rx_lon, _ = sites[station]["gps_rx"]
        context_lons.append(rx_lon)
        context_lats.append(rx_lat)

    def setup_axis(ax, title, lons, lats, pad_frac, min_pad):
        if cfeature is not None and hasattr(ax, "add_feature"):
            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none")
            ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none")
            ax.coastlines(resolution="10m", linewidth=0.8, color="0.25")
            gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="0.65", alpha=0.8)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
        pad_lon = max(min_pad, pad_frac * (max(lons) - min(lons)))
        pad_lat = max(min_pad, pad_frac * (max(lats) - min(lats)))
        extent = [min(lons) - pad_lon, max(lons) + pad_lon, min(lats) - pad_lat, max(lats) + pad_lat]
        if hasattr(ax, "set_extent"):
            ax.set_extent(extent, crs=transform)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)

    def draw_echoes(ax, point_size=20, alpha=0.65):
        for si, llh in enumerate(sample_llh):
            ax.plot(
                llh[:, 1],
                llh[:, 0],
                color="0.55",
                lw=0.7,
                alpha=0.18,
                label="Tri-static uncertainty samples" if si == 0 else None,
                transform=transform,
                zorder=1,
            )
        for station in stations:
            rr = [r for r in rows if r["station"] == station]
            lon = np.array([r["lon_deg"] for r in rr])
            lat = np.array([r["lat_deg"] for r in rr])
            ax.scatter(
                lon,
                lat,
                s=point_size,
                alpha=alpha,
                color=colors[station],
                label=f"{station_label(station)} AoD + delay",
                transform=transform,
                zorder=3,
            )
        ax.plot(
            tri_llh[:, 1],
            tri_llh[:, 0],
            color="k",
            lw=2.4,
            label="Tri-static trajectory",
            transform=transform,
            zorder=4,
        )

    setup_axis(axes[0], "Station Geometry And Echo Context", context_lons, context_lats, pad_frac=0.18, min_pad=0.1)
    setup_axis(axes[1], "Zoom On Head-Echo Positions", echo_lons, echo_lats, pad_frac=0.35, min_pad=0.025)
    draw_echoes(axes[0], point_size=18, alpha=0.55)
    draw_echoes(axes[1], point_size=28, alpha=0.78)

    axes[0].scatter(tx_lon, tx_lat, marker="*", s=150, color="gold", edgecolor="0.2", label="Haystack TX", transform=transform, zorder=8)
    axes[1].scatter(tx_lon, tx_lat, marker="*", s=100, color="gold", edgecolor="0.2", transform=transform, zorder=8)
    for station in stations:
        rx_lat, rx_lon, _ = sites[station]["gps_rx"]
        axes[0].scatter(rx_lon, rx_lat, marker="^", s=70, color=colors[station], edgecolor="0.2", transform=transform, zorder=8)
        axes[0].text(rx_lon + 0.025, rx_lat + 0.015, station_label(station), fontsize=9, transform=transform)

    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Head-Echo AoD + Bistatic-Delay Positions")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_comparison(rows, fit, tx_ecef, rot, line_fit, summary, out_png):
    stations = sorted(set(r["station"] for r in rows))
    colors = dict(zip(stations, plt.cm.tab10(np.linspace(0, 1, len(stations)))))
    times = np.array([r["unix_time"] for r in rows], dtype=float)
    t0 = float(fit["t_ref_unix"])
    t_rel = times - t0
    pos = np.array([r["position_enu_m"] for r in rows], dtype=float)
    tri = np.array([model_fit_enu(fit, t) for t in times])
    err_km = np.linalg.norm(pos - tri, axis=1) / 1000.0

    t_grid = np.linspace(times.min(), times.max(), 250)
    tri_grid = np.array([model_fit_enu(fit, t) for t in t_grid])
    aod_fit_grid = line_fit["position_ref_enu_m"] + (t_grid - t0)[:, None] * line_fit["velocity_enu_mps"]
    _, tri_llh = enu_series_to_geodetic(tx_ecef, rot, tri_grid)
    _, aod_llh = enu_series_to_geodetic(tx_ecef, rot, aod_fit_grid)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0, 0.9]},
    )

    ax = axes[0]
    for station in stations:
        rr = [r for r in rows if r["station"] == station]
        ax.scatter(
            [r["lon_deg"] for r in rr],
            [r["lat_deg"] for r in rr],
            s=24,
            alpha=0.72,
            color=colors[station],
            label=f"{station_label(station)} AoD + delay",
        )
    ax.plot(tri_llh[:, 1], tri_llh[:, 0], color="k", lw=2.4, label="Tri-static fit")
    ax.plot(aod_llh[:, 1], aod_llh[:, 0], color="C1", lw=2.0, ls="--", label="AoD + delay fit")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Head-echo position solutions")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    for station in stations:
        mask = np.array([r["station"] == station for r in rows])
        ax.scatter(t_rel[mask], err_km[mask], s=24, alpha=0.75, color=colors[station], label=station_label(station))
    ax.axhline(summary["all"]["median_position_error_vs_tristatic_m"] / 1000.0, color="0.25", lw=1.6, ls=":", label="Median point error")
    ax.axhline(summary["all"]["aod_fit_rms_error_vs_tristatic_m"] / 1000.0, color="C1", lw=1.8, ls="--", label="Fit RMS separation")
    ax.set_xlabel("Time from trajectory reference (s)")
    ax.set_ylabel("AoD+delay point separation (km)")
    ax.set_title("Pointwise separation")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    bins = np.linspace(0.0, max(1.0, np.nanpercentile(err_km, 97.0)), 18)
    ax.hist(err_km, bins=bins, color="0.45", alpha=0.82, edgecolor="white")
    ax.axvline(summary["all"]["median_position_error_vs_tristatic_m"] / 1000.0, color="k", lw=2.0, label="Median")
    ax.axvline(summary["all"]["rms_position_error_vs_tristatic_m"] / 1000.0, color="C3", lw=2.0, ls="--", label="RMS")
    ax.text(
        0.98,
        0.96,
        (
            f"n = {summary['all']['n']}\n"
            f"median = {summary['all']['median_position_error_vs_tristatic_m']/1000.0:.1f} km\n"
            f"point RMS = {summary['all']['rms_position_error_vs_tristatic_m']/1000.0:.1f} km\n"
            f"fit RMS = {summary['all']['aod_fit_rms_error_vs_tristatic_m']/1000.0:.1f} km"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.92},
    )
    ax.set_xlabel("Separation (km)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Complementary Tri-static And Transmit-Interferometric Head-Echo Positions")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report_summary(summary, out_tex):
    all_stats = summary["all"]
    by_station = summary["by_station"]
    lines = [
        "% Auto-generated by aod_position_from_merged.py",
        f"\\newcommand{{\\AodTriN}}{{{all_stats['n']}}}",
        f"\\newcommand{{\\AodTriMedianPointKm}}{{{all_stats['median_position_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriRmsPointKm}}{{{all_stats['rms_position_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriFitRmsKm}}{{{all_stats['aod_fit_rms_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriFitMedianKm}}{{{all_stats['aod_fit_median_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriMedianUncertaintyKm}}{{{all_stats['median_position_sigma_3d_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriBeverlyMedianKm}}{{{by_station['beverly']['median_position_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriClintonMedianKm}}{{{by_station['clinton']['median_position_error_vs_tristatic_m']/1000.0:.1f}}}",
        f"\\newcommand{{\\AodTriPomfretMedianKm}}{{{by_station['pomfret']['median_position_error_vs_tristatic_m']/1000.0:.1f}}}",
    ]
    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows, line_fit, fit):
    out = {}
    t_ref = float(fit["t_ref_unix"])
    for station in sorted(set(r["station"] for r in rows)):
        rr = [r for r in rows if r["station"] == station]
        pos = np.array([r["position_enu_m"] for r in rr])
        tri = np.array([model_fit_enu(fit, r["unix_time"]) for r in rr])
        err = np.linalg.norm(pos - tri, axis=1)
        out[station] = {
            "n": len(rr),
            "median_snr_db": float(np.median([r["snr_db"] for r in rr])),
            "median_aod_prior_offset_deg": float(np.median([r["aod_prior_offset_deg"] for r in rr])),
            "median_position_error_vs_tristatic_m": float(np.median(err)),
            "rms_position_error_vs_tristatic_m": float(np.sqrt(np.mean(err**2))),
            "median_position_sigma_3d_m": float(
                np.median([r["uncertainty"]["position_sigma_3d_m"] for r in rr if r.get("uncertainty")])
            ),
            "median_position_sigma_horizontal_m": float(
                np.median([r["uncertainty"]["position_sigma_horizontal_m"] for r in rr if r.get("uncertainty")])
            ),
        }
    pos = np.array([r["position_enu_m"] for r in rows])
    tri = np.array([model_fit_enu(fit, r["unix_time"]) for r in rows])
    err = np.linalg.norm(pos - tri, axis=1)
    fit_pred = line_fit["predicted_enu_m"]
    tri_at_fit = np.array([model_fit_enu(fit, r["unix_time"]) for r in rows])
    fit_err = np.linalg.norm(fit_pred - tri_at_fit, axis=1)
    return {
        "by_station": out,
        "all": {
            "n": len(rows),
            "median_position_error_vs_tristatic_m": float(np.median(err)),
            "rms_position_error_vs_tristatic_m": float(np.sqrt(np.mean(err**2))),
            "median_position_sigma_3d_m": float(
                np.median([r["uncertainty"]["position_sigma_3d_m"] for r in rows if r.get("uncertainty")])
            ),
            "median_position_sigma_horizontal_m": float(
                np.median([r["uncertainty"]["position_sigma_horizontal_m"] for r in rows if r.get("uncertainty")])
            ),
            "aod_fit_median_error_vs_tristatic_m": float(np.median(fit_err)),
            "aod_fit_rms_error_vs_tristatic_m": float(np.sqrt(np.mean(fit_err**2))),
            "aod_fit_rms_residual_m": line_fit["rms_residual_m"],
            "aod_fit_median_residual_m": line_fit["median_residual_m"],
            "aod_fit_speed_mps": float(np.linalg.norm(line_fit["velocity_enu_mps"])),
            "tristatic_speed_mps": float(np.linalg.norm(fit["fit_parameters_enu_at_tx"]["velocity_mps"])),
        },
    }


def main():
    p = argparse.ArgumentParser(description="Estimate head-echo positions from merged RDMF AoD voltages and bistatic delay.")
    p.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY)
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--stations", nargs="+", default=["beverly", "clinton", "pomfret"])
    p.add_argument("--min-snr-db", type=float, default=8.0)
    p.add_argument("--cone-deg", type=float, default=20.0)
    p.add_argument("--grid-size", type=int, default=361)
    p.add_argument("--max-zenith-angle", type=float, default=90.0)
    p.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=-1.0)
    p.add_argument("--max-time-error-s", type=float, default=0.01)
    p.add_argument("--max-range-error-km", type=float, default=1.0)
    p.add_argument("--aod-phase-error-deg", type=float, default=15.0)
    p.add_argument("--position-uncertainty-mc", type=int, default=300)
    p.add_argument("--uncertainty-samples", type=int, default=200)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--report-fig", default=DEFAULT_REPORT_FIG)
    p.add_argument("--report-tex", default=DEFAULT_REPORT_TEX)
    args = p.parse_args()

    fit = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    wanted = set(args.stations)
    observations = [o for o in fit["observations"] if o["station"] in wanted and o["snr_db"] > args.min_snr_db]
    if not observations:
        raise SystemExit("No head-echo observations passed the SNR/station filters")

    sites = {station: load_station_yaml(args.data_dir, station) for station in wanted}
    first_site = sites[args.stations[0]]
    tx_ecef = geodetic_to_ecef(*first_site["gps_tx"])
    rot = enu_to_ecef_matrix(first_site["gps_tx"][0], first_site["gps_tx"][1])
    dir_unit, dir_az, dir_el = regular_direction_grid(args.grid_size, args.max_zenith_angle)

    rdmf = {}
    rows = []
    skipped = 0
    rng = np.random.default_rng(20260530)
    try:
        for station in wanted:
            path = Path(args.merged_dir) / f"rdmf_peak_{station}_merged.h5"
            if path.exists():
                rdmf[station] = load_rdmf_station(path)
        for ob in observations:
            station = ob["station"]
            if station not in rdmf:
                skipped += 1
                continue
            picked = pick_voltage(rdmf[station], ob, args.max_time_error_s, args.max_range_error_km)
            if picked is None:
                skipped += 1
                continue
            ti, ri, dt, dr, voltage, snr_db = picked
            prior_vec = trajectory_position_enu(fit, ob["unix_time"])
            prior_unit = prior_vec / np.linalg.norm(prior_vec)
            aod = estimate_one(voltage, sites[station], dir_unit, dir_az, dir_el, prior_unit, args.cone_deg, args.phase_sign)
            rx_ecef = geodetic_to_ecef(*sites[station]["gps_rx"])
            solved = aod_delay_position(tx_ecef, rx_ecef, aod["best_unit_enu"], rot, ob["range_m"])
            if solved is None:
                skipped += 1
                continue
            pos_ecef, tx_range_m, delay_residual_m = solved
            pos_enu = rot.T @ (pos_ecef - tx_ecef)
            lat, lon, alt_m = ecef_to_geodetic_wgs84(pos_ecef)
            tri_pos = trajectory_position_enu(fit, ob["unix_time"])
            angle_sigma_rad = tx_angle_sigma_rad(sites[station], args.aod_phase_error_deg)
            uncertainty = position_uncertainty_mc(
                tx_ecef,
                rx_ecef,
                rot,
                aod["best_unit_enu"],
                ob["range_m"],
                rdmf[station]["range_gate_width_m"],
                angle_sigma_rad,
                args.position_uncertainty_mc,
                rng,
            )
            if uncertainty is not None:
                uncertainty["aod_phase_error_deg"] = float(args.aod_phase_error_deg)
            rows.append(
                {
                    "station": station,
                    "unix_time": float(ob["unix_time"]),
                    "snr_db": float(snr_db),
                    "bistatic_range_m": float(ob["range_m"]),
                    "time_index": ti,
                    "range_index": ri,
                    "time_error_s": dt,
                    "range_error_km": dr,
                    "aod_az_deg": aod["best_az_deg"],
                    "aod_el_deg": aod["best_el_deg"],
                    "aod_unit_enu": aod["best_unit_enu"],
                    "aod_prior_az_deg": aod["prior_az_deg"],
                    "aod_prior_el_deg": aod["prior_el_deg"],
                    "aod_prior_offset_deg": aod["prior_offset_deg"],
                    "aod_score": aod["score"],
                    "tx_range_m": float(tx_range_m),
                    "delay_residual_m": delay_residual_m,
                    "position_ecef_m": pos_ecef.tolist(),
                    "position_enu_m": pos_enu.tolist(),
                    "lat_deg": float(lat),
                    "lon_deg": float(lon),
                    "alt_m": float(alt_m),
                    "position_error_vs_tristatic_m": float(np.linalg.norm(pos_enu - tri_pos)),
                    "uncertainty": uncertainty,
                }
            )
    finally:
        for val in rdmf.values():
            val["handle"].close()

    if not rows:
        raise SystemExit("No AoD+delay positions could be solved")

    line_fit = fit_line_enu(rows, float(fit["t_ref_unix"]))
    summary = summarize(rows, line_fit, fit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "aod_position_summary.json"
    out_png = out_dir / "aod_positions_map.png"
    payload = {
        "schema": "new_england_bolide.aod_position_from_merged.v1",
        "merged_dir": args.merged_dir,
        "trajectory_json": args.trajectory_json,
        "stations": args.stations,
        "min_snr_db": args.min_snr_db,
        "cone_deg": args.cone_deg,
        "phase_sign": args.phase_sign,
        "grid_size": args.grid_size,
        "aod_phase_error_deg": args.aod_phase_error_deg,
        "position_uncertainty_mc": args.position_uncertainty_mc,
        "n_observations": len(observations),
        "n_used": len(rows),
        "n_skipped": skipped,
        "summary": summary,
        "aod_position_fit": {
            "position_ref_enu_m": line_fit["position_ref_enu_m"].tolist(),
            "velocity_enu_mps": line_fit["velocity_enu_mps"].tolist(),
            "rms_residual_m": line_fit["rms_residual_m"],
            "median_residual_m": line_fit["median_residual_m"],
        },
        "positions": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    uncertainty_samples = trajectory_covariance_samples(fit, n_samples=args.uncertainty_samples)
    plot_map(rows, fit, sites, tx_ecef, rot, line_fit, out_png, uncertainty_samples=uncertainty_samples)
    plot_appendix_comparison(rows, fit, tx_ecef, rot, line_fit, summary, args.report_fig)
    write_report_summary(summary, args.report_tex)

    print(f"used={len(rows)} skipped={skipped} snr>{args.min_snr_db:g}dB cone={args.cone_deg:g}deg phase_sign={args.phase_sign:g}")
    print(
        "AoD+delay fit: "
        f"speed={summary['all']['aod_fit_speed_mps']/1000.0:.2f} km/s "
        f"residual_rms={summary['all']['aod_fit_rms_residual_m']/1000.0:.2f} km "
        f"vs_tristatic_rms={summary['all']['aod_fit_rms_error_vs_tristatic_m']/1000.0:.2f} km"
    )
    for station, stats in summary["by_station"].items():
        print(
            f"{station}: n={stats['n']} "
            f"median_aod_offset={stats['median_aod_prior_offset_deg']:.2f}deg "
            f"median_pos_err={stats['median_position_error_vs_tristatic_m']/1000.0:.2f}km "
            f"rms_pos_err={stats['rms_position_error_vs_tristatic_m']/1000.0:.2f}km"
        )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print(f"Wrote {args.report_fig}")
    print(f"Wrote {args.report_tex}")


if __name__ == "__main__":
    main()
