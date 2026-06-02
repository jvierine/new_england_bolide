#!/usr/bin/env python3
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml
from estimate_head_echo_trajectory import (
    C,
    ecef_to_geodetic_wgs84,
    model_state_series,
    predict_dopplers,
    predict_ranges,
    sample_fit_parameters,
)


DATA_DIR = ROOT / "2026_05_30_bolide"
PRODUCT_DIR = ROOT / "products" / "head_echo_trajectory_roi_fit"
FIG_DIR = ROOT / "reports" / "figures"
FIT_JSON = PRODUCT_DIR / "head_echo_trajectory.json"


def set_publication_plot_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )


def load_fit():
    fit = json.loads(FIT_JSON.read_text(encoding="utf-8"))
    stations = fit["used_stations"]
    geom = {}
    for station in stations:
        site = load_station_yaml(DATA_DIR, station)
        geom[station] = {
            "tx": geodetic_to_ecef(*site["gps_tx"]),
            "rx": geodetic_to_ecef(*site["gps_rx"]),
            "frequency_hz": site["frequency_hz"],
            "gps_tx": site["gps_tx"].tolist(),
            "gps_rx": site["gps_rx"].tolist(),
        }
    tx_site = load_station_yaml(DATA_DIR, stations[0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    return fit, geom, tx0, rot, params


def covariance_samples(fit, n_samples=800):
    cov = np.array(fit["linearized_fit_error"]["covariance"], dtype=float)
    return sample_fit_parameters(
        np.r_[
            fit["fit_parameters_enu_at_tx"]["position_ref_m"],
            fit["fit_parameters_enu_at_tx"]["velocity_mps"],
        ],
        cov,
        n_samples=n_samples,
        seed=20260530,
    )


def plot_range_doppler_measurements(fit, geom, tx0, rot, params, samples):
    obs = fit["observations"]
    t_ref = fit["t_ref_unix"]
    stations = fit["used_stations"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)

    for si, station in enumerate(stations):
        color = f"C{si}"
        station_obs = [o for o in obs if o["station"] == station]
        station_obs.sort(key=lambda o: o["unix_time"])
        t_obs = np.array([o["unix_time"] for o in station_obs])
        tr = t_obs - t_ref
        range_km = np.array([o["range_m"] for o in station_obs]) / 1000.0
        dop_khz = np.array([o["doppler_hz"] for o in station_obs]) / 1000.0
        model_obs = [{"station": station, "unix_time": float(t)} for t in t_obs]
        pred_range_km = predict_ranges(params, model_obs, geom, t_ref, tx0, rot) / 1000.0
        pred_dop_khz = predict_dopplers(params, model_obs, geom, t_ref, tx0, rot) / 1000.0
        sample_range = np.array([predict_ranges(p, model_obs, geom, t_ref, tx0, rot) / 1000.0 for p in samples])
        sample_dop = np.array([predict_dopplers(p, model_obs, geom, t_ref, tx0, rot) / 1000.0 for p in samples])
        range_lo, range_hi = np.percentile(sample_range, [2.5, 97.5], axis=0)
        dop_lo, dop_hi = np.percentile(sample_dop, [2.5, 97.5], axis=0)

        uncertainty_label = "95% fit uncertainty" if si == 0 else None
        axes[0].scatter(tr, range_km, s=20, alpha=0.75, color=color, label=f"{station} measured")
        axes[0].fill_between(tr, range_lo, range_hi, color=color, alpha=0.18, linewidth=0, label=uncertainty_label)
        axes[0].plot(tr, pred_range_km, color=color, lw=2.0, label=f"{station} fit")
        axes[1].scatter(tr, dop_khz, s=20, alpha=0.75, color=color, label=f"{station} measured")
        axes[1].fill_between(tr, dop_lo, dop_hi, color=color, alpha=0.18, linewidth=0, label=uncertainty_label)
        axes[1].plot(tr, pred_dop_khz, color=color, lw=2.0, label=f"{station} fit")

    axes[0].set_ylabel("Bistatic range (km)")
    axes[0].set_title("Head-echo trajectory fit observables")
    axes[0].grid(True, color="0.9", linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=10)
    axes[1].set_xlabel("Seconds relative to fit reference time")
    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].grid(True, color="0.9", linewidth=0.6)
    axes[1].legend(ncol=2, fontsize=10)

    out = FIG_DIR / "head_echo_range_doppler_fit.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_altitude_velocity(fit, tx0, rot, params, samples):
    obs = fit["observations"]
    t_ref = fit["t_ref_unix"]
    t_obs = np.array([o["unix_time"] for o in obs], dtype=float)
    t_grid = np.linspace(float(np.min(t_obs)), float(np.max(t_obs)), 220)
    model = model_state_series(params, t_grid, t_ref, tx0, rot)
    sample_states = [model_state_series(p, t_grid, t_ref, tx0, rot) for p in samples]
    sample_alt_km = np.array([s["lat_lon_alt"][:, 2] / 1000.0 for s in sample_states])
    sample_vel_enu = np.array([s["velocity_enu_mps"] / 1000.0 for s in sample_states])
    sample_speed = np.array([s["speed_mps"] / 1000.0 for s in sample_states])
    alt_lo, alt_hi = np.percentile(sample_alt_km, [2.5, 97.5], axis=0)
    speed_lo, speed_hi = np.percentile(sample_speed, [2.5, 97.5], axis=0)
    vel_lo, vel_hi = np.percentile(sample_vel_enu, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), sharex=True, constrained_layout=True)
    tr = model["time_relative_s"]
    axes[0].fill_between(tr, alt_lo, alt_hi, color="C0", alpha=0.2, linewidth=0, label="95% fit uncertainty")
    axes[0].plot(tr, model["lat_lon_alt"][:, 2] / 1000.0, color="C0", lw=2.2, label="fit")
    axes[0].scatter([0.0], [fit["geodetic_ref_wgs84"]["alt_m"] / 1000.0], color="k", s=32, zorder=4, label="t_ref")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Best-fit head echo altitude and velocity")
    axes[0].grid(True, color="0.9", linewidth=0.6)
    axes[0].legend(loc="upper right")

    vel = model["velocity_enu_mps"] / 1000.0
    for i, (label, color) in enumerate((("east", "C0"), ("north", "C1"), ("up", "C2"))):
        axes[1].fill_between(tr, vel_lo[:, i], vel_hi[:, i], color=color, alpha=0.12, linewidth=0)
        axes[1].plot(tr, vel[:, i], color=color, lw=1.8, label=label)
    axes[1].fill_between(tr, speed_lo, speed_hi, color="k", alpha=0.12, linewidth=0)
    axes[1].plot(tr, model["speed_mps"] / 1000.0, color="k", lw=2.5, label="speed")
    axes[1].set_xlabel(f"Seconds relative to {fit['t_ref_utc']}")
    axes[1].set_ylabel("Velocity (km/s)")
    axes[1].grid(True, color="0.9", linewidth=0.6)
    axes[1].legend(ncol=4)

    out = FIG_DIR / "head_echo_altitude_velocity.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_cartopy_figure():
    src = PRODUCT_DIR / "head_echo_cartopy_trajectory_uncertainty.png"
    dst = FIG_DIR / "head_echo_cartopy_trajectory_uncertainty.png"
    dst.write_bytes(src.read_bytes())
    return dst


def write_trajectory_stats(fit, tx0, rot, params):
    obs = fit["observations"]
    t_first = min(o["unix_time"] for o in obs)
    first_state = model_state_series(params, [t_first], fit["t_ref_unix"], tx0, rot)
    first_lat, first_lon, first_alt_m = first_state["lat_lon_alt"][0]
    ref_lat, ref_lon, ref_alt_m = ecef_to_geodetic_wgs84(np.array(fit["position_ref_ecef_m"]))
    v_enu = np.array(fit["fit_parameters_enu_at_tx"]["velocity_mps"])
    speed = np.linalg.norm(v_enu)
    first_utc = datetime.fromtimestamp(t_first, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    lines = [
        f"\\newcommand{{\\TrajectoryNObservations}}{{{fit['n_observations']}}}\n",
        f"\\newcommand{{\\TrajectoryNRejected}}{{{fit['n_rejected_roi_points']}}}\n",
        f"\\newcommand{{\\TrajectoryFirstDetectionUTC}}{{{first_utc}}}\n",
        f"\\newcommand{{\\TrajectoryInitialHeightKm}}{{{first_alt_m / 1000.0:.1f}}}\n",
        f"\\newcommand{{\\TrajectoryInitialLat}}{{{first_lat:.3f}}}\n",
        f"\\newcommand{{\\TrajectoryInitialLon}}{{{first_lon:.3f}}}\n",
        f"\\newcommand{{\\TrajectoryRefHeightKm}}{{{ref_alt_m / 1000.0:.1f}}}\n",
        f"\\newcommand{{\\TrajectoryRefLat}}{{{ref_lat:.3f}}}\n",
        f"\\newcommand{{\\TrajectoryRefLon}}{{{ref_lon:.3f}}}\n",
        f"\\newcommand{{\\TrajectorySpeedKmS}}{{{speed / 1000.0:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryVEastKmS}}{{{v_enu[0] / 1000.0:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryVNorthKmS}}{{{v_enu[1] / 1000.0:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryVUpKmS}}{{{v_enu[2] / 1000.0:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryRangeRmsM}}{{{fit['range_residual_m']['rms']:.0f}}}\n",
        f"\\newcommand{{\\TrajectoryDopplerRmsHz}}{{{fit['doppler_residual_hz']['rms']:.0f}}}\n",
    ]
    out = ROOT / "reports" / "trajectory_stats.tex"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def main():
    set_publication_plot_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fit, geom, tx0, rot, params = load_fit()
    samples = covariance_samples(fit)
    measurement_fig = plot_range_doppler_measurements(fit, geom, tx0, rot, params, samples)
    altitude_fig = plot_altitude_velocity(fit, tx0, rot, params, samples)
    cartopy_fig = copy_cartopy_figure()
    stats = write_trajectory_stats(fit, tx0, rot, params)
    print(measurement_fig)
    print(altitude_fig)
    print(cartopy_fig)
    print(stats)


if __name__ == "__main__":
    main()
