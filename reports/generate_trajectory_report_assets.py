#!/usr/bin/env python3
import json
import shutil
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


def plot_range_doppler_measurements(fit, geom, tx0, rot, params):
    obs = fit["observations"]
    t_ref = fit["t_ref_unix"]
    stations = fit["used_stations"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)

    for station in stations:
        station_obs = [o for o in obs if o["station"] == station]
        station_obs.sort(key=lambda o: o["unix_time"])
        t_obs = np.array([o["unix_time"] for o in station_obs])
        tr = t_obs - t_ref
        range_km = np.array([o["range_m"] for o in station_obs]) / 1000.0
        dop_khz = np.array([o["doppler_hz"] for o in station_obs]) / 1000.0
        model_obs = [{"station": station, "unix_time": float(t)} for t in t_obs]
        pred_range_km = predict_ranges(params, model_obs, geom, t_ref, tx0, rot) / 1000.0
        pred_dop_khz = predict_dopplers(params, model_obs, geom, t_ref, tx0, rot) / 1000.0

        axes[0].scatter(tr, range_km, s=20, alpha=0.75, label=f"{station} measured")
        axes[0].plot(tr, pred_range_km, lw=2.0, label=f"{station} fit")
        axes[1].scatter(tr, dop_khz, s=20, alpha=0.75, label=f"{station} measured")
        axes[1].plot(tr, pred_dop_khz, lw=2.0, label=f"{station} fit")

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


def copy_trajectory_figures():
    copied = []
    for name in (
        "head_echo_altitude_velocity.png",
        "head_echo_cartopy_trajectory_uncertainty.png",
    ):
        src = PRODUCT_DIR / name
        dst = FIG_DIR / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


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
    measurement_fig = plot_range_doppler_measurements(fit, geom, tx0, rot, params)
    copied = copy_trajectory_figures()
    stats = write_trajectory_stats(fit, tx0, rot, params)
    print(measurement_fig)
    for path in copied:
        print(path)
    print(stats)


if __name__ == "__main__":
    main()
