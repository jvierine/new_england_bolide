#!/usr/bin/env python3
"""Monte Carlo orbit determination for the 2026-05-30 Cape Cod Bay bolide.

This adapts the workflow in ../meteor/orbit_det_sample.py to local trajectory
products. It supports both the original head-echo ENU trajectory product and
the ballistic GCRS initial-state JSON products.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time

import dasst
from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml


DEFAULT_TRAJECTORY = Path("products/head_echo_trajectory_roi_fit/head_echo_trajectory.json")
DEFAULT_DATA_DIR = Path("2026_05_30_bolide")
DEFAULT_KERNEL = Path("data/de440s.bsp")
DEFAULT_OUT = Path("products/orbit_determination/head_echo_orbit_samples.h5")
OMEGA_EARTH = 7.2921150e-5

KEPLER_OUT_FRAMES = ["ICRS", "HeliocentricMeanEcliptic"]
RADIANT_OUT_FRAMES = ["GCRS", "GeocentricMeanEcliptic"]
KEPLER_NAMES = ["a_m", "e", "inclination_deg", "arg_periapsis_deg", "raan_deg", "true_anomaly_deg"]


def iso_utc(unix_time: float) -> str:
    return datetime.fromtimestamp(float(unix_time), UTC).isoformat().replace("+00:00", "Z")


def clean_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    vals, vecs = np.linalg.eigh(covariance)
    vals = np.maximum(vals, 0.0)
    return (vecs * vals) @ vecs.T


def load_trajectory_state(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == "new_england_bolide.ballistic_initial_state.v1":
        if payload.get("trajectory_fit_version", "unspecified") == "unspecified":
            raise ValueError(f"Ballistic initial-state JSON has unspecified trajectory_fit_version: {path}")
        return {
            "payload": payload,
            "source_kind": "ballistic_initial_state",
            "t_ref_unix": float(payload["time_unix"]),
            "mean_state": np.asarray(payload["state"], dtype=float),
            "cov_state": clean_covariance(np.asarray(payload["covariance"], dtype=float)),
            "fit_state_frame": payload.get("frame", "GCRS"),
            "trajectory_fit_version": payload.get("trajectory_fit_version", "unspecified"),
            "trajectory_fit_version_description": payload.get("trajectory_fit_version_description", "unspecified"),
            "used_stations": ["beverly"],
        }
    params = payload["fit_parameters_enu_at_tx"]
    errors = payload["linearized_fit_error"]
    return {
        "payload": payload,
        "source_kind": "head_echo_enu",
        "t_ref_unix": float(payload["t_ref_unix"]),
        "mean_enu": np.r_[params["position_ref_m"], params["velocity_mps"]].astype(float),
        "cov_enu": clean_covariance(np.array(errors["covariance"], dtype=float)),
        "fit_state_frame": "ENU_at_tx",
        "used_stations": list(payload["used_stations"]),
    }


def haystack_frame(data_dir: Path, station: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site = load_station_yaml(data_dir, station)
    gps_tx = np.asarray(site["gps_tx"], dtype=float)
    tx_ecef = geodetic_to_ecef(*gps_tx)
    rot_enu_to_ecef = enu_to_ecef_matrix(gps_tx[0], gps_tx[1])
    return gps_tx, tx_ecef, rot_enu_to_ecef


def sample_enu_states(mean_enu: np.ndarray, cov_enu: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_samples == 1:
        return mean_enu[None, :]
    return rng.multivariate_normal(mean_enu, cov_enu, size=n_samples)


def enu_samples_to_ecef_states(samples_enu: np.ndarray, tx_ecef: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pos_enu = samples_enu[:, :3]
    vel_enu = samples_enu[:, 3:6]
    states = np.empty((6, len(samples_enu)), dtype=float)
    states[:3, :] = (tx_ecef[None, :] + pos_enu @ rot.T).T
    states[3:, :] = (vel_enu @ rot.T).T
    return states


def gmst_angle(unix_time: float | np.ndarray) -> np.ndarray:
    unix_time = np.asarray(unix_time, dtype=float)
    jd = unix_time / 86400.0 + 2440587.5
    t_ut1 = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t_ut1**2
        - t_ut1**3 / 38710000.0
    )
    return np.deg2rad(np.mod(gmst_deg, 360.0))


def eci_to_ecef_position(pos_eci: np.ndarray, unix_time: float | np.ndarray) -> np.ndarray:
    pos_eci = np.asarray(pos_eci, dtype=float)
    theta = gmst_angle(unix_time)
    c = np.cos(theta)
    s = np.sin(theta)
    x = pos_eci[..., 0]
    y = pos_eci[..., 1]
    z = pos_eci[..., 2]
    return np.stack((c * x + s * y, -s * x + c * y, z), axis=-1)


def gcrs_samples_to_itrf_states(samples_state: np.ndarray, t_ref_unix: float) -> np.ndarray:
    """Convert GCRS inertial samples to ITRS position/velocity.

    The ballistic products use the repository's GCRS state convention.  DASST
    expects ITRS states, including velocity in the rotating terrestrial frame,
    so the Earth-rotation term is removed from the rotated inertial velocity.
    """
    pos_eci = samples_state[:, :3]
    vel_eci = samples_state[:, 3:6]
    pos_ecef = eci_to_ecef_position(pos_eci, t_ref_unix)
    vel_rotated = eci_to_ecef_position(vel_eci, t_ref_unix)
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    vel_ecef = vel_rotated - np.cross(omega, pos_ecef)
    states = np.empty((6, len(samples_state)), dtype=float)
    states[:3, :] = pos_ecef.T
    states[3:, :] = vel_ecef.T
    return states


def sample_state(mean_state: np.ndarray, cov_state: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_samples == 1:
        return mean_state[None, :]
    return rng.multivariate_normal(mean_state, cov_state, size=n_samples)


def summarize_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(samples, axis=1), np.nanstd(samples, axis=1)


def write_results(
    out: Path,
    args: argparse.Namespace,
    trajectory: dict,
    gps_tx: np.ndarray,
    states_ecef: np.ndarray,
    samples_enu: np.ndarray,
    results: dict,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        h.attrs["schema"] = "new_england_bolide.orbit_determination_sample.v1"
        h.attrs["trajectory_json"] = str(args.trajectory_json)
        h.attrs["kernel"] = str(args.kernel)
        h.attrs["t_ref_unix"] = trajectory["t_ref_unix"]
        h.attrs["t_ref_utc"] = iso_utc(trajectory["t_ref_unix"])
        h.attrs["n_samples"] = args.samples
        h.attrs["seed"] = args.seed
        h.attrs["dt_s"] = args.dt
        h.attrs["max_t_s"] = args.max_t
        h.attrs["termination_check"] = not args.no_termination
        h.attrs["save_propagation"] = args.save_propagation
        h.attrs["kepler_names"] = json.dumps(KEPLER_NAMES)
        h.create_dataset("haystack_gps_tx_lat_lon_alt", data=gps_tx)
        h.attrs["source_kind"] = trajectory["source_kind"]
        h.attrs["fit_state_frame"] = trajectory["fit_state_frame"]
        h.attrs["trajectory_fit_version"] = trajectory.get("trajectory_fit_version", "unspecified")
        h.attrs["trajectory_fit_version_description"] = trajectory.get("trajectory_fit_version_description", "unspecified")
        if "mean_enu" in trajectory:
            h.create_dataset("fit_mean_enu", data=trajectory["mean_enu"])
            h.create_dataset("fit_covariance_enu", data=trajectory["cov_enu"])
            h.create_dataset("sample_enu", data=samples_enu)
        else:
            h.attrs["fit_state_names"] = json.dumps(["x_gcrs_m", "y_gcrs_m", "z_gcrs_m", "vx_gcrs_mps", "vy_gcrs_mps", "vz_gcrs_mps"])
            h.create_dataset("fit_mean_state", data=trajectory["mean_state"])
            h.create_dataset("fit_covariance_state", data=trajectory["cov_state"])
            h.create_dataset("sample_state_input", data=samples_enu)
        h.create_dataset("sample_state_ecef_m_mps", data=states_ecef)

        t_sec = results["t"].sec
        h.create_dataset("propagation_time_s", data=t_sec)
        h.create_dataset("hcrs_state_at_exit_m_mps", data=results["hcrs_states"])
        if args.save_propagation:
            h.create_dataset("hcrs_states_m_mps", data=results["states"])

        for frame in KEPLER_OUT_FRAMES:
            kepler = results[f"kepler_{frame}"]
            final = kepler[:, -1, :]
            if args.save_propagation:
                h.create_dataset(f"kepler_{frame}", data=kepler)
            h.create_dataset(f"kepler_{frame}_final", data=final)
            mean, std = summarize_samples(final)
            h.create_dataset(f"kepler_{frame}_final_mean", data=mean)
            h.create_dataset(f"kepler_{frame}_final_std", data=std)

        for frame in RADIANT_OUT_FRAMES:
            for prefix in ("radiant_obs", "radiant_orbit", "radiant_sun"):
                key = f"{prefix}_{frame}"
                if key in results:
                    h.create_dataset(key, data=results[key])


def plot_eccentricity(out_h5: Path, frame: str = "HeliocentricMeanEcliptic") -> Path:
    import matplotlib.pyplot as plt

    with h5py.File(out_h5, "r") as h:
        e = h[f"kepler_{frame}_final"][1, :]
    out_png = out_h5.with_name(out_h5.stem + f"_eccentricity_{frame}.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.hist(e[np.isfinite(e)], bins=min(30, max(8, int(np.sqrt(len(e))))), color="C0", alpha=0.85)
    ax.axvline(np.nanmean(e), color="k", lw=1.5, label=f"mean={np.nanmean(e):.4g}")
    ax.set_xlabel("Eccentricity")
    ax.set_ylabel("Monte Carlo count")
    ax.set_title("Cape Cod Bay bolide pre-encounter orbit sample")
    ax.grid(True, color="0.9")
    ax.legend()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run(args: argparse.Namespace) -> None:
    trajectory = load_trajectory_state(args.trajectory_json)
    station = args.station or trajectory["used_stations"][0]
    gps_tx, tx_ecef, rot = haystack_frame(args.data_dir, station)
    if trajectory["source_kind"] == "ballistic_initial_state":
        samples = sample_state(trajectory["mean_state"], trajectory["cov_state"], args.samples, args.seed)
        states_ecef = gcrs_samples_to_itrf_states(samples, trajectory["t_ref_unix"])
    else:
        samples = sample_enu_states(trajectory["mean_enu"], trajectory["cov_enu"], args.samples, args.seed)
        states_ecef = enu_samples_to_ecef_states(samples, tx_ecef, rot)

    epoch = Time(trajectory["t_ref_unix"], scale="utc", format="unix")
    results = dasst.orbit_determination.rebound_od(
        states_ecef,
        epoch,
        str(args.kernel),
        kepler_out_frame=KEPLER_OUT_FRAMES,
        radiant_out_frame=RADIANT_OUT_FRAMES,
        termination_check=not args.no_termination,
        dt=args.dt,
        max_t=args.max_t,
        settings=None,
        progress_bar=not args.no_progress,
    )

    write_results(args.out, args, trajectory, gps_tx, states_ecef, samples, results)
    if args.plot:
        png = plot_eccentricity(args.out)
        print(f"Wrote {png}")

    with h5py.File(args.out, "r") as h:
        mean = h["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        std = h["kepler_HeliocentricMeanEcliptic_final_std"][()]
    print(f"Wrote {args.out}")
    print(f"e = {mean[1]:.6g} +/- {std[1]:.3g}")
    print(f"a = {mean[0] / 1.495978707e11:.6g} +/- {std[0] / 1.495978707e11:.3g} AU")
    print(f"i = {mean[2]:.6g} +/- {std[2]:.3g} deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo Keplerian orbit determination for the Cape Cod Bay bolide.")
    parser.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--station", default=None, help="Station YAML used to recover the common Haystack TX ENU frame.")
    parser.add_argument("--dt", type=float, default=10.0, help="Backward propagation timestep in seconds.")
    parser.add_argument("--max-t", type=float, default=7 * 24 * 3600.0, help="Maximum backward propagation time in seconds.")
    parser.add_argument("--no-termination", action="store_true", help="Propagate for max-t instead of stopping outside 0.01 AU from Earth.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-propagation", action="store_true", help="Also save full HCRS and Kepler propagation histories. This can be large.")
    parser.add_argument("--plot", action="store_true", help="Write an eccentricity histogram next to the HDF5 output.")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if not args.trajectory_json.exists():
        parser.error(f"--trajectory-json does not exist: {args.trajectory_json}")
    if not args.kernel.exists():
        parser.error(f"--kernel does not exist: {args.kernel}")
    return args


if __name__ == "__main__":
    run(parse_args())
