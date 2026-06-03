#!/usr/bin/env python3
"""Combined inertial trajectory fit using head echoes and trail positions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from estimate_head_echo_trajectory import (
    C,
    DEFAULT_ANNOTATIONS,
    ecef_to_geodetic_wgs84,
    enu_to_ecef_matrix,
    geodetic_to_ecef,
    load_observations,
    sample_fit_parameters,
    station_geometry,
)
from station_config import short_station_name


ROOT = Path(__file__).resolve().parent
DEFAULT_HEAD_FIT = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
DEFAULT_TRAIL = ROOT / "products" / "specular_trail_positions" / "specular_trail_position_summary.json"
DEFAULT_DATA_DIR = ROOT / "2026_05_30_bolide"
DEFAULT_OUT_DIR = ROOT / "products" / "combined_head_trail_eci_fit"
DEFAULT_REPORT_DIR = ROOT / "reports" / "figures"
OMEGA_EARTH = 7.2921150e-5


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


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


def ecef_to_eci_position(pos_ecef: np.ndarray, unix_time: float | np.ndarray) -> np.ndarray:
    pos_ecef = np.asarray(pos_ecef, dtype=float)
    theta = gmst_angle(unix_time)
    c = np.cos(theta)
    s = np.sin(theta)
    x = pos_ecef[..., 0]
    y = pos_ecef[..., 1]
    z = pos_ecef[..., 2]
    return np.stack((c * x - s * y, s * x + c * y, z), axis=-1)


def eci_to_ecef_position(pos_eci: np.ndarray, unix_time: float | np.ndarray) -> np.ndarray:
    pos_eci = np.asarray(pos_eci, dtype=float)
    theta = gmst_angle(unix_time)
    c = np.cos(theta)
    s = np.sin(theta)
    x = pos_eci[..., 0]
    y = pos_eci[..., 1]
    z = pos_eci[..., 2]
    return np.stack((c * x + s * y, -s * x + c * y, z), axis=-1)


def ecef_to_gcrs(ecef_m: np.ndarray, t_unix: float) -> np.ndarray:
    return ecef_to_eci_position(ecef_m, t_unix)


def geodetic_to_gcrs_posvel(lat_deg: float, lon_deg: float, alt_m: float, t_unix: float) -> tuple[np.ndarray, np.ndarray]:
    pos_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    vel_ecef = np.cross(omega_vec, pos_ecef)
    return ecef_to_eci_position(pos_ecef, t_unix), ecef_to_eci_position(vel_ecef, t_unix)


def gcrs_to_ecef(gcrs_m: np.ndarray, t_unix: float) -> np.ndarray:
    return eci_to_ecef_position(gcrs_m, t_unix)


def load_head_data(args):
    annotations, obs, used_stations, rejected = load_observations(
        args.annotations,
        args.snr_threshold_db,
        reject_outliers=not args.no_reject_outliers,
        delay_sigma=args.delay_sigma,
        doppler_sigma=args.doppler_sigma,
        station_start_trims=parse_station_start_trims(args.station_start_trim),
    )
    geom = station_geometry(args.data_dir, used_stations)
    return annotations, obs, used_stations, rejected, geom


def cache_head_station_states(obs: list[dict], geom: dict) -> list[dict]:
    cached = [dict(ob) for ob in obs]
    for station in sorted(set(ob["station"] for ob in cached)):
        idx = [i for i, ob in enumerate(cached) if ob["station"] == station]
        times = np.array([float(cached[i]["unix_time"]) for i in idx], dtype=float)
        g = geom[station]
        tx_ecef = geodetic_to_ecef(*g["gps_tx"])
        rx_ecef = geodetic_to_ecef(*g["gps_rx"])
        omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
        tx_vel_ecef = np.cross(omega_vec, tx_ecef)
        rx_vel_ecef = np.cross(omega_vec, rx_ecef)
        tx_pos = ecef_to_eci_position(np.broadcast_to(tx_ecef, (len(times), 3)), times)
        rx_pos = ecef_to_eci_position(np.broadcast_to(rx_ecef, (len(times), 3)), times)
        tx_vel = ecef_to_eci_position(np.broadcast_to(tx_vel_ecef, (len(times), 3)), times)
        rx_vel = ecef_to_eci_position(np.broadcast_to(rx_vel_ecef, (len(times), 3)), times)
        for jj, ii in enumerate(idx):
            cached[ii]["tx_gcrs_m"] = tx_pos[jj]
            cached[ii]["rx_gcrs_m"] = rx_pos[jj]
            cached[ii]["tx_gcrs_mps"] = tx_vel[jj]
            cached[ii]["rx_gcrs_mps"] = rx_vel[jj]
            cached[ii]["frequency_hz"] = float(g["frequency_hz"])
    return cached


def parse_station_start_trims(trim_specs):
    trims = {}
    for spec in trim_specs or []:
        if "=" not in spec:
            raise ValueError(f"Trim spec must have form station=seconds, got {spec!r}")
        station, value = spec.split("=", 1)
        trims[station.strip()] = float(value)
    return trims


def normalize_trail_row(row: dict) -> dict:
    unc = row.get("uncertainty") or {}
    sigma = float(unc.get("position_sigma_3d_m", 3000.0))
    sigma = float(np.clip(sigma, 1200.0, 12000.0))
    return {
        "station": row["station"],
        "unix_time": float(row["unix_time"]),
        "snr_db": float(row.get("snr_db", np.nan)),
        "aod_prior_offset_deg": float(row.get("aod_prior_offset_deg", 0.0)),
        "aod_score": float(row.get("aod_score", np.nan)),
        "position_ecef_m": np.asarray(row["position_ecef_m"], dtype=float),
        "position_enu_m": np.asarray(row["position_enu_m"], dtype=float),
        "lat_deg": float(row["lat_deg"]),
        "lon_deg": float(row["lon_deg"]),
        "alt_m": float(row["alt_m"]),
        "sigma_m": sigma,
    }


def trail_row_key(row: dict) -> tuple:
    return (
        row["station"],
        round(float(row["unix_time"]), 3),
        round(float(row["lat_deg"]), 7),
        round(float(row["lon_deg"]), 7),
        round(float(row["alt_m"]), 1),
    )


def load_all_trail_positions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [normalize_trail_row(row) for row in payload.get("positions", [])]


def load_trail_positions(path: Path, min_snr_db: float, max_aod_offset_deg: float, min_alt_km: float, max_alt_km: float):
    rows = []
    for row in load_all_trail_positions(path):
        if float(row.get("snr_db", -np.inf)) < min_snr_db:
            continue
        if abs(float(row.get("aod_prior_offset_deg", 0.0))) > max_aod_offset_deg:
            continue
        alt_km = float(row["alt_m"]) / 1000.0
        if alt_km < min_alt_km or alt_km > max_alt_km:
            continue
        rows.append(row)
    return rows


def params_from_head_fit(path: Path, t_ref: float) -> np.ndarray:
    fit = json.loads(path.read_text(encoding="utf-8"))
    p_ref_ecef = np.asarray(fit["position_ref_ecef_m"], dtype=float)
    v_ref_ecef = np.asarray(fit["velocity_ecef_mps"], dtype=float)
    p_ref = ecef_to_gcrs(p_ref_ecef, t_ref)
    # Transform velocity by finite differencing the existing head-only path in fast ECI.
    p1_ecef = p_ref_ecef + v_ref_ecef
    v_ref = ecef_to_gcrs(p1_ecef, t_ref + 1.0) - p_ref
    return np.r_[p_ref, v_ref]


def predict_head(params: np.ndarray, obs: list[dict], geom: dict, t_ref: float):
    r0 = params[:3]
    v = params[3:6]
    ranges = np.empty(len(obs), dtype=float)
    dopplers = np.empty(len(obs), dtype=float)
    for i, ob in enumerate(obs):
        t = float(ob["unix_time"])
        dt = t - t_ref
        x = r0 + v * dt
        if "tx_gcrs_m" in ob:
            tx = np.asarray(ob["tx_gcrs_m"], dtype=float)
            rx = np.asarray(ob["rx_gcrs_m"], dtype=float)
            vtx = np.asarray(ob["tx_gcrs_mps"], dtype=float)
            vrx = np.asarray(ob["rx_gcrs_mps"], dtype=float)
            frequency_hz = float(ob["frequency_hz"])
        else:
            g = geom[ob["station"]]
            tx, vtx = geodetic_to_gcrs_posvel(*g["gps_tx"], t)
            rx, vrx = geodetic_to_gcrs_posvel(*g["gps_rx"], t)
            frequency_hz = float(g["frequency_hz"])
        rtx = x - tx
        rrx = x - rx
        ntx = np.linalg.norm(rtx)
        nrx = np.linalg.norm(rrx)
        u_tx = rtx / ntx
        u_rx = rrx / nrx
        ranges[i] = ntx + nrx
        dopplers[i] = frequency_hz * (np.dot(u_tx, v - vtx) + np.dot(u_rx, v - vrx)) / C
    return ranges, dopplers


def trail_line_residual_vectors(params: np.ndarray, trail_gcrs: np.ndarray):
    r0 = params[:3]
    v = params[3:6]
    vv = float(np.dot(v, v))
    tau = ((trail_gcrs - r0[None, :]) @ v) / vv
    closest = r0[None, :] + tau[:, None] * v[None, :]
    return closest - trail_gcrs, tau


def low_altitude_trail_timing_residuals(params, trail_rows: list[dict], tau: np.ndarray, t_ref: float, args) -> np.ndarray:
    if not args.trail_timing_enable:
        return np.array([], dtype=float)
    alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in trail_rows], dtype=float)
    echo_time = np.array([float(r["unix_time"]) for r in trail_rows], dtype=float)
    keep = np.isfinite(alt_km) & np.isfinite(echo_time) & (alt_km <= args.trail_timing_max_alt_km)
    if not np.any(keep):
        return np.array([], dtype=float)
    predicted_echo_time = t_ref + tau[keep] + args.trail_timing_lag_s
    return (predicted_echo_time - echo_time[keep]) / args.trail_timing_sigma_s


def combined_residuals(params, head_obs, trail_rows, trail_gcrs, geom, t_ref, head_range_sigma, head_doppler_sigma, trail_sigma, args):
    pred_range, pred_doppler = predict_head(params, head_obs, geom, t_ref)
    meas_range = np.array([o["range_m"] for o in head_obs], dtype=float)
    meas_doppler = np.array([o["doppler_hz"] for o in head_obs], dtype=float)
    r_head = (pred_range - meas_range) / head_range_sigma
    d_head = (pred_doppler - meas_doppler) / head_doppler_sigma
    trail_vec, tau = trail_line_residual_vectors(params, trail_gcrs)
    r_trail = (trail_vec / trail_sigma[:, None]).ravel()
    r_trail_time = low_altitude_trail_timing_residuals(params, trail_rows, tau, t_ref, args)
    return np.r_[r_head, d_head, r_trail, r_trail_time]


def robust_reject_trail(params, trail_rows, trail_gcrs, max_sigma=3.5, min_threshold_m=4500.0):
    vec, tau = trail_line_residual_vectors(params, trail_gcrs)
    dist = np.linalg.norm(vec, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(dist))
    threshold = max(min_threshold_m, med + max_sigma * sigma)
    keep = dist <= threshold
    kept = [r for r, k in zip(trail_rows, keep, strict=True) if k]
    rejected = []
    for row, k, d, ti in zip(trail_rows, keep, dist, tau, strict=True):
        if not k:
            out = dict(row)
            out["line_residual_m"] = float(d)
            out["trajectory_time_unix"] = float(ti + json.loads(DEFAULT_HEAD_FIT.read_text())["t_ref_unix"]) if False else None
            rejected.append(out)
    return kept, rejected, threshold, dist


def covariance_from_jacobian(result):
    jac = np.asarray(result.jac, dtype=float)
    dof = max(1, jac.shape[0] - jac.shape[1])
    s2 = float(2.0 * result.cost / dof)
    cov = np.linalg.pinv(jac.T @ jac) * s2
    cov = 0.5 * (cov + cov.T)
    return cov, s2


def robust_std(x: np.ndarray, floor: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(floor)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad if mad > 0.0 else float(np.std(x))
    return float(max(sigma, floor))


def empirical_measurement_sigmas(params: np.ndarray, obs: list[dict], trail_rows: list[dict], trail_gcrs: np.ndarray, geom: dict, t_ref: float) -> dict:
    pr, pd = predict_head(params, obs, geom, t_ref)
    mr = np.array([o["range_m"] for o in obs], dtype=float)
    md = np.array([o["doppler_hz"] for o in obs], dtype=float)
    trail_vec, tau = trail_line_residual_vectors(params, trail_gcrs)
    alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in trail_rows], dtype=float)
    echo_time = np.array([float(r["unix_time"]) for r in trail_rows], dtype=float)
    low = np.isfinite(alt_km) & np.isfinite(echo_time) & (alt_km <= 60.0)
    timing_resid = (t_ref + tau[low] + 1.0) - echo_time[low]
    return {
        "head_range_sigma_m": robust_std(pr - mr, floor=150.0),
        "head_doppler_sigma_hz": robust_std(pd - md, floor=25.0),
        "trail_position_sigma_m": robust_std(trail_vec.ravel(), floor=500.0),
        "low_altitude_trail_timing_sigma_s": robust_std(timing_resid, floor=1.0),
    }


def sample_params(params, cov, n=500, seed=20260530):
    rng = np.random.default_rng(seed)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    clean = (vecs * vals) @ vecs.T
    return rng.multivariate_normal(params, clean, size=n)


def summarize_fit(params, cov, obs, trail_rows, trail_gcrs, geom, t_ref):
    pr, pd = predict_head(params, obs, geom, t_ref)
    mr = np.array([o["range_m"] for o in obs], dtype=float)
    md = np.array([o["doppler_hz"] for o in obs], dtype=float)
    tv, tau = trail_line_residual_vectors(params, trail_gcrs)
    trail_dist = np.linalg.norm(tv, axis=1)
    trail_alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in trail_rows], dtype=float)
    trail_echo_time = np.array([float(r["unix_time"]) for r in trail_rows], dtype=float)
    low_time = np.isfinite(trail_alt_km) & np.isfinite(trail_echo_time) & (trail_alt_km <= 60.0)
    p_ecef = gcrs_to_ecef(params[:3], t_ref)
    v_ecef = gcrs_to_ecef(params[:3] + params[3:6], t_ref + 1.0) - p_ecef
    lat, lon, alt = ecef_to_geodetic_wgs84(p_ecef)
    return {
        "range_residual_m": pr - mr,
        "doppler_residual_hz": pd - md,
        "trail_residual_m": trail_dist,
        "trail_tau_s": tau,
        "low_altitude_trail_timing_residual_s": (t_ref + tau[low_time] + 1.0) - trail_echo_time[low_time],
        "n_low_altitude_timing_constraints": int(np.sum(low_time)),
        "position_ref_ecef_m": p_ecef,
        "velocity_ecef_mps": v_ecef,
        "lat_ref_deg": lat,
        "lon_ref_deg": lon,
        "alt_ref_m": alt,
        "speed_mps": float(np.linalg.norm(params[3:6])),
        "one_sigma": np.sqrt(np.maximum(np.diag(cov), 0.0)),
    }


def write_json(out_dir, args, params, cov, s2, summary, obs, trail_rows, rejected, t_ref, empirical_sigmas=None):
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out = {
        "schema": "new_england_bolide.combined_head_trail_eci.v1",
        "t_ref_unix": float(t_ref),
        "t_ref_utc": iso_utc(t_ref),
        "parameter_frame": "ECI_GMST",
        "parameter_order": ["x_eci_m", "y_eci_m", "z_eci_m", "vx_eci_mps", "vy_eci_mps", "vz_eci_mps"],
        "fit_parameters_gcrs": params.tolist(),
        "covariance_gcrs": cov.tolist(),
        "one_sigma_gcrs": summary["one_sigma"].tolist(),
        "residual_variance": s2,
        "n_head_observations": len(obs),
        "n_trail_positions": len(trail_rows),
        "n_rejected_trail_positions": len(rejected),
        "position_ref_ecef_m": summary["position_ref_ecef_m"].tolist(),
        "velocity_ecef_mps": summary["velocity_ecef_mps"].tolist(),
        "geodetic_ref_wgs84": {
            "lat_deg": summary["lat_ref_deg"],
            "lon_deg": summary["lon_ref_deg"],
            "alt_m": summary["alt_ref_m"],
        },
        "speed_kmps": summary["speed_mps"] / 1000.0,
        "range_residual_m": {
            "rms": float(np.sqrt(np.mean(summary["range_residual_m"] ** 2))),
            "std": float(np.std(summary["range_residual_m"])),
            "p95_abs": float(np.percentile(np.abs(summary["range_residual_m"]), 95.0)),
        },
        "doppler_residual_hz": {
            "rms": float(np.sqrt(np.mean(summary["doppler_residual_hz"] ** 2))),
            "std": float(np.std(summary["doppler_residual_hz"])),
            "p95_abs": float(np.percentile(np.abs(summary["doppler_residual_hz"]), 95.0)),
        },
        "trail_residual_m": {
            "rms": float(np.sqrt(np.mean(summary["trail_residual_m"] ** 2))),
            "median": float(np.median(summary["trail_residual_m"])),
            "p95": float(np.percentile(summary["trail_residual_m"], 95.0)),
        },
        "low_altitude_trail_timing": {
            "max_alt_km": float(args.trail_timing_max_alt_km),
            "lag_s": float(args.trail_timing_lag_s),
            "sigma_s": float(args.trail_timing_sigma_s),
            "n_constraints": int(summary["n_low_altitude_timing_constraints"]),
            "residual_s": summary["low_altitude_trail_timing_residual_s"].tolist(),
            "rms_s": float(np.sqrt(np.mean(summary["low_altitude_trail_timing_residual_s"] ** 2)))
            if summary["n_low_altitude_timing_constraints"] > 0
            else None,
        },
        "rejected_trail_positions": [
            {
                "station": r["station"],
                "unix_time": r["unix_time"],
                "lat_deg": r["lat_deg"],
                "lon_deg": r["lon_deg"],
                "alt_m": r["alt_m"],
                "line_residual_m": r.get("line_residual_m"),
            }
            for r in rejected
        ],
        "empirical_measurement_sigmas": empirical_sigmas or {},
        "settings": settings,
    }
    out_path = out_dir / "combined_head_trail_eci_fit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def state_llh(params, t_grid, t_ref):
    llh = []
    for t in t_grid:
        ecef = gcrs_to_ecef(params[:3] + params[3:6] * (t - t_ref), t)
        llh.append(ecef_to_geodetic_wgs84(ecef))
    return np.asarray(llh)


def sample_altitudes(samples, t_grid, t_ref):
    out = []
    for p in samples:
        out.append(state_llh(p, t_grid, t_ref)[:, 2] / 1000.0)
    return np.asarray(out)


def transmitter_local_frame(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    first = next(iter(geom.values()))
    lat_deg, lon_deg, alt_m = first["gps_tx"]
    tx0 = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    rot = enu_to_ecef_matrix(lat_deg, lon_deg)
    return tx0, rot


def state_enu_km(params: np.ndarray, t_grid, t_ref: float, tx0: np.ndarray, rot: np.ndarray) -> np.ndarray:
    enu = []
    for t in t_grid:
        ecef = gcrs_to_ecef(params[:3] + params[3:6] * (float(t) - t_ref), float(t))
        enu.append((ecef - tx0) @ rot)
    return np.asarray(enu, dtype=float) / 1000.0


def sample_enu_components(samples: np.ndarray, t_grid, t_ref: float, tx0: np.ndarray, rot: np.ndarray) -> np.ndarray:
    out = []
    for p in samples:
        out.append(state_enu_km(p, t_grid, t_ref, tx0, rot))
    return np.asarray(out)


def plot_range_doppler(out, params, cov, obs, geom, t_ref):
    samples = sample_params(params, cov, n=300)
    stations = sorted(set(o["station"] for o in obs))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for si, station in enumerate(stations):
        color = f"C{si}"
        so = [o for o in obs if o["station"] == station]
        so.sort(key=lambda x: x["unix_time"])
        t = np.array([o["unix_time"] for o in so], dtype=float)
        tr = t - t_ref
        meas_r = np.array([o["range_m"] for o in so]) / 1000.0
        meas_d = np.array([o["doppler_hz"] for o in so]) / 1000.0
        pr, pd = predict_head(params, so, geom, t_ref)
        sr = []
        sd = []
        for p in samples:
            rr, dd = predict_head(p, so, geom, t_ref)
            sr.append(rr / 1000.0)
            sd.append(dd / 1000.0)
        sr = np.asarray(sr)
        sd = np.asarray(sd)
        rlo, rhi = np.percentile(sr, [2.5, 97.5], axis=0)
        dlo, dhi = np.percentile(sd, [2.5, 97.5], axis=0)
        label = short_station_name(station)
        axes[0].scatter(tr, meas_r, s=18, color=color, alpha=0.75, label=f"{label} measured")
        axes[0].plot(tr, pr / 1000.0, color=color, lw=2.0, label=f"{label} fit")
        axes[0].fill_between(tr, rlo, rhi, color=color, alpha=0.16, linewidth=0)
        axes[1].scatter(tr, meas_d, s=18, color=color, alpha=0.75, label=f"{label} measured")
        axes[1].plot(tr, pd / 1000.0, color=color, lw=2.0, label=f"{label} fit")
        axes[1].fill_between(tr, dlo, dhi, color=color, alpha=0.16, linewidth=0)
    axes[0].set_title("Combined head/trail ECI trajectory fit observables")
    axes[0].set_ylabel("Bistatic range (km)")
    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].set_xlabel("Seconds relative to fit reference time")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.6)
        ax.legend(ncol=2, fontsize=9)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def split_trail_display_rows(all_trail_rows: list[dict], used_rows: list[dict], rejected_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    used_keys = {trail_row_key(r) for r in used_rows}
    rejected_keys = {trail_row_key(r) for r in rejected_rows}
    excluded_rows = [r for r in all_trail_rows if trail_row_key(r) not in used_keys and trail_row_key(r) not in rejected_keys]
    return used_rows, rejected_rows, excluded_rows


def trail_projected_time_and_alt(params: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([], dtype=float), np.array([], dtype=float)
    tau = trail_line_residual_vectors(params, np.array([r["gcrs_ref_m"] for r in rows]))[1]
    alt_km = np.array([r["alt_m"] for r in rows], dtype=float) / 1000.0
    return tau, alt_km


def trail_projected_time_and_enu(params: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([], dtype=float), np.empty((0, 3), dtype=float)
    tau = trail_line_residual_vectors(params, np.array([r["gcrs_ref_m"] for r in rows]))[1]
    enu_km = np.array([r["position_enu_m"] for r in rows], dtype=float) / 1000.0
    return tau, enu_km


def scatter_trail_panels(axes, params: np.ndarray, used_rows: list[dict], rejected_rows: list[dict], excluded_rows: list[dict]) -> None:
    panel_values = [
        lambda rows: trail_projected_time_and_alt(params, rows),
        lambda rows: (trail_projected_time_and_enu(params, rows)[0], trail_projected_time_and_enu(params, rows)[1][:, 0]),
        lambda rows: (trail_projected_time_and_enu(params, rows)[0], trail_projected_time_and_enu(params, rows)[1][:, 1]),
    ]
    for ai, (ax, values_for_rows) in enumerate(zip(axes, panel_values, strict=True)):
        if excluded_rows:
            tau_ex, y_ex = values_for_rows(excluded_rows)
            ax.scatter(
                tau_ex,
                y_ex,
                s=24,
                facecolors="none",
                edgecolors="C1",
                linewidths=0.9,
                alpha=0.75,
                label="trail positions outside fit gates" if ai == 0 else None,
            )
        tau, y = values_for_rows(used_rows)
        ax.scatter(tau, y, s=18, color="0.25", alpha=0.45, label="trail positions used in fit" if ai == 0 else None)
        if rejected_rows:
            tau_rej, y_rej = values_for_rows(rejected_rows)
            ax.scatter(
                tau_rej,
                y_rej,
                s=42,
                color="C3",
                marker="x",
                linewidths=1.4,
                label="trail line outliers" if ai == 0 else None,
            )


def scatter_head_position_panels(axes, params: np.ndarray, obs: list[dict], t_ref: float, tx0: np.ndarray, rot: np.ndarray) -> None:
    if not obs:
        return
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    tr = t_head - t_ref
    llh = state_llh(params, t_head, t_ref)
    enu = state_enu_km(params, t_head, t_ref, tx0, rot)
    values = [llh[:, 2] / 1000.0, enu[:, 0], enu[:, 1]]
    for ai, ax in enumerate(axes):
        ax.scatter(
            tr,
            values[ai],
            s=58,
            color="C0",
            alpha=0.95,
            marker="x",
            linewidths=1.6,
            label="head-echo derived positions" if ai == 0 else None,
            zorder=7,
        )


def plot_altitude(out, params, cov, obs, trail_rows, t_ref, geom, all_trail_rows=None, rejected_trail=None):
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    trail_tau = trail_line_residual_vectors(params, np.array([r["gcrs_ref_m"] for r in trail_rows]))[1] if trail_rows else np.array([], dtype=float)
    t0 = min(float(np.min(t_head)), float(t_ref + np.min(trail_tau)) if trail_tau.size else float(np.min(t_head)))
    t1 = max(float(np.max(t_head)), float(t_ref + np.max(trail_tau)) if trail_tau.size else float(np.max(t_head)))
    t = np.linspace(t0, t1, 160)
    llh = state_llh(params, t, t_ref)
    samples = sample_params(params, cov, n=80)
    alt_s = sample_altitudes(samples, t, t_ref)
    lo, hi = np.percentile(alt_s, [2.5, 97.5], axis=0)
    tx0, rot = transmitter_local_frame(geom)
    enu_fit = state_enu_km(params, t, t_ref, tx0, rot)
    enu_samples = sample_enu_components(samples, t, t_ref, tx0, rot)
    enu_lo, enu_hi = np.percentile(enu_samples, [2.5, 97.5], axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.4), constrained_layout=True, sharex=True)
    tr = t - t_ref
    axes[0].fill_between(tr, lo, hi, color="C0", alpha=0.20, linewidth=0, label="95% fit uncertainty")
    for sample_alt in alt_s[:35]:
        axes[0].plot(tr, sample_alt, color="C0", alpha=0.055, lw=0.8)
    axes[0].plot(tr, llh[:, 2] / 1000.0, color="C0", lw=2.2, label="combined fit")
    for ax, idx in zip(axes[1:], [0, 1], strict=True):
        ax.fill_between(tr, enu_lo[:, idx], enu_hi[:, idx], color="C0", alpha=0.20, linewidth=0)
        for sample_enu in enu_samples[:35]:
            ax.plot(tr, sample_enu[:, idx], color="C0", alpha=0.055, lw=0.8)
        ax.plot(tr, enu_fit[:, idx], color="C0", lw=2.2)
    used_rows, rejected_rows, excluded_rows = split_trail_display_rows(all_trail_rows or trail_rows, trail_rows, rejected_trail or [])
    scatter_trail_panels(axes, params, used_rows, rejected_rows, excluded_rows)
    scatter_head_position_panels(axes, params, obs, t_ref, tx0, rot)
    axes[0].set_ylabel("Altitude (km)")
    axes[1].set_ylabel("East offset (km)")
    axes[2].set_ylabel("North offset (km)")
    axes[2].set_xlabel(f"Seconds relative to {iso_utc(t_ref)}")
    axes[0].set_title("Combined head/trail ECI trajectory in transmitter ENU frame")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].legend(loc="best")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_map(out, params, cov, obs, trail_rows, t_ref, all_trail_rows=None, rejected_trail=None):
    try:
        if "readline" not in sys.modules:
            readline_stub = types.ModuleType("readline")
            readline_stub.get_completer = lambda: None
            readline_stub.set_completer = lambda func: None
            readline_stub.parse_and_bind = lambda spec: None
            sys.modules["readline"] = readline_stub
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        ccrs = None
        cfeature = None
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    tau = trail_line_residual_vectors(params, np.array([r["gcrs_ref_m"] for r in trail_rows]))[1]
    t = np.linspace(min(np.min(t_head), t_ref + np.min(tau)) - 0.5, max(np.max(t_head), t_ref + np.max(tau)) + 0.5, 80)
    llh = state_llh(params, t, t_ref)
    samples = sample_params(params, cov, n=25)
    cloud = []
    for p in samples:
        cloud.append(state_llh(p, t, t_ref)[:, [1, 0]])
    cloud = np.reshape(np.asarray(cloud), (-1, 2))
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    hull_poly = None
    if len(cloud) > 4:
        cen = np.median(cloud, axis=0)
        rad = np.linalg.norm(cloud - cen[None, :], axis=1)
        cloud = cloud[rad <= np.percentile(rad, 95)]
        hull_poly = cloud[ConvexHull(cloud).vertices] if len(cloud) > 4 else None
    data_crs = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(9.4, 7.5), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=data_crs) if data_crs else fig.add_subplot(1, 1, 1)
    lon = llh[:, 1]
    lat = llh[:, 0]
    used_rows, rejected_rows, excluded_rows = split_trail_display_rows(all_trail_rows or trail_rows, trail_rows, rejected_trail or [])
    trail_lon = np.array([r["lon_deg"] for r in all_trail_rows or trail_rows])
    trail_lat = np.array([r["lat_deg"] for r in all_trail_rows or trail_rows])
    all_lon = np.r_[lon, trail_lon, hull_poly[:, 0] if hull_poly is not None else []]
    all_lat = np.r_[lat, trail_lat, hull_poly[:, 1] if hull_poly is not None else []]
    extent = [np.min(all_lon) - 0.35, np.max(all_lon) + 0.35, np.min(all_lat) - 0.25, np.max(all_lat) + 0.25]
    if data_crs:
        ax.set_extent(extent, crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none", zorder=0)
        ax.coastlines(resolution="10m", color="0.15", linewidth=0.9, zorder=3)
        ax.gridlines(draw_labels=True, color="0.7", linewidth=0.4)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if hull_poly is not None:
        ax.fill(hull_poly[:, 0], hull_poly[:, 1], color="C0", alpha=0.18, transform=data_crs, label="95% trajectory uncertainty")
    ax.plot(lon, lat, color="C3", lw=2.6, transform=data_crs, label="combined ECI fit")
    if excluded_rows:
        ax.scatter(
            [r["lon_deg"] for r in excluded_rows],
            [r["lat_deg"] for r in excluded_rows],
            s=42,
            facecolors="none",
            edgecolors="C1",
            marker="o",
            linewidths=1.25,
            alpha=0.88,
            transform=data_crs,
            label="trail positions outside fit gates",
        )
    ax.scatter(
        [r["lon_deg"] for r in used_rows],
        [r["lat_deg"] for r in used_rows],
        s=18,
        color="0.25",
        alpha=0.45,
        transform=data_crs,
        label="trail positions used in fit",
    )
    if rejected_rows:
        ax.scatter(
            [r["lon_deg"] for r in rejected_rows],
            [r["lat_deg"] for r in rejected_rows],
            s=58,
            color="C3",
            marker="x",
            linewidths=1.75,
            transform=data_crs,
            label="trail line outliers",
        )
    head_llh = state_llh(params, t_head, t_ref)
    ax.scatter(head_llh[:, 1], head_llh[:, 0], s=14, color="C3", alpha=0.55, transform=data_crs, label="head fit positions")
    ax.set_title("Combined head/trail ECI trajectory fit")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_fig_dir).mkdir(parents=True, exist_ok=True)
    print("Loading head echo observations", flush=True)
    annotations, obs, used_stations, rejected_head, geom = load_head_data(args)
    print("Caching station ECI states", flush=True)
    obs = cache_head_station_states(obs, geom)
    t_ref = float(np.median([o["unix_time"] for o in obs]))
    print("Loading trail positions", flush=True)
    all_trail_rows = load_all_trail_positions(Path(args.trail_json))
    trail_rows = load_trail_positions(Path(args.trail_json), args.trail_min_snr_db, args.max_aod_offset_deg, args.min_trail_alt_km, args.max_trail_alt_km)
    for row in all_trail_rows:
        row["gcrs_ref_m"] = ecef_to_gcrs(row["position_ecef_m"], t_ref)
    for row in trail_rows:
        row["gcrs_ref_m"] = ecef_to_gcrs(row["position_ecef_m"], t_ref)
    x0 = params_from_head_fit(Path(args.head_fit_json), t_ref)
    trail_gcrs = np.array([r["gcrs_ref_m"] for r in trail_rows])
    trail_sigma = np.array([r["sigma_m"] for r in trail_rows])
    print(f"First combined fit with {len(obs)} head observations and {len(trail_rows)} trail positions", flush=True)
    res1 = least_squares(
        combined_residuals,
        x0,
        args=(obs, trail_rows, trail_gcrs, geom, t_ref, args.head_range_sigma_m, args.head_doppler_sigma_hz, trail_sigma, args),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=args.max_nfev,
    )
    print("Rejecting trail outliers", flush=True)
    kept_rows, rejected_trail, reject_threshold, _dist = robust_reject_trail(res1.x, trail_rows, trail_gcrs)
    trail_rows = kept_rows
    trail_gcrs = np.array([r["gcrs_ref_m"] for r in trail_rows])
    trail_sigma = np.array([r["sigma_m"] for r in trail_rows])
    print(f"Second combined fit with {len(trail_rows)} trail positions", flush=True)
    res2 = least_squares(
        combined_residuals,
        res1.x,
        args=(obs, trail_rows, trail_gcrs, geom, t_ref, args.head_range_sigma_m, args.head_doppler_sigma_hz, trail_sigma, args),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=args.max_nfev,
    )
    empirical_sigmas = empirical_measurement_sigmas(res2.x, obs, trail_rows, trail_gcrs, geom, t_ref)
    print(
        "Empirical sigmas: "
        f"range {empirical_sigmas['head_range_sigma_m']:.1f} m, "
        f"doppler {empirical_sigmas['head_doppler_sigma_hz']:.1f} Hz, "
        f"trail {empirical_sigmas['trail_position_sigma_m']:.1f} m",
        flush=True,
    )
    trail_sigma_empirical = np.full(len(trail_rows), empirical_sigmas["trail_position_sigma_m"], dtype=float)
    print("Final combined fit with empirical measurement scales", flush=True)
    res3 = least_squares(
        combined_residuals,
        res2.x,
        args=(
            obs,
            trail_rows,
            trail_gcrs,
            geom,
            t_ref,
            empirical_sigmas["head_range_sigma_m"],
            empirical_sigmas["head_doppler_sigma_hz"],
            trail_sigma_empirical,
            args,
        ),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=args.max_nfev,
    )
    empirical_sigmas = empirical_measurement_sigmas(res3.x, obs, trail_rows, trail_gcrs, geom, t_ref)
    trail_sigma_empirical = np.full(len(trail_rows), empirical_sigmas["trail_position_sigma_m"], dtype=float)
    print("Computing covariance and summary", flush=True)
    cov, s2 = covariance_from_jacobian(res3)
    summary = summarize_fit(res3.x, cov, obs, trail_rows, trail_gcrs, geom, t_ref)
    json_path = write_json(out_dir, args, res3.x, cov, s2, summary, obs, trail_rows, rejected_trail, t_ref, empirical_sigmas=empirical_sigmas)
    if not args.skip_plots:
        print("Writing plots", flush=True)
        range_plot = out_dir / "combined_head_trail_range_doppler_fit.png"
        altitude_plot = out_dir / "combined_head_trail_altitude.png"
        map_plot = out_dir / "combined_head_trail_map.png"
        plot_range_doppler(range_plot, res3.x, cov, obs, geom, t_ref)
        plot_altitude(altitude_plot, res3.x, cov, obs, trail_rows, t_ref, geom, all_trail_rows=all_trail_rows, rejected_trail=rejected_trail)
        plot_map(map_plot, res3.x, cov, obs, trail_rows, t_ref, all_trail_rows=all_trail_rows, rejected_trail=rejected_trail)
        report_dir = Path(args.report_fig_dir)
        shutil.copy2(range_plot, report_dir / range_plot.name)
        shutil.copy2(altitude_plot, report_dir / altitude_plot.name)
        shutil.copy2(map_plot, report_dir / map_plot.name)
    print(f"Wrote {json_path}")
    print(f"Head range RMS {summary['range_residual_m'].std():.1f} m")
    print(f"Head Doppler RMS {summary['doppler_residual_hz'].std():.1f} Hz")
    print(f"Trail median residual {np.median(summary['trail_residual_m']):.1f} m; rejected {len(rejected_trail)} above {reject_threshold:.1f} m")
    print(f"Low-altitude trail timing constraints {summary['n_low_altitude_timing_constraints']}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--head-fit-json", type=Path, default=DEFAULT_HEAD_FIT)
    p.add_argument("--trail-json", type=Path, default=DEFAULT_TRAIL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-fig-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--snr-threshold-db", type=float, default=5.0)
    p.add_argument("--no-reject-outliers", action="store_true")
    p.add_argument("--delay-sigma", type=float, default=4.0)
    p.add_argument("--doppler-sigma", type=float, default=4.0)
    p.add_argument("--station-start-trim", action="append", default=["pomfret=0.05"])
    p.add_argument("--head-range-sigma-m", type=float, default=950.0)
    p.add_argument("--head-doppler-sigma-hz", type=float, default=510.0)
    p.add_argument("--trail-min-snr-db", type=float, default=8.0)
    p.add_argument("--max-aod-offset-deg", type=float, default=35.0)
    p.add_argument("--min-trail-alt-km", type=float, default=20.0)
    p.add_argument("--max-trail-alt-km", type=float, default=115.0)
    p.add_argument("--trail-timing-enable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trail-timing-max-alt-km", type=float, default=60.0)
    p.add_argument("--trail-timing-lag-s", type=float, default=1.0)
    p.add_argument("--trail-timing-sigma-s", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=200)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
