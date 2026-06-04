#!/usr/bin/env python3
"""Ballistic head/trail fit with central gravity and atmospheric drag."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from fit_combined_head_trail_eci import (
    C,
    DEFAULT_ANNOTATIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_HEAD_FIT,
    DEFAULT_TRAIL,
    cache_head_station_states,
    ecef_to_gcrs,
    gcrs_to_ecef,
    load_all_trail_positions,
    load_head_data,
    load_trail_positions,
    params_from_head_fit,
    pre_head_reference_time,
    predict_head,
    split_trail_display_rows,
    trail_row_key,
    transmitter_local_frame,
)
from estimate_head_echo_trajectory import ecef_to_geodetic_wgs84, geodetic_to_ecef
from station_config import load_all_station_configs, short_station_name


ROOT = Path(__file__).resolve().parent
DEFAULT_COMBINED_FIT = ROOT / "products" / "combined_head_trail_eci_fit" / "combined_head_trail_eci_fit.json"
DEFAULT_OUT_DIR = ROOT / "products" / "ballistic_head_trail_fit"
DEFAULT_REPORT_DIR = ROOT / "reports" / "figures"
DEFAULT_REPORT_TABLE_DIR = ROOT / "reports"
MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
OMEGA_EARTH = 7.2921150e-5
DEFAULT_FRAGMENT_DIAMETER_M = 0.10
DEFAULT_FRAGMENT_DENSITY_KG_M3 = 3000.0
DEFAULT_FRAGMENT_CD = 1.0
TRAIL_POSITION_EXCLUDE = [
    # Isolated Beverly AoD solution that appears as the first trail point in Figure 8.
    {"station": "beverly", "unix_time": 1780164384.88, "tol_s": 0.03},
]


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


def table_utc(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).strftime("%H:%M:%S.%f")[:-2]


def mjd2000_days(t_unix: float) -> float:
    return float(t_unix) / 86400.0 - 10957.5


def gcrs_to_geodetic(gcrs_m: np.ndarray, t_unix: float) -> tuple[float, float, float]:
    return ecef_to_geodetic_wgs84(gcrs_to_ecef(gcrs_m, t_unix))


def is_excluded_trail_row(row: dict) -> bool:
    for item in TRAIL_POSITION_EXCLUDE:
        if row.get("station") != item["station"]:
            continue
        if abs(float(row.get("unix_time", np.nan)) - item["unix_time"]) <= item["tol_s"]:
            return True
    return False


def atmosphere_velocity_gcrs(pos_gcrs: np.ndarray, t_unix: float) -> np.ndarray:
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    return np.cross(omega_vec, np.asarray(pos_gcrs, dtype=float))


def build_density_profile(times_unix: np.ndarray, trail_rows: list[dict], head_params: np.ndarray, t_ref: float) -> dict:
    t_min = float(np.min(times_unix))
    t_max = float(np.max(times_unix))
    p0 = head_params[:3] + head_params[3:6] * (0.5 * (t_min + t_max) - t_ref)
    lat, lon, _alt = gcrs_to_geodetic(p0, 0.5 * (t_min + t_max))
    alts = [r["alt_m"] for r in trail_rows]
    alts.extend([gcrs_to_geodetic(head_params[:3] + head_params[3:6] * (t - t_ref), t)[2] for t in times_unix])
    top = max(160e3, float(np.nanmax(alts)) + 30e3)
    alt_grid = np.linspace(0.0, top, 420)
    rho = None
    try:
        from pymsis import msis

        t64 = np.full(alt_grid.shape, float(0.5 * (t_min + t_max))).astype("datetime64[s]")
        data = msis.run(
            t64,
            np.full(alt_grid.shape, lat),
            np.full(alt_grid.shape, lon),
            alt_grid / 1000.0,
            geomagnetic_activity=-1,
        )
        arr = np.asarray(data).squeeze()
        rho = arr[..., 0].reshape(-1) if arr.ndim > 1 else arr.reshape(-1)
    except Exception:
        rho0 = 1.225
        scale_h = 7200.0
        rho = rho0 * np.exp(-alt_grid / scale_h)
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-30)
    return {"altitude_m": alt_grid, "rho_kg_m3": rho}


def density_at_alt(profile: dict, alt_m: float) -> float:
    alt = np.clip(float(alt_m), profile["altitude_m"][0], profile["altitude_m"][-1])
    return float(np.exp(np.interp(alt, profile["altitude_m"], np.log(profile["rho_kg_m3"]))))


def sphere_cd_area_over_mass(diameter_m: float, density_kg_m3: float, cd: float) -> float:
    radius = 0.5 * float(diameter_m)
    area = math.pi * radius**2
    mass = (4.0 / 3.0) * math.pi * radius**3 * float(density_kg_m3)
    return float(cd) * area / mass


def gravity_accel(pos_gcrs: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(pos_gcrs))
    return -MU_EARTH * pos_gcrs / (r**3)


def dynamics(t_rel: float, y: np.ndarray, t_ref: float, ballistic_coeff: float, density_profile: dict, use_drag: bool) -> np.ndarray:
    pos = y[:3]
    vel = y[3:]
    acc = gravity_accel(pos)
    if use_drag and ballistic_coeff > 0.0:
        t_abs = t_ref + float(t_rel)
        alt = float(np.linalg.norm(pos) - R_EARTH)
        rho = density_at_alt(density_profile, alt)
        v_atm = atmosphere_velocity_gcrs(pos, t_abs)
        v_rel = vel - v_atm
        speed = float(np.linalg.norm(v_rel))
        if speed > 0.0:
            acc = acc - 0.5 * rho * ballistic_coeff * speed * v_rel
    return np.r_[vel, acc]


def propagate_state(
    params: np.ndarray,
    t_eval_abs: np.ndarray,
    t_ref: float,
    density_profile: dict,
    use_drag: bool,
    integration_dt_s: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    t_eval_abs = np.asarray(t_eval_abs, dtype=float)
    t_sorted, inverse = np.unique(t_eval_abs, return_inverse=True)
    x0 = params[:6]
    ballistic_coeff = 10.0 ** float(params[6])
    t_rel = t_sorted - t_ref
    t_min = float(np.min(t_rel))
    t_max = float(np.max(t_rel))

    def rk4_step(tnow: float, ynow: np.ndarray, h: float) -> np.ndarray:
        k1 = dynamics(tnow, ynow, t_ref, ballistic_coeff, density_profile, use_drag)
        k2 = dynamics(tnow + 0.5 * h, ynow + 0.5 * h * k1, t_ref, ballistic_coeff, density_profile, use_drag)
        k3 = dynamics(tnow + 0.5 * h, ynow + 0.5 * h * k2, t_ref, ballistic_coeff, density_profile, use_drag)
        k4 = dynamics(tnow + h, ynow + h * k3, t_ref, ballistic_coeff, density_profile, use_drag)
        return ynow + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_to(limit: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        ts = [0.0]
        ys = [x0.copy()]
        tnow = 0.0
        ynow = x0.copy()
        while (step > 0 and tnow < limit - 1e-12) or (step < 0 and tnow > limit + 1e-12):
            h = min(abs(step), abs(limit - tnow)) * (1.0 if step > 0 else -1.0)
            ynow = rk4_step(tnow, ynow, h)
            tnow += h
            ts.append(tnow)
            ys.append(ynow.copy())
        return np.asarray(ts), np.asarray(ys)

    dt = float(integration_dt_s)
    ts_parts = []
    ys_parts = []
    if t_min < 0.0:
        tb, yb = integrate_to(t_min, -dt)
        ts_parts.append(tb[::-1])
        ys_parts.append(yb[::-1])
    else:
        ts_parts.append(np.array([], dtype=float))
        ys_parts.append(np.empty((0, 6), dtype=float))
    tf, yf = integrate_to(max(t_max, 0.0), dt)
    ts_parts.append(tf)
    ys_parts.append(yf)
    tg = np.concatenate(ts_parts)
    yg = np.vstack(ys_parts)
    keep = np.r_[True, np.diff(tg) > 1e-12]
    tg = tg[keep]
    yg = yg[keep]
    y_sorted = np.column_stack([np.interp(t_rel, tg, yg[:, i]) for i in range(6)])
    out = y_sorted[inverse]
    return out[:, :3], out[:, 3:]


def interp_model(t_grid: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_query, t_grid, values[:, i]) for i in range(values.shape[1])])


def head_predictions_from_states(pos: np.ndarray, vel: np.ndarray, obs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.empty(len(obs), dtype=float)
    dopplers = np.empty(len(obs), dtype=float)
    for i, ob in enumerate(obs):
        tx = np.asarray(ob["tx_gcrs_m"], dtype=float)
        rx = np.asarray(ob["rx_gcrs_m"], dtype=float)
        vtx = np.asarray(ob["tx_gcrs_mps"], dtype=float)
        vrx = np.asarray(ob["rx_gcrs_mps"], dtype=float)
        rtx = pos[i] - tx
        rrx = pos[i] - rx
        ntx = np.linalg.norm(rtx)
        nrx = np.linalg.norm(rrx)
        u_tx = rtx / ntx
        u_rx = rrx / nrx
        ranges[i] = ntx + nrx
        dopplers[i] = float(ob["frequency_hz"]) * (np.dot(u_tx, vel[i] - vtx) + np.dot(u_rx, vel[i] - vrx)) / C
    return ranges, dopplers


def closest_curve_vectors(points: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = curve[None, :, :] - points[:, None, :]
    d2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(d2, axis=1)
    return curve[idx] - points, idx


def residuals(params, obs, trail_rows, t_ref, t_model, density_profile, args):
    try:
        pos_head, vel_head = propagate_state(params, np.array([o["unix_time"] for o in obs]), t_ref, density_profile, use_drag=not args.no_drag)
        pos_model, _vel_model = propagate_state(params, t_model, t_ref, density_profile, use_drag=not args.no_drag)
    except Exception:
        return np.full(2 * len(obs) + 3 * len(trail_rows), 1e8, dtype=float)
    pred_range, pred_doppler = head_predictions_from_states(pos_head, vel_head, obs)
    meas_range = np.array([o["range_m"] for o in obs], dtype=float)
    meas_doppler = np.array([o["doppler_hz"] for o in obs], dtype=float)
    r_head = (pred_range - meas_range) / args.head_range_sigma_m
    d_head = (pred_doppler - meas_doppler) / args.head_doppler_sigma_hz
    trail_points = np.array([r["gcrs_ref_m"] for r in trail_rows], dtype=float)
    trail_vec, _idx = closest_curve_vectors(trail_points, pos_model)
    trail_sigma = np.array([r["sigma_m"] for r in trail_rows], dtype=float)
    return np.r_[r_head, d_head, (trail_vec / trail_sigma[:, None]).ravel()]


def parameters_from_delta(delta: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    return np.r_[base_state[:3] + delta[:3], base_state[3:6] + delta[3:6], delta[6]]


def shift_eci_state_linear(state: np.ndarray, source_t_ref: float, target_t_ref: float) -> np.ndarray:
    state = np.asarray(state[:6], dtype=float)
    dt = float(target_t_ref) - float(source_t_ref)
    return np.r_[state[:3] + dt * state[3:6], state[3:6]]


def residuals_delta(delta, base_state, obs, trail_rows, t_ref, t_model, density_profile, args):
    return residuals(parameters_from_delta(delta, base_state), obs, trail_rows, t_ref, t_model, density_profile, args)


def write_json(path: Path, result, params, obs, trail_rows, t_ref, t_model, density_profile, args):
    pos_model, vel_model = propagate_state(params, t_model, t_ref, density_profile, use_drag=not args.no_drag)
    speed = np.linalg.norm(vel_model, axis=1)
    llh = np.array([gcrs_to_geodetic(p, t) for p, t in zip(pos_model, t_model, strict=True)])
    trail_points = np.array([r["gcrs_ref_m"] for r in trail_rows], dtype=float)
    trail_vec, _idx = closest_curve_vectors(trail_points, pos_model)
    out = {
        "schema": "new_england_bolide.ballistic_head_trail.v1",
        "t_ref_unix": float(t_ref),
        "t_ref_utc": iso_utc(t_ref),
        "parameter_order": ["x_gcrs_m", "y_gcrs_m", "z_gcrs_m", "vx_gcrs_mps", "vy_gcrs_mps", "vz_gcrs_mps", "log10_cd_area_over_mass_m2_kg"],
        "fit_parameters": params.tolist(),
        "fit_covariance": getattr(result, "covariance", np.full((len(params), len(params)), np.nan)).tolist(),
        "cd_area_over_mass_m2_kg": float(10.0 ** params[6]),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message,
        "n_head_observations": len(obs),
        "n_trail_positions": len(trail_rows),
        "model": {
            "time_unix": t_model.tolist(),
            "lat_deg": llh[:, 0].tolist(),
            "lon_deg": llh[:, 1].tolist(),
            "alt_m": llh[:, 2].tolist(),
            "speed_mps": speed.tolist(),
            "position_gcrs_m": pos_model.tolist(),
            "velocity_gcrs_mps": vel_model.tolist(),
        },
        "trail_residual_m": {
            "median": float(np.median(np.linalg.norm(trail_vec, axis=1))),
            "p95": float(np.percentile(np.linalg.norm(trail_vec, axis=1), 95.0)),
        },
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def interpolate_crossing_time(t: np.ndarray, y: np.ndarray, target: float) -> float | None:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(1, len(t)):
        y0 = y[i - 1] - target
        y1 = y[i] - target
        if y0 == 0.0:
            return float(t[i - 1])
        if y0 * y1 <= 0.0:
            frac = y0 / (y0 - y1) if y0 != y1 else 0.0
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def geodetic_path_from_states(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.array([gcrs_to_geodetic(p, tt) for p, tt in zip(pos, t, strict=True)], dtype=float)


def state_at_time(
    params: np.ndarray,
    t_abs: float,
    t_ref: float,
    density_profile: dict,
    use_drag: bool,
    integration_dt_s: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    pos, vel = propagate_state(
        params,
        np.array([t_abs], dtype=float),
        t_ref,
        density_profile,
        use_drag=use_drag,
        integration_dt_s=integration_dt_s,
    )
    return pos[0], vel[0]


def fragment_impact_path(
    params: np.ndarray,
    t_ref: float,
    density_profile: dict,
    args,
    sample_dt_s: float = 0.5,
    max_duration_s: float | None = None,
    fragment_diameter_m: float | None = None,
) -> dict | None:
    if max_duration_s is None:
        max_duration_s = float(args.fragment_impact_max_duration_s)
    diameter_m = float(args.fragment_diameter_m if fragment_diameter_m is None else fragment_diameter_m)
    pre_dt_s = min(0.1, sample_dt_s)
    model_t = np.arange(t_ref - 5.0, t_ref + args.fragment_breakup_search_s + pre_dt_s, pre_dt_s)
    pos_pre, _vel_pre = propagate_state(params, model_t, t_ref, density_profile, use_drag=not args.no_drag)
    llh_pre = geodetic_path_from_states(pos_pre, model_t)
    t_break = interpolate_crossing_time(model_t, llh_pre[:, 2], args.fragment_breakup_alt_km * 1000.0)
    if t_break is None:
        return None
    pos_break, vel_break = state_at_time(params, t_break, t_ref, density_profile, use_drag=not args.no_drag)
    cd_area_mass = sphere_cd_area_over_mass(diameter_m, args.fragment_density_kg_m3, args.fragment_cd)

    def ode(t_rel: float, y: np.ndarray) -> np.ndarray:
        return dynamics(t_rel, y, t_break, cd_area_mass, density_profile, use_drag=True)

    def sea_level_event(t_rel: float, y: np.ndarray) -> float:
        return float(gcrs_to_geodetic(y[:3], t_break + float(t_rel))[2])

    sea_level_event.terminal = True
    sea_level_event.direction = -1.0
    sol = solve_ivp(
        ode,
        (0.0, float(max_duration_s)),
        np.r_[pos_break, vel_break],
        method="RK45",
        rtol=args.fragment_rtol,
        atol=args.fragment_atol,
        max_step=args.fragment_integration_dt_s,
        dense_output=True,
        events=sea_level_event,
    )
    if len(sol.t_events[0]) == 0 or sol.y_events[0].size == 0:
        return None
    t_impact = float(t_break + sol.t_events[0][0])
    y_impact = sol.y_events[0][0]
    pos_impact = y_impact[:3]
    vel_impact = y_impact[3:]
    llh_impact = np.array(gcrs_to_geodetic(pos_impact, t_impact), dtype=float)
    rel_frag = np.arange(0.0, sol.t_events[0][0] + sample_dt_s, sample_dt_s)
    rel_frag = rel_frag[rel_frag <= sol.t_events[0][0]]
    if len(rel_frag) == 0 or rel_frag[-1] < sol.t_events[0][0]:
        rel_frag = np.r_[rel_frag, sol.t_events[0][0]]
    y_frag = sol.sol(rel_frag).T
    pos_frag = y_frag[:, :3]
    t_frag = t_break + rel_frag
    llh_frag = geodetic_path_from_states(pos_frag, t_frag)
    keep_pre = model_t <= t_break
    return {
        "t_break_unix": float(t_break),
        "t_impact_unix": float(t_impact),
        "breakup_alt_km": float(args.fragment_breakup_alt_km),
        "fragment_cd_area_over_mass_m2_kg": float(cd_area_mass),
        "fragment_diameter_m": diameter_m,
        "fragment_density_kg_m3": float(args.fragment_density_kg_m3),
        "fragment_cd": float(args.fragment_cd),
        "pre_break_llh": llh_pre[keep_pre],
        "pre_break_time_unix": model_t[keep_pre],
        "fragment_llh": llh_frag,
        "fragment_time_unix": t_frag,
        "impact_llh": llh_impact,
        "impact_speed_mps": float(np.linalg.norm(vel_impact)),
    }


def estimate_covariance(result, n_params: int) -> np.ndarray:
    jac = np.asarray(result.jac, dtype=float)
    n_resid = jac.shape[0]
    dof = max(1, n_resid - n_params)
    s2 = 2.0 * float(result.cost) / dof
    try:
        return np.linalg.pinv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        return np.full((n_params, n_params), np.nan)


def sample_ballistic_params(params: np.ndarray, cov: np.ndarray, n: int = 120, seed: int = 20260603) -> np.ndarray:
    if not np.all(np.isfinite(cov)):
        return np.empty((0, len(params)), dtype=float)
    cov = 0.5 * (cov + cov.T)
    jitter = np.eye(cov.shape[0]) * max(1e-12, 1e-12 * float(np.nanmax(np.diag(cov))))
    rng = np.random.default_rng(seed)
    try:
        return rng.multivariate_normal(params, cov + jitter, size=n, check_valid="ignore")
    except np.linalg.LinAlgError:
        return np.empty((0, len(params)), dtype=float)


def impact_sample_diameters(args, n: int, seed: int = 20260604) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    lo = float(args.fragment_diameter_min_m)
    hi = float(args.fragment_diameter_max_m)
    if lo <= 0.0 or hi <= 0.0 or hi < lo:
        raise ValueError("fragment diameter uncertainty range must be positive and ordered")
    rng = np.random.default_rng(seed)
    # The fragment size is poorly known over orders of magnitude, so sample it
    # uniformly in log diameter rather than overweighting the largest fragments.
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def impact_summary(nominal: dict, impact_samples: list[dict], t_ref: float, args) -> dict:
    if impact_samples:
        llh = np.array([s["impact_llh"] for s in impact_samples], dtype=float)
        t_impact = np.array([s["t_impact_unix"] for s in impact_samples], dtype=float)
        t_break = np.array([s["t_break_unix"] for s in impact_samples], dtype=float)
        diameter = np.array([s["fragment_diameter_m"] for s in impact_samples], dtype=float)
        speed = np.array([s["impact_speed_mps"] for s in impact_samples], dtype=float)
    else:
        llh = np.asarray(nominal["impact_llh"], dtype=float)[None, :]
        t_impact = np.array([float(nominal["t_impact_unix"])], dtype=float)
        t_break = np.array([float(nominal["t_break_unix"])], dtype=float)
        diameter = np.array([float(nominal["fragment_diameter_m"])], dtype=float)
        speed = np.array([float(nominal["impact_speed_mps"])], dtype=float)

    def pct(x):
        return np.percentile(np.asarray(x, dtype=float), [2.5, 50.0, 97.5])

    lat_p = pct(llh[:, 0])
    lon_p = pct(llh[:, 1])
    time_p = pct(t_impact)
    duration_p = pct(t_impact - t_break)
    diameter_p = pct(diameter)
    speed_p = pct(speed)
    nominal_llh = np.asarray(nominal["impact_llh"], dtype=float)
    return {
        "nominal_lat_deg": float(nominal_llh[0]),
        "nominal_lon_deg": float(nominal_llh[1]),
        "nominal_alt_m": float(nominal_llh[2]),
        "nominal_impact_unix": float(nominal["t_impact_unix"]),
        "nominal_impact_utc": iso_utc(nominal["t_impact_unix"]),
        "nominal_breakup_unix": float(nominal["t_break_unix"]),
        "nominal_breakup_utc": iso_utc(nominal["t_break_unix"]),
        "nominal_fall_duration_s": float(nominal["t_impact_unix"] - nominal["t_break_unix"]),
        "nominal_diameter_m": float(nominal["fragment_diameter_m"]),
        "nominal_impact_speed_mps": float(nominal["impact_speed_mps"]),
        "sample_count": int(len(impact_samples)),
        "diameter_range_m": [float(args.fragment_diameter_min_m), float(args.fragment_diameter_max_m)],
        "diameter_percentile_m": diameter_p.tolist(),
        "lat_percentile_deg": lat_p.tolist(),
        "lon_percentile_deg": lon_p.tolist(),
        "impact_time_percentile_unix": time_p.tolist(),
        "impact_time_percentile_utc": [iso_utc(x) for x in time_p],
        "fall_duration_percentile_s": duration_p.tolist(),
        "impact_speed_percentile_mps": speed_p.tolist(),
        "time_reference_unix": float(t_ref),
    }


def write_fragment_impact_table(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def interval(values, fmt: str) -> str:
        lo, med, hi = values
        return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"

    t_lo, t_med, t_hi = summary["impact_time_percentile_unix"]
    text = rf"""\begin{{table}}[ht]
  \centering
  \caption{{Sea-level impact estimate for the nominal 10 cm fragment and for fragment diameters sampled between 1 and 100 cm.  The sampled interval includes the linearized ballistic fit covariance and the unknown post-30 km fragment size at fixed density \(3000\ \mathrm{{kg\,m^{{-3}}}}\).}}
  \label{{tab:fragment-impact}}
  \resizebox{{\linewidth}}{{!}}{{%
  \begin{{tabular}}{{lll}}
    \toprule
    Quantity & Nominal 10 cm fragment & 95\% sampled interval \\
    \midrule
    Latitude (deg) & {summary["nominal_lat_deg"]:.4f} & {interval(summary["lat_percentile_deg"], "{:.4f}")} \\
    Longitude (deg) & {summary["nominal_lon_deg"]:.4f} & {interval(summary["lon_percentile_deg"], "{:.4f}")} \\
    Sea-level time (UTC) & {table_utc(summary["nominal_impact_unix"])} & {table_utc(t_med)} [{table_utc(t_lo)}, {table_utc(t_hi)}] \\
    Fall time below 30 km (s) & {summary["nominal_fall_duration_s"]:.0f} & {interval(summary["fall_duration_percentile_s"], "{:.0f}")} \\
    Impact speed (m s$^{{-1}}$) & {summary["nominal_impact_speed_mps"]:.0f} & {interval(summary["impact_speed_percentile_mps"], "{:.0f}")} \\
    Fragment diameter (cm) & {100.0 * summary["nominal_diameter_m"]:.1f} & {interval([100.0 * x for x in summary["diameter_percentile_m"]], "{:.1f}")} \\
    \bottomrule
  \end{{tabular}}
  }}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def tex_sci(value: float, digits: int = 3) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    if -2 <= exponent < 4:
        decimals = max(0, digits - 1 - exponent)
        return f"{value:.{decimals}f}"
    return rf"{mantissa:.{digits - 1}f}\times 10^{{{exponent}}}"


def write_initial_state_json(path: Path, params: np.ndarray, cov: np.ndarray, t_ref: float) -> dict:
    state_order = ["x_gcrs_m", "y_gcrs_m", "z_gcrs_m", "vx_gcrs_mps", "vy_gcrs_mps", "vz_gcrs_mps"]
    state_units = ["m", "m", "m", "m s^-1", "m s^-1", "m s^-1"]
    cov6 = np.asarray(cov, dtype=float)[:6, :6]
    payload = {
        "schema": "new_england_bolide.ballistic_initial_state.v1",
        "description": "Initial inertial state from the ballistic head/trail fit, evaluated 4 s before the first retained head echo.",
        "time_unix": float(t_ref),
        "time_utc": iso_utc(t_ref),
        "time_mjd2000_days_utc": mjd2000_days(t_ref),
        "time_scale": "UTC",
        "frame": "GCRS/ECI_GMST",
        "state_order": state_order,
        "state_units": state_units,
        "state": np.asarray(params[:6], dtype=float).tolist(),
        "covariance": cov6.tolist(),
        "covariance_units": [[f"{ui} {uj}" for uj in state_units] for ui in state_units],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_initial_state_table_from_json(json_path: Path, tex_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    labels = ["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$"]
    state = np.asarray(payload["state"], dtype=float)
    cov = np.asarray(payload["covariance"], dtype=float)

    def scaled(value: float, digits: int = 4) -> str:
        return f"{value / 1.0e4:.{digits}g}"

    time_utc = payload["time_utc"].replace("T", " ").replace("Z", " UTC")
    mjd2000 = payload["time_mjd2000_days_utc"]
    lines = [
        "\\begin{table}\n",
        "  \\centering\n",
        f"  \\caption{{Initial GCRS/ECI state used for orbit propagation at {time_utc} (MJD2000 UTC {mjd2000:.8f}), 4 s before the first retained head echo.  State order is $(x,y,z,v_x,v_y,v_z)$ with SI units $(\\mathrm{{m}},\\mathrm{{m}},\\mathrm{{m}},\\mathrm{{m\\,s^{{-1}}}},\\mathrm{{m\\,s^{{-1}}}},\\mathrm{{m\\,s^{{-1}}}})$; covariance entries use the corresponding unit products.}}\n",
        "  \\label{tab:ballistic-initial-state}\n",
        "  \\begingroup\n",
        "  \\small\n",
        "  \\setlength{\\tabcolsep}{2.5pt}\n",
        "  \\begin{tabular}{@{}ll|rrrrrr@{}}\n",
        "    \\toprule\n",
        "     & Mean / $10^4$ & \\multicolumn{6}{c@{}}{Covariance / $10^4$} \\\\\n",
        "    \\midrule\n",
        "    Variable & Estimate & $x$ & $y$ & $z$ & $v_x$ & $v_y$ & $v_z$ \\\\\n",
        "    \\midrule\n",
    ]
    for label, mean, row in zip(labels, state, cov, strict=True):
        cov_values = " & ".join(f"${scaled(v)}$" for v in row)
        lines.append(f"    {label} & ${scaled(mean, 5)}$ & {cov_values} \\\\\n")
    lines.extend(
        [
            "    \\bottomrule\n",
            "  \\end{tabular}\n",
            "  \\endgroup\n",
            "\\end{table}\n",
        ]
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("".join(lines), encoding="utf-8")


def enu_from_llh_time(llh: np.ndarray, t_unix: np.ndarray, tx0: np.ndarray, rot: np.ndarray) -> np.ndarray:
    # For this local display, ENU offsets are referenced to the fixed transmitter ECEF frame.
    ecef = np.array([geodetic_to_ecef(lat, lon, alt) for lat, lon, alt in llh], dtype=float)
    return (ecef - tx0[None, :]) @ rot / 1000.0


def plot_altitude_enu(
    out: Path,
    payload: dict,
    trail_rows: list[dict],
    tx0: np.ndarray,
    rot: np.ndarray,
    t_ref: float,
    impact_payload: dict | None = None,
    params: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    obs: list[dict] | None = None,
    density_profile: dict | None = None,
    args=None,
):
    model = payload["model"]
    t = np.array(model["time_unix"], dtype=float)
    pos = np.array(model["position_gcrs_m"], dtype=float)
    enu = np.array([(gcrs_to_ecef(p, tt) - tx0) @ rot for p, tt in zip(pos, t, strict=True)]) / 1000.0
    alt = np.array(model["alt_m"], dtype=float) / 1000.0
    tau = []
    for r in trail_rows:
        d2 = np.sum((pos - np.asarray(r["gcrs_ref_m"], dtype=float)[None, :]) ** 2, axis=1)
        tau.append(t[int(np.argmin(d2))] - t_ref)
    tau = np.asarray(tau)
    trail_enu = np.array([r["position_enu_m"] for r in trail_rows], dtype=float) / 1000.0
    trail_alt = np.array([r["alt_m"] for r in trail_rows], dtype=float) / 1000.0
    trail_sigma_enu = np.array(
        [
            (r.get("uncertainty") or {}).get("position_sigma_enu_m", [np.nan, np.nan, np.nan])
            for r in trail_rows
        ],
        dtype=float,
    ) / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.4), sharex=True, constrained_layout=True)
    tr = t - t_ref
    if params is not None and cov is not None and density_profile is not None and args is not None:
        samples = sample_ballistic_params(params, cov, n=args.altitude_uncertainty_samples)
        sample_values = []
        for p in samples:
            try:
                pos_s, _vel_s = propagate_state(p, t, t_ref, density_profile, use_drag=not args.no_drag)
                llh_s = geodetic_path_from_states(pos_s, t)
                enu_s = np.array([(gcrs_to_ecef(x, tt) - tx0) @ rot for x, tt in zip(pos_s, t, strict=True)]) / 1000.0
                vals = np.stack([llh_s[:, 2] / 1000.0, enu_s[:, 0], enu_s[:, 1]], axis=0)
                if np.all(np.isfinite(vals)):
                    sample_values.append(vals)
            except Exception:
                continue
        if sample_values:
            sample_values = np.asarray(sample_values, dtype=float)
            lo, hi = np.percentile(sample_values, [2.5, 97.5], axis=0)
            for ai, ax in enumerate(axes):
                ax.fill_between(
                    tr,
                    lo[ai],
                    hi[ai],
                    color="C2",
                    alpha=0.16,
                    linewidth=0,
                    label="95% fit uncertainty" if ai == 0 else None,
                )
                for vals in sample_values[:30, ai, :]:
                    ax.plot(tr, vals, color="C2", alpha=0.04, lw=0.8)
    for ax, y, scat, sigma, ylabel in [
        (axes[0], alt, trail_alt, trail_sigma_enu[:, 2], "Altitude (km)"),
        (axes[1], enu[:, 0], trail_enu[:, 0], trail_sigma_enu[:, 0], "East offset (km)"),
        (axes[2], enu[:, 1], trail_enu[:, 1], trail_sigma_enu[:, 1], "North offset (km)"),
    ]:
        ax.plot(tr, y, color="C2", lw=2.2, label="ballistic fit")
        ax.errorbar(
            tau,
            scat,
            yerr=sigma,
            fmt="o",
            ms=4.2,
            color="0.25",
            ecolor="0.05",
            elinewidth=0.95,
            capsize=2.2,
            capthick=0.9,
            alpha=0.82,
            label="trail positions with 1$\\sigma$ errors",
            zorder=6,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].axhline(30.0, color="0.35", lw=0.9, ls="--", alpha=0.7)
    if params is not None and obs is not None and density_profile is not None and args is not None:
        t_head = np.array([o["unix_time"] for o in obs], dtype=float)
        head_pos, _head_vel = propagate_state(params, t_head, t_ref, density_profile, use_drag=not args.no_drag)
        head_llh = geodetic_path_from_states(head_pos, t_head)
        head_enu = np.array([(gcrs_to_ecef(p, tt) - tx0) @ rot for p, tt in zip(head_pos, t_head, strict=True)]) / 1000.0
        head_tr = t_head - t_ref
        head_vals = [head_llh[:, 2] / 1000.0, head_enu[:, 0], head_enu[:, 1]]
        for ax, y in zip(axes, head_vals, strict=True):
            ax.scatter(head_tr, y, s=54, color="C0", marker="x", linewidths=1.5, alpha=0.95, label="head-echo positions")
    axes[2].set_xlabel(f"Seconds relative to {iso_utc(t_ref)}")
    axes[0].legend(loc="best")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def robust_std(x: np.ndarray, floor: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(floor)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad if mad > 0.0 else float(np.std(x))
    return float(max(sigma, floor))


def snr_binned_scatter(snr_db: np.ndarray, values: np.ndarray, n_bins: int = 4) -> list[dict]:
    snr_db = np.asarray(snr_db, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(snr_db) & np.isfinite(values)
    snr_db = snr_db[good]
    values = values[good]
    if snr_db.size < 8:
        return []
    edges = np.unique(np.quantile(snr_db, np.linspace(0.0, 1.0, min(n_bins, max(2, snr_db.size // 5)) + 1)))
    if edges.size < 3:
        return []
    rows = []
    for i in range(edges.size - 1):
        if i == edges.size - 2:
            keep = (snr_db >= edges[i]) & (snr_db <= edges[i + 1])
        else:
            keep = (snr_db >= edges[i]) & (snr_db < edges[i + 1])
        if np.sum(keep) < 3:
            continue
        rows.append(
            {
                "snr_db": float(np.median(snr_db[keep])),
                "std": robust_std(values[keep]),
                "n": int(np.sum(keep)),
            }
        )
    return rows


def plot_trail_position_error_debug(
    out: Path,
    payload: dict,
    trail_rows: list[dict],
    t_ref: float,
    tx0: np.ndarray,
    rot: np.ndarray,
):
    model = payload["model"]
    t = np.array(model["time_unix"], dtype=float)
    pos = np.array(model["position_gcrs_m"], dtype=float)
    trail_points = np.array([r["gcrs_ref_m"] for r in trail_rows], dtype=float)
    residual_gcrs, idx = closest_curve_vectors(trail_points, pos)
    residual_enu = np.array([(gcrs_to_ecef(v, t[int(ii)]) - gcrs_to_ecef(np.zeros(3), t[int(ii)])) @ rot for v, ii in zip(residual_gcrs, idx, strict=True)])
    residual_norm_km = np.linalg.norm(residual_enu, axis=1) / 1000.0
    snr_db = np.array([r.get("snr_db", np.nan) for r in trail_rows], dtype=float)
    formal_sigma_3d_km = np.array(
        [
            (r.get("uncertainty") or {}).get("position_sigma_3d_m", np.nan)
            for r in trail_rows
        ],
        dtype=float,
    ) / 1000.0
    ls_sigma_deg = np.array([r.get("aod_lstsq_sigma_angular_deg", np.nan) for r in trail_rows], dtype=float)
    grid_to_fit_deg = np.array([r.get("aod_lstsq_grid_to_fit_offset_deg", np.nan) for r in trail_rows], dtype=float)
    ls_success = np.array([bool(r.get("aod_lstsq_success", False)) for r in trail_rows], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    sc = ax.scatter(snr_db, residual_norm_km, c=ls_success, cmap="coolwarm", s=28, alpha=0.78, edgecolor="0.2", linewidth=0.25)
    bins = snr_binned_scatter(snr_db, residual_norm_km * 1000.0)
    if bins:
        ax.plot([b["snr_db"] for b in bins], [b["std"] / 1000.0 for b in bins], "o-", color="C1", lw=2.0, ms=7, label="binned robust std")
    ax.set_ylabel("3-D position residual (km)")
    ax.set_title("Trail position residual vs SNR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.scatter(snr_db, formal_sigma_3d_km, s=28, color="C2", alpha=0.72, edgecolor="0.2", linewidth=0.25)
    bins = snr_binned_scatter(snr_db, formal_sigma_3d_km * 1000.0)
    if bins:
        ax.plot([b["snr_db"] for b in bins], [b["std"] / 1000.0 for b in bins], "o-", color="C1", lw=2.0, ms=7, label="binned robust std")
    ax.set_ylabel("Formal 3-D position $1\\sigma$ (km)")
    ax.set_title("Propagated AoD/range uncertainty")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    labels = ["East", "North", "Up"]
    colors = ["C0", "C1", "C3"]
    for ci, (label, color) in enumerate(zip(labels, colors, strict=True)):
        vals = residual_enu[:, ci] / 1000.0
        ax.scatter(snr_db, vals, s=18, color=color, alpha=0.42, label=label)
        bins = snr_binned_scatter(snr_db, vals * 1000.0)
        if bins:
            ax.plot([b["snr_db"] for b in bins], [b["std"] / 1000.0 for b in bins], color=color, lw=1.8)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_ylabel("Component residual / binned std (km)")
    ax.set_title("ENU residual components")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1, 1]
    ax.scatter(snr_db, ls_sigma_deg, s=28, color="C4", alpha=0.72, edgecolor="0.2", linewidth=0.25, label="LS angular sigma")
    ax.scatter(snr_db, grid_to_fit_deg, s=20, color="0.35", alpha=0.55, label="beam grid to LS fit offset")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Beamforming seed and LS refinement")
    ax.legend(loc="best", fontsize=8)

    n_lsq = int(np.sum(ls_success))
    for ax in axes.ravel():
        ax.set_xlabel("Matched-filter SNR (dB)")
        ax.grid(True, color="0.9", lw=0.6)
    axes[0, 0].text(
        0.02,
        0.96,
        f"LS AoD refinement succeeded for {n_lsq}/{len(trail_rows)} trail positions",
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.86},
    )
    cbar = fig.colorbar(sc, ax=axes[0, 0], shrink=0.82, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["beam only", "LS refined"])
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def unit_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1) / np.maximum(an * bn, 1e-30)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def plot_trail_aod_angle_error_debug(
    out: Path,
    payload: dict,
    trail_rows: list[dict],
    t_ref: float,
    tx0: np.ndarray,
    rot: np.ndarray,
):
    model = payload["model"]
    t = np.array(model["time_unix"], dtype=float)
    pos = np.array(model["position_gcrs_m"], dtype=float)
    trail_points = np.array([r["gcrs_ref_m"] for r in trail_rows], dtype=float)
    _residual_gcrs, idx = closest_curve_vectors(trail_points, pos)

    model_enu = np.array(
        [
            (gcrs_to_ecef(pos[int(ii)], t[int(ii)]) - tx0) @ rot
            for ii in idx
        ],
        dtype=float,
    )
    model_unit = model_enu / np.linalg.norm(model_enu, axis=1)[:, None]
    measured_unit = np.array([r["aod_unit_enu"] for r in trail_rows], dtype=float)
    residual_angle_deg = unit_angle_deg(measured_unit, model_unit)
    linearized_sigma_deg = np.array([r.get("aod_lstsq_sigma_angular_deg", np.nan) for r in trail_rows], dtype=float)
    sigma_major_deg = np.array([r.get("aod_lstsq_sigma_major_deg", np.nan) for r in trail_rows], dtype=float)
    sigma_minor_deg = np.array([r.get("aod_lstsq_sigma_minor_deg", np.nan) for r in trail_rows], dtype=float)
    grid_to_fit_deg = np.array([r.get("aod_lstsq_grid_to_fit_offset_deg", np.nan) for r in trail_rows], dtype=float)
    snr_db = np.array([r.get("snr_db", np.nan) for r in trail_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    sc = ax.scatter(linearized_sigma_deg, residual_angle_deg, c=snr_db, cmap="viridis", s=30, alpha=0.78, edgecolor="0.2", linewidth=0.25)
    lim = max(1.0, float(np.nanpercentile(np.r_[linearized_sigma_deg, residual_angle_deg], 98.0)))
    ax.plot([0.0, lim], [0.0, lim], color="0.3", lw=1.1, ls="--", label="1:1")
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel("Linearized AoD $1\\sigma$ (deg)")
    ax.set_ylabel("Fit-residual AoD error (deg)")
    ax.set_title("Formal versus residual AoD angle error")
    ax.legend(loc="best", fontsize=8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("SNR (dB)")

    ax = axes[0, 1]
    ax.scatter(snr_db, residual_angle_deg, s=24, color="0.35", alpha=0.58, label="residual angle")
    bins = snr_binned_scatter(snr_db, residual_angle_deg)
    if bins:
        ax.plot([b["snr_db"] for b in bins], [b["std"] for b in bins], "o-", color="C1", lw=2.0, ms=7, label="binned robust scatter")
    ax.set_xlabel("Matched-filter SNR (dB)")
    ax.set_ylabel("AoD residual angle (deg)")
    ax.set_title("Residual angle versus SNR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.scatter(snr_db, linearized_sigma_deg, s=24, color="C4", alpha=0.68, label="combined tangent-plane $1\\sigma$")
    ax.scatter(snr_db, sigma_major_deg, s=18, color="C3", alpha=0.45, label="major-axis $1\\sigma$")
    ax.scatter(snr_db, sigma_minor_deg, s=18, color="C0", alpha=0.45, label="minor-axis $1\\sigma$")
    ax.set_xlabel("Matched-filter SNR (dB)")
    ax.set_ylabel("Linearized AoD uncertainty (deg)")
    ax.set_title("Formal AoD uncertainty versus SNR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ratio = residual_angle_deg / np.maximum(linearized_sigma_deg, 1e-12)
    ax.scatter(snr_db, ratio, s=24, color="C5", alpha=0.62, label="residual / formal")
    med_ratio = float(np.nanmedian(ratio))
    ax.axhline(1.0, color="0.35", lw=1.0, ls="--", label="1$\\sigma$")
    ax.axhline(med_ratio, color="C1", lw=1.5, label=f"median={med_ratio:.2f}")
    ax.scatter(snr_db, grid_to_fit_deg / np.maximum(linearized_sigma_deg, 1e-12), s=18, color="0.35", alpha=0.45, label="grid-to-LS / formal")
    ax.set_xlabel("Matched-filter SNR (dB)")
    ax.set_ylabel("Dimensionless ratio")
    ax.set_title("Residual-to-formal uncertainty ratio")
    ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, color="0.9", lw=0.6)
    axes[0, 0].text(
        0.02,
        0.96,
        f"Median residual angle {np.nanmedian(residual_angle_deg):.2f} deg; "
        f"median formal {np.nanmedian(linearized_sigma_deg):.2f} deg",
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.86},
    )
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_impact_map(
    out: Path,
    payload: dict,
    params: np.ndarray,
    cov: np.ndarray,
    obs: list[dict],
    trail_rows: list[dict],
    t_ref: float,
    density_profile: dict,
    args,
) -> dict:
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

    print("  nominal fragment trajectory", flush=True)
    nominal = fragment_impact_path(params, t_ref, density_profile, args, sample_dt_s=args.impact_nominal_dt_s)
    if nominal is None:
        raise RuntimeError("Could not propagate nominal fragment path to impact")

    print(f"  sampled fragment trajectories: {args.impact_uncertainty_samples}", flush=True)
    samples = sample_ballistic_params(params, cov, n=args.impact_uncertainty_samples)
    diameters = impact_sample_diameters(args, len(samples))
    sample_paths = []
    impact_points = []
    impact_samples = []
    for p, diameter_m in zip(samples, diameters, strict=True):
        try:
            path = fragment_impact_path(
                p,
                t_ref,
                density_profile,
                args,
                sample_dt_s=args.impact_sample_dt_s,
                fragment_diameter_m=float(diameter_m),
            )
            if path is None:
                continue
            sample_paths.append(path)
            impact_points.append(path["impact_llh"])
            impact_samples.append(path)
        except Exception:
            continue
    impact_points = np.asarray(impact_points, dtype=float) if impact_points else np.empty((0, 3), dtype=float)
    summary = impact_summary(nominal, impact_samples, t_ref, args)
    write_fragment_impact_table(Path(args.report_table_dir) / "fragment_impact_table.tex", summary)

    model = payload["model"]
    model_llh = np.column_stack([
        np.array(model["lat_deg"], dtype=float),
        np.array(model["lon_deg"], dtype=float),
        np.array(model["alt_m"], dtype=float),
    ])
    min_trail_alt_m = float(np.nanmin([r["alt_m"] for r in trail_rows]))
    fit_to_lowest = model_llh[:, 2] >= min_trail_alt_m
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    head_pos, _head_vel = propagate_state(params, t_head, t_ref, density_profile, use_drag=not args.no_drag)
    head_llh = geodetic_path_from_states(head_pos, t_head)

    data_crs = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(9.2, 7.4), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=data_crs) if data_crs else fig.add_subplot(1, 1, 1)

    trail_lon = np.array([r["lon_deg"] for r in trail_rows], dtype=float)
    trail_lat = np.array([r["lat_deg"] for r in trail_rows], dtype=float)
    trail_sigma_enu_m = np.array(
        [
            (r.get("uncertainty") or {}).get("position_sigma_enu_m", [np.nan, np.nan, np.nan])
            for r in trail_rows
        ],
        dtype=float,
    )
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(trail_lat))
    trail_lon_sigma_deg = trail_sigma_enu_m[:, 0] / meters_per_deg_lon
    trail_lat_sigma_deg = trail_sigma_enu_m[:, 1] / meters_per_deg_lat
    all_lon = np.r_[
        model_llh[:, 1],
        nominal["fragment_llh"][:, 1],
        nominal["impact_llh"][1],
        trail_lon,
        impact_points[:, 1] if impact_points.size else [],
    ]
    all_lat = np.r_[
        model_llh[:, 0],
        nominal["fragment_llh"][:, 0],
        nominal["impact_llh"][0],
        trail_lat,
        impact_points[:, 0] if impact_points.size else [],
    ]
    extent = [np.min(all_lon) - 0.18, np.max(all_lon) + 0.18, np.min(all_lat) - 0.12, np.max(all_lat) + 0.12]
    if data_crs:
        ax.set_extent(extent, crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none", zorder=0)
        ax.coastlines(resolution="10m", color="0.15", linewidth=0.9, zorder=3)
        ax.gridlines(draw_labels=True, color="0.7", linewidth=0.4)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    for path in sample_paths:
        ax.plot(path["pre_break_llh"][:, 1], path["pre_break_llh"][:, 0], color="C2", alpha=0.045, lw=0.8, transform=data_crs)
        ax.plot(path["fragment_llh"][:, 1], path["fragment_llh"][:, 0], color="C1", alpha=0.055, lw=0.8, transform=data_crs)
    if len(impact_points) > 0:
        ax.scatter(
            impact_points[:, 1],
            impact_points[:, 0],
            s=26,
            color="C1",
            marker="x",
            linewidths=0.9,
            alpha=0.55,
            transform=data_crs,
            label="1-100 cm sampled impact points",
            zorder=5,
        )
    ax.errorbar(
        trail_lon,
        trail_lat,
        xerr=trail_lon_sigma_deg,
        yerr=trail_lat_sigma_deg,
        fmt="o",
        ms=3.4,
        color="0.25",
        ecolor="0.25",
        elinewidth=0.45,
        capsize=0.0,
        alpha=0.45,
        transform=data_crs,
        label="trail positions",
        zorder=6,
    )
    ax.scatter(head_llh[:, 1], head_llh[:, 0], s=50, color="C0", marker="x", linewidths=1.4, alpha=0.9, transform=data_crs, label="head-echo positions")
    ax.plot(
        model_llh[fit_to_lowest, 1],
        model_llh[fit_to_lowest, 0],
        color="C2",
        lw=2.5,
        transform=data_crs,
        label="ballistic fit to lowest trail echo",
    )
    ax.plot(nominal["fragment_llh"][:, 1], nominal["fragment_llh"][:, 0], color="C1", lw=2.5, transform=data_crs, label="10 cm fragment path")
    ax.scatter(
        nominal["impact_llh"][1],
        nominal["impact_llh"][0],
        s=115,
        color="C3",
        marker="*",
        edgecolors="k",
        linewidths=0.6,
        transform=data_crs,
        label="nominal water impact",
        zorder=8,
    )
    ax.set_title("Ballistic trajectory and fragment impact estimate")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {
        "nominal": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in nominal.items()},
        "impact_sample_llh": impact_points.tolist(),
        "impact_samples": [
            {
                "t_break_unix": float(path["t_break_unix"]),
                "t_impact_unix": float(path["t_impact_unix"]),
                "fragment_diameter_m": float(path["fragment_diameter_m"]),
                "impact_llh": np.asarray(path["impact_llh"], dtype=float).tolist(),
                "impact_speed_mps": float(path["impact_speed_mps"]),
            }
            for path in impact_samples
        ],
        "summary": summary,
    }


def ballistic_llh_from_altitude(
    params: np.ndarray,
    t_ref: float,
    density_profile: dict,
    args,
    start_alt_km: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    search_t = np.arange(t_ref - 40.0, t_ref + args.fragment_breakup_search_s + 0.1, 0.1)
    search_pos, _search_vel = propagate_state(params, search_t, t_ref, density_profile, use_drag=not args.no_drag, integration_dt_s=0.05)
    search_llh = geodetic_path_from_states(search_pos, search_t)
    t_start = interpolate_crossing_time(search_t, search_llh[:, 2], start_alt_km * 1000.0)
    t_break = interpolate_crossing_time(search_t, search_llh[:, 2], args.fragment_breakup_alt_km * 1000.0)
    if t_start is None:
        above = np.where(search_llh[:, 2] >= start_alt_km * 1000.0)[0]
        t_start = float(search_t[above[-1]]) if above.size else float(search_t[0])
    if t_break is None:
        t_break = float(search_t[-1])
    t_path = np.linspace(float(t_start), float(t_break), 500)
    pos_path, _vel_path = propagate_state(params, t_path, t_ref, density_profile, use_drag=not args.no_drag, integration_dt_s=0.05)
    return t_path, geodetic_path_from_states(pos_path, t_path)


def plot_station_measurement_ballistic_overview(
    out: Path,
    params: np.ndarray,
    obs: list[dict],
    trail_rows: list[dict],
    geom: dict,
    t_ref: float,
    density_profile: dict,
    impact_payload: dict,
    args,
) -> None:
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
    head_pos, _head_vel = propagate_state(params, t_head, t_ref, density_profile, use_drag=not args.no_drag)
    head_llh = geodetic_path_from_states(head_pos, t_head)
    trail_llh = np.column_stack(
        [
            np.array([r["lat_deg"] for r in trail_rows], dtype=float),
            np.array([r["lon_deg"] for r in trail_rows], dtype=float),
            np.array([r["alt_m"] for r in trail_rows], dtype=float),
        ]
    )
    _t_ballistic, ballistic_llh = ballistic_llh_from_altitude(params, t_ref, density_profile, args, start_alt_km=300.0)
    nominal = impact_payload["nominal"]
    fragment_llh = np.asarray(nominal["fragment_llh"], dtype=float)
    impact_llh = np.asarray(nominal["impact_llh"], dtype=float)

    station_configs = load_all_station_configs(args.data_dir)
    station_lons = []
    station_lats = []
    tx_ref_lat = None
    tx_ref_lon = None
    for cfg in station_configs.values():
        rx_lat, rx_lon, _rx_alt = cfg.gps_rx
        tx_lat, tx_lon, _tx_alt = cfg.gps_tx
        station_lons.extend([rx_lon, tx_lon])
        station_lats.extend([rx_lat, tx_lat])
        if tx_ref_lat is None:
            tx_ref_lat = float(tx_lat)
            tx_ref_lon = float(tx_lon)

    measurement_lon = np.r_[trail_llh[:, 1], head_llh[:, 1]]
    measurement_lat = np.r_[trail_llh[:, 0], head_llh[:, 0]]
    all_lon = np.r_[station_lons, ballistic_llh[:, 1], fragment_llh[:, 1], impact_llh[1], measurement_lon]
    all_lat = np.r_[station_lats, ballistic_llh[:, 0], fragment_llh[:, 0], impact_llh[0], measurement_lat]
    extent = [
        float(np.nanmin(all_lon) - 0.28),
        float(np.nanmax(all_lon) + 0.38),
        float(np.nanmin(all_lat) - 0.35),
        float(np.nanmax(all_lat) + 0.25),
    ]

    data_crs = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(12.4, 8.3), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=data_crs) if data_crs else fig.add_subplot(1, 1, 1)
    if data_crs:
        ax.set_extent(extent, crs=data_crs)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none", zorder=0)
        ax.coastlines(resolution="10m", color="0.15", linewidth=0.85, zorder=3)
        gl = ax.gridlines(draw_labels=True, color="0.72", linewidth=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, color="0.72", linewidth=0.4)
        ax.tick_params(labelsize=18)

    if tx_ref_lat is not None and tx_ref_lon is not None:
        contour_lon = np.linspace(extent[0], extent[1], 260)
        contour_lat = np.linspace(extent[2], extent[3], 220)
        lon_grid, lat_grid = np.meshgrid(contour_lon, contour_lat)
        tx_lat_rad = np.deg2rad(tx_ref_lat)
        tx_lon_rad = np.deg2rad(tx_ref_lon)
        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_grid)
        dlat = lat_rad - tx_lat_rad
        dlon = lon_rad - tx_lon_rad
        a = np.sin(0.5 * dlat) ** 2 + np.cos(tx_lat_rad) * np.cos(lat_rad) * np.sin(0.5 * dlon) ** 2
        dist_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        range_contours = ax.contour(
            lon_grid,
            lat_grid,
            dist_km,
            levels=[50, 100, 150, 200, 250, 300, 350],
            colors="0.25",
            linewidths=0.55,
            alpha=0.30,
            transform=data_crs,
            zorder=1.5,
        )
        contour_labels = ax.clabel(range_contours, fmt=lambda value: f"{int(value)} km", fontsize=12, inline=True)
        for text in contour_labels:
            text.set_alpha(0.45)
            text.set_zorder(1.6)

    h, lon_edges, lat_edges = np.histogram2d(measurement_lon, measurement_lat, bins=(220, 170), range=[extent[:2], extent[2:]])
    h = h.T
    h[h <= 0] = np.nan
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        h,
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(2.0, float(np.nanmax(h)))),
        shading="auto",
        alpha=0.82,
        transform=data_crs,
        zorder=20,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.76, pad=0.02)
    cbar.set_label("head/trail measurement count per bin")
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.yaxis.label.set_size(20)

    ax.plot(
        ballistic_llh[:, 1],
        ballistic_llh[:, 0],
        color="C2",
        lw=3.2,
        transform=data_crs,
        zorder=6,
        label="ballistic fit, 300 km to 30 km",
    )
    alt_label_offsets = {
        300: (-0.30, -0.27),
        200: (-0.21, -0.225),
        90: (-0.09, -0.195),
        71: (0.165, -0.225),
        30: (0.285, -0.30),
    }
    alt_km = ballistic_llh[:, 2] / 1000.0
    order = np.argsort(alt_km)
    for label_km, (dlon, dlat) in alt_label_offsets.items():
        if label_km < np.nanmin(alt_km) or label_km > np.nanmax(alt_km):
            continue
        lat_at_alt = float(np.interp(label_km, alt_km[order], ballistic_llh[order, 0]))
        lon_at_alt = float(np.interp(label_km, alt_km[order], ballistic_llh[order, 1]))
        label_lon = lon_at_alt + dlon
        label_lat = lat_at_alt + dlat
        ax.plot(
            [lon_at_alt, label_lon],
            [lat_at_alt, label_lat],
            color="0.08",
            lw=1.15,
            transform=data_crs,
            zorder=8,
        )
        ax.text(
            label_lon,
            label_lat,
            f"{label_km} km",
            fontsize=17,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
            transform=data_crs,
            zorder=10,
        )
    ax.plot(
        fragment_llh[:, 1],
        fragment_llh[:, 0],
        color="C1",
        lw=3.2,
        transform=data_crs,
        zorder=6,
        label="nominal 10 cm ground fall",
    )
    ax.scatter(
        impact_llh[1],
        impact_llh[0],
        s=210,
        color="C3",
        marker="*",
        edgecolors="k",
        linewidths=0.7,
        transform=data_crs,
        zorder=8,
        label="nominal sea-level impact",
    )

    receiver_labeled = False
    tx_seen = set()
    for station, cfg in station_configs.items():
        rx_lat, rx_lon, _rx_alt = cfg.gps_rx
        tx_lat, tx_lon, _tx_alt = cfg.gps_tx
        ax.scatter(
            rx_lon,
            rx_lat,
            marker="^",
            s=100,
            color="0.1",
            transform=data_crs,
            zorder=8,
            label=None if receiver_labeled else "receivers",
        )
        receiver_labeled = True
        ax.text(rx_lon + 0.035, rx_lat + 0.02, short_station_name(station), fontsize=17, transform=data_crs, zorder=9, clip_on=True)
        tx_key = (round(float(tx_lat), 5), round(float(tx_lon), 5))
        if tx_key not in tx_seen:
            ax.scatter(tx_lon, tx_lat, marker="*", s=215, color="gold", edgecolor="0.2", transform=data_crs, zorder=9, label="Haystack transmitter")
            ax.text(tx_lon + 0.035, tx_lat + 0.02, "TX", fontsize=17, weight="bold", transform=data_crs, zorder=9, clip_on=True)
            tx_seen.add(tx_key)

    ax.set_title("Radar station geometry and bolide trajectory overview", fontsize=23)
    ax.legend(loc="lower left", fontsize=17, framealpha=0.92)
    fig.savefig(out, dpi=420, bbox_inches="tight")
    plt.close(fig)


def plot_lonlat_height(
    out: Path,
    payload: dict,
    params: np.ndarray,
    cov: np.ndarray,
    obs: list[dict],
    trail_rows: list[dict],
    t_ref: float,
    t_model: np.ndarray,
    density_profile: dict,
    args,
) -> None:
    model = payload["model"]
    lon = np.array(model["lon_deg"], dtype=float)
    lat = np.array(model["lat_deg"], dtype=float)
    alt = np.array(model["alt_m"], dtype=float) / 1000.0
    trail_lon = np.array([r["lon_deg"] for r in trail_rows], dtype=float)
    trail_lat = np.array([r["lat_deg"] for r in trail_rows], dtype=float)
    trail_alt = np.array([r["alt_m"] for r in trail_rows], dtype=float) / 1000.0
    trail_sigma_enu_m = np.array(
        [
            (r.get("uncertainty") or {}).get("position_sigma_enu_m", [np.nan, np.nan, np.nan])
            for r in trail_rows
        ],
        dtype=float,
    )
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(trail_lat))
    trail_lon_sigma_deg = trail_sigma_enu_m[:, 0] / meters_per_deg_lon
    trail_lat_sigma_deg = trail_sigma_enu_m[:, 1] / meters_per_deg_lat
    trail_alt_sigma_km = trail_sigma_enu_m[:, 2] / 1000.0
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    head_pos, _head_vel = propagate_state(params, t_head, t_ref, density_profile, use_drag=not args.no_drag)
    head_llh = np.array([gcrs_to_geodetic(p, tt) for p, tt in zip(head_pos, t_head, strict=True)])

    samples = sample_ballistic_params(params, cov, n=140)
    sample_llh = []
    for p in samples:
        try:
            pos_s, _vel_s = propagate_state(p, t_model, t_ref, density_profile, use_drag=not args.no_drag)
            llh_s = np.array([gcrs_to_geodetic(x, tt) for x, tt in zip(pos_s, t_model, strict=True)])
            if np.all(np.isfinite(llh_s)):
                sample_llh.append(llh_s)
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True, constrained_layout=True)
    panels = [
        (axes[0], lon, trail_lon, trail_lon_sigma_deg, head_llh[:, 1], "Longitude (deg)"),
        (axes[1], lat, trail_lat, trail_lat_sigma_deg, head_llh[:, 0], "Latitude (deg)"),
    ]
    for pi, (ax, x, trail_x, trail_x_sigma, head_x, xlabel) in enumerate(panels):
        for llh_s in sample_llh:
            sx = llh_s[:, 1] if pi == 0 else llh_s[:, 0]
            ax.plot(sx, llh_s[:, 2] / 1000.0, color="C2", alpha=0.045, lw=0.8)
        ax.plot(x, alt, color="C2", lw=2.4, label="ballistic fit")
        ax.errorbar(
            trail_x,
            trail_alt,
            xerr=trail_x_sigma,
            yerr=trail_alt_sigma_km,
            fmt="o",
            ms=4.2,
            color="0.25",
            ecolor="0.05",
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.8,
            alpha=0.5,
            label="trail positions with 1$\\sigma$ errors",
            zorder=6,
        )
        ax.scatter(
            head_x,
            head_llh[:, 2] / 1000.0,
            s=58,
            color="C0",
            marker="x",
            linewidths=1.6,
            alpha=0.95,
            label="head-echo positions",
            zorder=30,
        )
        ax.set_xlabel(xlabel)
        ax.grid(True, color="0.9", lw=0.6)
    axes[0].set_ylabel("Height (km)")
    axes[0].set_title("Ballistic trajectory: longitude-height")
    axes[1].set_title("Ballistic trajectory: latitude-height")
    axes[0].legend(loc="best")
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--head-fit-json", type=Path, default=DEFAULT_HEAD_FIT)
    p.add_argument("--combined-fit-json", type=Path, default=DEFAULT_COMBINED_FIT)
    p.add_argument("--trail-json", type=Path, default=DEFAULT_TRAIL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-fig-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--report-table-dir", type=Path, default=DEFAULT_REPORT_TABLE_DIR)
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
    p.add_argument("--model-padding-s", type=float, default=0.8)
    p.add_argument("--model-dt-s", type=float, default=0.10)
    p.add_argument("--initial-cd-area-over-mass", type=float, default=0.002)
    p.add_argument("--fragment-breakup-alt-km", type=float, default=30.0)
    p.add_argument("--fragment-diameter-m", type=float, default=DEFAULT_FRAGMENT_DIAMETER_M)
    p.add_argument("--fragment-diameter-min-m", type=float, default=0.01)
    p.add_argument("--fragment-diameter-max-m", type=float, default=1.0)
    p.add_argument("--fragment-density-kg-m3", type=float, default=DEFAULT_FRAGMENT_DENSITY_KG_M3)
    p.add_argument("--fragment-cd", type=float, default=DEFAULT_FRAGMENT_CD)
    p.add_argument("--fragment-breakup-search-s", type=float, default=30.0)
    p.add_argument("--fragment-impact-max-duration-s", type=float, default=3600.0)
    p.add_argument("--fragment-integration-dt-s", type=float, default=10.0)
    p.add_argument("--fragment-rtol", type=float, default=1e-5)
    p.add_argument("--fragment-atol", type=float, default=1e-1)
    p.add_argument("--impact-uncertainty-samples", type=int, default=100)
    p.add_argument("--impact-nominal-dt-s", type=float, default=2.0)
    p.add_argument("--impact-sample-dt-s", type=float, default=45.0)
    p.add_argument("--altitude-uncertainty-samples", type=int, default=120)
    p.add_argument("--no-drag", action="store_true")
    p.add_argument("--max-nfev", type=int, default=45)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_fig_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_table_dir).mkdir(parents=True, exist_ok=True)
    print("Loading head and trail constraints", flush=True)
    annotations, obs, used_stations, rejected_head, geom = load_head_data(args)
    obs = cache_head_station_states(obs, geom)
    t_ref = pre_head_reference_time(obs)
    all_trail = load_all_trail_positions(args.trail_json)
    trail_rows = load_trail_positions(args.trail_json, args.trail_min_snr_db, args.max_aod_offset_deg, args.min_trail_alt_km, args.max_trail_alt_km)
    before_exclude = len(trail_rows)
    trail_rows = [row for row in trail_rows if not is_excluded_trail_row(row)]
    if len(trail_rows) != before_exclude:
        print(f"Excluded {before_exclude - len(trail_rows)} manually flagged trail position(s)", flush=True)
    for row in all_trail + trail_rows:
        row["gcrs_ref_m"] = ecef_to_gcrs(row["position_ecef_m"], row["unix_time"])
    x_line = params_from_head_fit(args.head_fit_json, t_ref)
    try:
        combined = json.loads(args.combined_fit_json.read_text())
        x_line = shift_eci_state_linear(
            np.asarray(combined["fit_parameters_gcrs"], dtype=float),
            float(combined["t_ref_unix"]),
            t_ref,
        )
    except Exception:
        pass
    q0 = np.r_[np.full(3, 1.0), np.full(3, 0.01), math.log10(args.initial_cd_area_over_mass)]
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    t_line = []
    v2 = float(np.dot(x_line[3:6], x_line[3:6]))
    for r in trail_rows:
        tau = float(np.dot(np.asarray(r["gcrs_ref_m"]) - x_line[:3], x_line[3:6]) / v2)
        t_line.append(t_ref + tau)
    t_min = min(np.min(t_head), np.min(t_line)) - args.model_padding_s
    t_max = max(np.max(t_head), np.max(t_line)) + args.model_padding_s
    t_model = np.arange(t_min, t_max + 0.5 * args.model_dt_s, args.model_dt_s)
    density_profile = build_density_profile(np.r_[t_head, t_model], trail_rows, x_line, t_ref)
    print("Fitting ballistic trajectory", flush=True)
    lower = np.r_[np.full(3, -20000.0), np.full(3, -3000.0), -6.0]
    upper = np.r_[np.full(3, 20000.0), np.full(3, 3000.0), -0.2]
    res = least_squares(
        residuals_delta,
        q0,
        args=(x_line[:6], obs, trail_rows, t_ref, t_model, density_profile, args),
        bounds=(lower, upper),
        x_scale=np.r_[np.full(3, 3000.0), np.full(3, 500.0), 1.0],
        diff_step=np.r_[np.full(3, 100.0), np.full(3, 10.0), 0.05],
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=args.max_nfev,
        verbose=1,
    )
    xhat = parameters_from_delta(res.x, x_line[:6])
    print("Writing ballistic fit products", flush=True)
    payload = write_json(args.out_dir / "ballistic_head_trail_fit.json", res, xhat, obs, trail_rows, t_ref, t_model, density_profile, args)
    cov_delta = estimate_covariance(res, len(res.x))
    cov_params = cov_delta.copy()
    payload["fit_covariance"] = cov_params.tolist()
    (args.out_dir / "ballistic_head_trail_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    initial_state_json = args.out_dir / "ballistic_initial_state.json"
    initial_state_table = Path(args.report_table_dir) / "ballistic_initial_state_table.tex"
    write_initial_state_json(initial_state_json, xhat, cov_params, t_ref)
    write_initial_state_table_from_json(initial_state_json, initial_state_table)
    tx0, rot = transmitter_local_frame(geom)
    impact_plot = args.out_dir / "ballistic_head_trail_impact_map.png"
    print("Propagating fragment impact map", flush=True)
    impact_payload = plot_impact_map(impact_plot, payload, xhat, cov_params, obs, trail_rows, t_ref, density_profile, args)
    payload["fragment_impact"] = impact_payload
    (args.out_dir / "ballistic_head_trail_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copy2(impact_plot, Path(args.report_fig_dir) / impact_plot.name)
    overview_plot = args.out_dir / "station_measurement_ballistic_overview.png"
    plot_station_measurement_ballistic_overview(overview_plot, xhat, obs, trail_rows, geom, t_ref, density_profile, impact_payload, args)
    shutil.copy2(overview_plot, Path(args.report_fig_dir) / overview_plot.name)
    plot_path = args.out_dir / "ballistic_head_trail_altitude_enu.png"
    plot_altitude_enu(
        plot_path,
        payload,
        trail_rows,
        tx0,
        rot,
        t_ref,
        impact_payload=impact_payload,
        params=xhat,
        cov=cov_params,
        obs=obs,
        density_profile=density_profile,
        args=args,
    )
    shutil.copy2(plot_path, Path(args.report_fig_dir) / plot_path.name)
    trail_error_plot = args.out_dir / "trail_position_error_vs_snr.png"
    plot_trail_position_error_debug(trail_error_plot, payload, trail_rows, t_ref, tx0, rot)
    shutil.copy2(trail_error_plot, Path(args.report_fig_dir) / trail_error_plot.name)
    trail_angle_error_plot = args.out_dir / "trail_aod_angle_error_vs_snr.png"
    plot_trail_aod_angle_error_debug(trail_angle_error_plot, payload, trail_rows, t_ref, tx0, rot)
    shutil.copy2(trail_angle_error_plot, Path(args.report_fig_dir) / trail_angle_error_plot.name)
    lonlat_plot = args.out_dir / "ballistic_head_trail_lonlat_height.png"
    plot_lonlat_height(lonlat_plot, payload, xhat, cov_params, obs, trail_rows, t_ref, t_model, density_profile, args)
    shutil.copy2(lonlat_plot, Path(args.report_fig_dir) / lonlat_plot.name)
    print(f"Wrote {args.out_dir / 'ballistic_head_trail_fit.json'}")
    print(f"Cd*A/m {payload['cd_area_over_mass_m2_kg']:.4g} m^2/kg")
    print(f"Trail median residual {payload['trail_residual_m']['median']:.1f} m")


if __name__ == "__main__":
    main()
