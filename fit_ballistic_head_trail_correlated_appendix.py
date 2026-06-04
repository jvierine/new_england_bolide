#!/usr/bin/env python3
"""Appendix ballistic fit using selected head echoes and overlap covariance."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from fit_ballistic_head_trail import (
    DEFAULT_COMBINED_FIT,
    DEFAULT_DATA_DIR,
    DEFAULT_FRAGMENT_DIAMETER_M,
    DEFAULT_FRAGMENT_CD,
    DEFAULT_FRAGMENT_DENSITY_KG_M3,
    DEFAULT_TRAIL,
    build_density_profile,
    cache_head_station_states,
    closest_curve_vectors,
    ecef_to_gcrs,
    fragment_impact_path,
    geodetic_path_from_states,
    gcrs_to_geodetic,
    head_predictions_from_states,
    impact_sample_diameters,
    impact_summary,
    is_excluded_trail_row,
    load_all_trail_positions,
    load_trail_positions,
    plot_altitude_enu,
    plot_impact_map,
    plot_lonlat_height,
    plot_station_measurement_ballistic_overview,
    plot_trail_aod_angle_error_debug,
    plot_trail_position_error_debug,
    propagate_state,
    shift_eci_state_linear,
    sphere_cd_area_over_mass,
    transmitter_local_frame,
    write_initial_state_json,
    write_json,
)
from fit_combined_head_trail_eci import (
    geodetic_to_gcrs_posvel,
    station_geometry,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_HEAD_SELECTION = (
    ROOT / "reports" / "head_echo_python_selection_workflow_ext0p5" / "head_echo_manual_selection_real.csv"
)
DEFAULT_OUT_DIR = ROOT / "products" / "ballistic_head_trail_correlated_appendix_fit"
DEFAULT_REPORT_DIR = ROOT / "reports" / "figures" / "appendix_correlated_fit"
DEFAULT_REPORT_TABLE_DIR = ROOT / "reports" / "appendix_correlated_fit"
OMEGA_EARTH = 7.2921150e-5
C = 299792458.0
MU_SUN = 1.32712440018e20
AU_M = 1.495978707e11
FIXED_BODY_DIAMETER_M = 1.6
FIXED_BODY_MASS_KG = 5000.0
FIXED_BODY_CD = 1.0
FIXED_BODY_DENSITY_KG_M3 = FIXED_BODY_MASS_KG / ((4.0 / 3.0) * math.pi * (0.5 * FIXED_BODY_DIAMETER_M) ** 3)


def fixed_cd_area_over_mass(args) -> float:
    radius = 0.5 * float(args.body_diameter_m)
    area = math.pi * radius**2
    return float(args.body_cd) * area / float(args.body_mass_kg)


def fixed_log10_cd_area_over_mass(args) -> float:
    return math.log10(fixed_cd_area_over_mass(args))


def load_selected_head_observations(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    df = df[df["selection"].astype(str).str.lower() == "real"].copy()
    df["station"] = df["station"].astype(str).str.lower()
    required = ["station", "time_unix", "fit_range_m", "fit_doppler_hz"]
    df = df.dropna(subset=required)
    rows = []
    for _idx, row in df.sort_values(["station", "time_unix", "idx"]).iterrows():
        rows.append(
            {
                "station": str(row["station"]),
                "unix_time": float(row["time_unix"]),
                "range_m": float(row["fit_range_m"]),
                "doppler_hz": float(row["fit_doppler_hz"]),
                "snr_db": float(row.get("amplitude_db", np.nan)),
                "range_sigma_m": float(max(row.get("sigma_range_m", np.nan), 500.0)),
                "doppler_sigma_hz": float(max(row.get("sigma_doppler_hz", np.nan), 25.0)),
                "source_idx": int(row.get("idx", -1)),
            }
        )
    if not rows:
        raise ValueError(f"No real selected head observations found in {path}")
    return rows


def cache_head_station_states_gcrs(obs: list[dict], geom: dict) -> list[dict]:
    cached = cache_head_station_states(obs, geom)
    for ob in cached:
        if "frequency_hz" not in ob:
            ob["frequency_hz"] = float(geom[ob["station"]]["frequency_hz"])
    return cached


def pre_head_reference_time(obs: list[dict], offset_s: float = 4.0) -> float:
    return float(min(o["unix_time"] for o in obs) - offset_s)


def parameters_from_delta(delta: np.ndarray, base_state: np.ndarray, args) -> np.ndarray:
    return np.r_[base_state[:3] + delta[:3], base_state[3:6] + delta[3:6], fixed_log10_cd_area_over_mass(args)]


def low_altitude_trail_timing_raw(trail_rows: list[dict], model_time: np.ndarray, args) -> np.ndarray:
    if not args.trail_timing_enable:
        return np.array([], dtype=float)
    alt_km = np.array([float(r["alt_m"]) / 1000.0 for r in trail_rows], dtype=float)
    echo_time = np.array([float(r["unix_time"]) for r in trail_rows], dtype=float)
    keep = np.isfinite(alt_km) & np.isfinite(echo_time) & (alt_km <= args.trail_timing_max_alt_km)
    if not np.any(keep):
        return np.array([], dtype=float)
    return np.asarray(model_time, dtype=float)[keep] + args.trail_timing_lag_s - echo_time[keep]


def raw_residuals(params, obs, trail_rows, t_ref, t_model, density_profile, args) -> np.ndarray:
    try:
        pos_head, vel_head = propagate_state(
            params,
            np.array([o["unix_time"] for o in obs], dtype=float),
            t_ref,
            density_profile,
            use_drag=not args.no_drag,
        )
        pos_model, _vel_model = propagate_state(params, t_model, t_ref, density_profile, use_drag=not args.no_drag)
    except Exception:
        return np.full(2 * len(obs) + 3 * len(trail_rows), 1e12, dtype=float)
    pred_range, pred_doppler = head_predictions_from_states(pos_head, vel_head, obs)
    meas_range = np.array([o["range_m"] for o in obs], dtype=float)
    meas_doppler = np.array([o["doppler_hz"] for o in obs], dtype=float)
    trail_points = np.array([r["gcrs_ref_m"] for r in trail_rows], dtype=float)
    trail_vec, idx = closest_curve_vectors(trail_points, pos_model)
    timing = low_altitude_trail_timing_raw(trail_rows, t_model[idx], args)
    return np.r_[pred_range - meas_range, pred_doppler - meas_doppler, trail_vec.ravel(), timing]


def overlap_corr(dt_s: np.ndarray, code_period_s: float, n_codes: int) -> np.ndarray:
    lag_codes = np.rint(np.abs(dt_s) / code_period_s).astype(int)
    return np.maximum(0.0, 1.0 - lag_codes / float(n_codes))


def fill_correlated_block(
    cov: np.ndarray,
    indices: list[int],
    times: np.ndarray,
    sigma: np.ndarray,
    *,
    n_codes: int,
    code_period_s: float,
) -> None:
    if len(indices) == 0:
        return
    times = np.asarray(times, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = overlap_corr(times[:, None] - times[None, :], code_period_s, n_codes)
    block = corr * sigma[:, None] * sigma[None, :]
    cov[np.ix_(indices, indices)] = block


def build_measurement_covariance(obs: list[dict], trail_rows: list[dict], args) -> tuple[np.ndarray, dict]:
    n_head = len(obs)
    n_trail = len(trail_rows)
    n_timing = 0
    if args.trail_timing_enable:
        n_timing = int(np.sum(np.array([float(r["alt_m"]) / 1000.0 for r in trail_rows]) <= args.trail_timing_max_alt_km))
    n = 2 * n_head + 3 * n_trail + n_timing
    cov = np.zeros((n, n), dtype=float)

    for station in sorted(set(o["station"] for o in obs)):
        idx_obs = [i for i, o in enumerate(obs) if o["station"] == station]
        times = np.array([obs[i]["unix_time"] for i in idx_obs], dtype=float)
        range_sigma = np.array([obs[i]["range_sigma_m"] for i in idx_obs], dtype=float)
        dop_sigma = np.array([obs[i]["doppler_sigma_hz"] for i in idx_obs], dtype=float)
        fill_correlated_block(
            cov,
            idx_obs,
            times,
            range_sigma,
            n_codes=args.head_coherent_codes,
            code_period_s=args.code_period_s,
        )
        fill_correlated_block(
            cov,
            [n_head + i for i in idx_obs],
            times,
            dop_sigma,
            n_codes=args.head_coherent_codes,
            code_period_s=args.code_period_s,
        )

    trail_base = 2 * n_head
    for comp in range(3):
        for station in sorted(set(r["station"] for r in trail_rows)):
            idx_rows = [i for i, r in enumerate(trail_rows) if r["station"] == station]
            times = np.array([trail_rows[i]["unix_time"] for i in idx_rows], dtype=float)
            sigma = np.array([trail_rows[i]["sigma_m"] for i in idx_rows], dtype=float)
            indices = [trail_base + 3 * i + comp for i in idx_rows]
            fill_correlated_block(
                cov,
                indices,
                times,
                sigma,
                n_codes=args.trail_coherent_codes,
                code_period_s=args.code_period_s,
            )

    timing_base = trail_base + 3 * n_trail
    if n_timing:
        timing_sigma = np.full(n_timing, float(args.trail_timing_sigma_s), dtype=float)
        timing_times = np.array(
            [float(r["unix_time"]) for r in trail_rows if float(r["alt_m"]) / 1000.0 <= args.trail_timing_max_alt_km],
            dtype=float,
        )
        fill_correlated_block(
            cov,
            list(range(timing_base, timing_base + n_timing)),
            timing_times,
            timing_sigma,
            n_codes=args.trail_coherent_codes,
            code_period_s=args.code_period_s,
        )

    diag_floor = np.maximum(np.abs(np.diag(cov)) * 1e-8, 1e-6)
    cov += np.diag(diag_floor)
    meta = {
        "n_residuals": int(n),
        "n_head_observations": int(n_head),
        "n_trail_positions": int(n_trail),
        "n_timing_constraints": int(n_timing),
        "head_coherent_codes": int(args.head_coherent_codes),
        "trail_coherent_codes": int(args.trail_coherent_codes),
        "code_period_s": float(args.code_period_s),
        "description": "Block covariance with triangular coherent-integration overlap correlation within each station and observable.",
    }
    return cov, meta


def whitening_from_covariance(cov: np.ndarray) -> np.ndarray:
    jitter = 0.0
    for _ in range(8):
        try:
            return cholesky(cov + np.eye(cov.shape[0]) * jitter, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            jitter = max(1e-6, 10.0 * jitter if jitter else 1e-6)
    return cholesky(cov + np.eye(cov.shape[0]) * jitter, lower=True, check_finite=False)


def whiten(residual: np.ndarray, chol: np.ndarray) -> np.ndarray:
    return solve_triangular(chol, residual, lower=True, check_finite=False)


def whitened_delta_residuals(delta, base_state, obs, trail_rows, t_ref, t_model, density_profile, args, chol):
    params = parameters_from_delta(delta, base_state, args)
    return whiten(raw_residuals(params, obs, trail_rows, t_ref, t_model, density_profile, args), chol)


def covariance_from_whitened_jacobian(result, n_params: int) -> tuple[np.ndarray, float]:
    jac = np.asarray(result.jac, dtype=float)
    normal = jac.T @ jac
    cov = np.linalg.pinv(normal)
    cov = 0.5 * (cov + cov.T)
    dof = max(1, jac.shape[0] - n_params)
    chi2_red = float(np.sum(np.asarray(result.fun) ** 2) / dof)
    return cov, chi2_red


def embed_state_covariance(cov6: np.ndarray) -> np.ndarray:
    cov7 = np.zeros((7, 7), dtype=float)
    cov7[:6, :6] = np.asarray(cov6, dtype=float)
    return cov7


def load_initial_state(args, t_ref: float) -> np.ndarray:
    old = json.loads(Path(args.initial_ballistic_json).read_text(encoding="utf-8"))
    state = np.asarray(old["fit_parameters"][:6], dtype=float)
    return shift_eci_state_linear(state, float(old["t_ref_unix"]), t_ref)


def write_appendix_summary(path: Path, payload: dict, old_payload: dict, cov_meta: dict) -> None:
    def fmt(x, nd=3):
        return f"{float(x):.{nd}f}"

    new_state = np.asarray(payload["fit_parameters"][:6], dtype=float)
    old_state = shift_eci_state_linear(
        np.asarray(old_payload["fit_parameters"][:6], dtype=float),
        float(old_payload["t_ref_unix"]),
        float(payload["t_ref_unix"]),
    )
    dp = new_state[:3] - old_state[:3]
    dv = new_state[3:6] - old_state[3:6]
    impact = payload.get("fragment_impact", {}).get("summary", {})
    old_impact = old_payload.get("fragment_impact", {}).get("summary", {})
    lines = [
        "% Auto-generated by fit_ballistic_head_trail_correlated_appendix.py\n",
        f"\\newcommand{{\\CorrFitNHead}}{{{payload['n_head_observations']}}}\n",
        f"\\newcommand{{\\CorrFitNTrail}}{{{payload['n_trail_positions']}}}\n",
        f"\\newcommand{{\\CorrFitNResiduals}}{{{cov_meta['n_residuals']}}}\n",
        f"\\newcommand{{\\CorrFitReducedChiSq}}{{{fmt(payload['correlated_covariance']['reduced_chi2'], 2)}}}\n",
        f"\\newcommand{{\\CorrFitDeltaPositionKm}}{{{fmt(np.linalg.norm(dp) / 1000.0, 2)}}}\n",
        f"\\newcommand{{\\CorrFitDeltaVelocityMps}}{{{fmt(np.linalg.norm(dv), 1)}}}\n",
        f"\\newcommand{{\\CorrFitCdAreaMass}}{{{payload['cd_area_over_mass_m2_kg']:.3e}}}\n",
        f"\\newcommand{{\\CorrFitBodyDiameterM}}{{{payload['fixed_body']['diameter_m']:.1f}}}\n",
        f"\\newcommand{{\\CorrFitBodyMassTon}}{{{payload['fixed_body']['mass_kg'] / 1000.0:.1f}}}\n",
        f"\\newcommand{{\\CorrFitTrailMedianKm}}{{{fmt(payload['trail_residual_m']['median'] / 1000.0, 2)}}}\n",
    ]
    if impact and old_impact:
        dlat = float(impact["nominal_lat_deg"]) - float(old_impact["nominal_lat_deg"])
        dlon = float(impact["nominal_lon_deg"]) - float(old_impact["nominal_lon_deg"])
        lines.extend(
            [
                f"\\newcommand{{\\CorrFitImpactLat}}{{{fmt(impact['nominal_lat_deg'], 4)}}}\n",
                f"\\newcommand{{\\CorrFitImpactLon}}{{{fmt(impact['nominal_lon_deg'], 4)}}}\n",
                f"\\newcommand{{\\CorrFitDeltaImpactLat}}{{{fmt(dlat, 4)}}}\n",
                f"\\newcommand{{\\CorrFitDeltaImpactLon}}{{{fmt(dlon, 4)}}}\n",
            ]
        )
    orbit = payload.get("nominal_orbit_comparison", {})
    if orbit:
        old_orb = orbit["old"]
        new_orb = orbit["new"]
        lines.extend(
            [
                f"\\newcommand{{\\CorrFitOldOrbitA}}{{{fmt(old_orb['a_AU'], 2)}}}\n",
                f"\\newcommand{{\\CorrFitNewOrbitA}}{{{fmt(new_orb['a_AU'], 2)}}}\n",
                f"\\newcommand{{\\CorrFitOldOrbitE}}{{{fmt(old_orb['e'], 3)}}}\n",
                f"\\newcommand{{\\CorrFitNewOrbitE}}{{{fmt(new_orb['e'], 3)}}}\n",
                f"\\newcommand{{\\CorrFitOldOrbitI}}{{{fmt(old_orb['i_deg'], 2)}}}\n",
                f"\\newcommand{{\\CorrFitNewOrbitI}}{{{fmt(new_orb['i_deg'], 2)}}}\n",
                f"\\newcommand{{\\CorrFitOldOrbitQ}}{{{fmt(old_orb['q_AU'], 3)}}}\n",
                f"\\newcommand{{\\CorrFitNewOrbitQ}}{{{fmt(new_orb['q_AU'], 3)}}}\n",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def sample_ballistic_params(params: np.ndarray, cov: np.ndarray, n: int, seed: int = 20260604) -> np.ndarray:
    if n <= 0:
        return np.empty((0, len(params)), dtype=float)
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, 0.0)
    clean = (vecs * vals) @ vecs.T
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(params, clean, size=n, check_valid="ignore")


def plot_correlated_range_doppler(out: Path, params, cov, obs, t_ref, density_profile, args) -> None:
    stations = sorted(set(o["station"] for o in obs))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    samples = sample_ballistic_params(params, cov, args.range_doppler_uncertainty_samples)
    for si, station in enumerate(stations):
        color = f"C{si}"
        so = [o for o in obs if o["station"] == station]
        so.sort(key=lambda o: o["unix_time"])
        times = np.array([o["unix_time"] for o in so], dtype=float)
        pos, vel = propagate_state(params, times, t_ref, density_profile, use_drag=not args.no_drag)
        pr, pd = head_predictions_from_states(pos, vel, so)
        meas_r = np.array([o["range_m"] for o in so], dtype=float) / 1000.0
        meas_d = np.array([o["doppler_hz"] for o in so], dtype=float) / 1000.0
        sig_r = np.array([o["range_sigma_m"] for o in so], dtype=float) / 1000.0
        sig_d = np.array([o["doppler_sigma_hz"] for o in so], dtype=float) / 1000.0
        tr = times - t_ref
        sample_r = []
        sample_d = []
        for p in samples:
            try:
                ps, vs = propagate_state(p, times, t_ref, density_profile, use_drag=not args.no_drag)
                rr, dd = head_predictions_from_states(ps, vs, so)
                if np.all(np.isfinite(rr)) and np.all(np.isfinite(dd)):
                    sample_r.append(rr / 1000.0)
                    sample_d.append(dd / 1000.0)
            except Exception:
                continue
        label = station.capitalize()
        axes[0].errorbar(tr, meas_r, yerr=sig_r, fmt="o", ms=4, color=color, ecolor=color, alpha=0.65, capsize=1.5, label=f"{label} selected")
        axes[1].errorbar(tr, meas_d, yerr=sig_d, fmt="o", ms=4, color=color, ecolor=color, alpha=0.65, capsize=1.5, label=f"{label} selected")
        axes[0].plot(tr, pr / 1000.0, color=color, lw=2.0, label=f"{label} ballistic fit")
        axes[1].plot(tr, pd / 1000.0, color=color, lw=2.0, label=f"{label} ballistic fit")
        if sample_r:
            rlo, rhi = np.percentile(np.asarray(sample_r), [2.5, 97.5], axis=0)
            dlo, dhi = np.percentile(np.asarray(sample_d), [2.5, 97.5], axis=0)
            axes[0].fill_between(tr, rlo, rhi, color=color, alpha=0.16, linewidth=0, label="95% correlated-fit covariance" if si == 0 else None)
            axes[1].fill_between(tr, dlo, dhi, color=color, alpha=0.16, linewidth=0, label="95% correlated-fit covariance" if si == 0 else None)
    axes[0].set_title("Appendix correlated ballistic fit: selected head-echo observables")
    axes[0].set_ylabel("Bistatic range (km)")
    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].set_xlabel(f"Seconds relative to {payload_time_label(t_ref)}")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.6)
        ax.legend(ncol=2, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def payload_time_label(t_ref: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(t_ref), UTC).isoformat().replace("+00:00", "Z")


def heliocentric_state_from_gcrs(t_unix: float, r_geo_m: np.ndarray, v_geo_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = Time(float(t_unix), format="unix", scale="utc")
    with solar_system_ephemeris.set("builtin"):
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", t)
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", t)
    r_helio = (earth_pos.xyz - sun_pos.xyz).to_value(u.m) + np.asarray(r_geo_m, dtype=float)
    v_helio = (earth_vel.xyz - sun_vel.xyz).to_value(u.m / u.s) + np.asarray(v_geo_mps, dtype=float)
    return r_helio, v_helio


def two_body_elements(r: np.ndarray, v: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    radius = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_norm = float(np.linalg.norm(n))
    e_vec = np.cross(v, h) / MU_SUN - r / radius
    ecc = float(np.linalg.norm(e_vec))
    energy = 0.5 * speed**2 - MU_SUN / radius
    semi_major = -MU_SUN / (2.0 * energy)
    inc = float(np.rad2deg(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))))
    raan = float(np.rad2deg(np.arctan2(n[1], n[0])) % 360.0) if n_norm > 0.0 else float("nan")
    argp = float("nan")
    if n_norm > 0.0 and ecc > 0.0:
        argp = float(np.rad2deg(np.arccos(np.clip(np.dot(n, e_vec) / (n_norm * ecc), -1.0, 1.0))))
        if e_vec[2] < 0.0:
            argp = 360.0 - argp
    true_anomaly = float("nan")
    if ecc > 0.0:
        true_anomaly = float(np.rad2deg(np.arccos(np.clip(np.dot(e_vec, r) / (ecc * radius), -1.0, 1.0))))
        if np.dot(r, v) < 0.0:
            true_anomaly = 360.0 - true_anomaly
    return {
        "a_AU": float(semi_major / AU_M),
        "e": ecc,
        "i_deg": inc,
        "raan_deg": raan,
        "arg_periapsis_deg": argp,
        "true_anomaly_deg": true_anomaly,
        "q_AU": float(semi_major * (1.0 - ecc) / AU_M),
    }


def nominal_orbit_elements(payload: dict) -> dict:
    t_ref = float(payload["t_ref_unix"])
    params = np.asarray(payload["fit_parameters"][:6], dtype=float)
    r, v = heliocentric_state_from_gcrs(t_ref, params[:3], params[3:6])
    return two_body_elements(r, v)


def write_orbit_comparison_table(path: Path, comparison: dict) -> None:
    old = comparison["old"]
    new = comparison["new"]
    rows = [
        ("Semimajor axis", "AU", "a_AU", 2),
        ("Perihelion distance", "AU", "q_AU", 3),
        ("Eccentricity", "--", "e", 3),
        ("Inclination", "deg", "i_deg", 2),
        ("Longitude of ascending node", "deg", "raan_deg", 2),
        ("Argument of perihelion", "deg", "arg_periapsis_deg", 2),
    ]
    lines = [
        "\\begin{table}\n",
        "  \\centering\n",
        "  \\caption{Nominal two-body heliocentric orbit comparison for the baseline ballistic fit and the appendix correlated-covariance fit.  This table uses Astropy's built-in Earth/Sun ephemeris and does not replace the full rebound Monte Carlo orbit solution used for the main article.}\n",
        "  \\label{tab:appendix-correlated-orbit-change}\n",
        "  \\begin{tabular}{llrrr}\n",
        "    \\toprule\n",
        "    Quantity & Unit & Baseline & Correlated fit & Difference \\\\\n",
        "    \\midrule\n",
    ]
    for label, unit, key, nd in rows:
        diff = float(new[key]) - float(old[key])
        lines.append(f"    {label} & {unit} & {old[key]:.{nd}f} & {new[key]:.{nd}f} & {diff:+.{nd}f} \\\\\n")
    lines.extend(["    \\bottomrule\n", "  \\end{tabular}\n", "\\end{table}\n"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_fig_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_table_dir).mkdir(parents=True, exist_ok=True)

    print("Loading selected head echoes", flush=True)
    obs = load_selected_head_observations(args.head_selection_csv)
    used_stations = sorted(set(o["station"] for o in obs))
    geom = station_geometry(args.data_dir, used_stations)
    obs = cache_head_station_states_gcrs(obs, geom)
    t_ref = pre_head_reference_time(obs)

    print("Loading trail positions", flush=True)
    all_trail = load_all_trail_positions(args.trail_json)
    trail_rows = load_trail_positions(args.trail_json, args.trail_min_snr_db, args.max_aod_offset_deg, args.min_trail_alt_km, args.max_trail_alt_km)
    trail_rows = [row for row in trail_rows if not is_excluded_trail_row(row)]
    for row in all_trail + trail_rows:
        row["gcrs_ref_m"] = ecef_to_gcrs(row["position_ecef_m"], row["unix_time"])

    base_state = load_initial_state(args, t_ref)
    old_payload = json.loads(Path(args.initial_ballistic_json).read_text(encoding="utf-8"))
    fixed_log_drag = fixed_log10_cd_area_over_mass(args)
    old_params_shifted = np.r_[base_state, fixed_log_drag]
    q0 = np.zeros(6, dtype=float)

    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    v2 = float(np.dot(base_state[3:6], base_state[3:6]))
    t_line = []
    for r in trail_rows:
        tau = float(np.dot(np.asarray(r["gcrs_ref_m"]) - base_state[:3], base_state[3:6]) / v2)
        t_line.append(t_ref + tau)
    t_min = min(np.min(t_head), np.min(t_line)) - args.model_padding_s
    t_max = max(np.max(t_head), np.max(t_line)) + args.model_padding_s
    t_model = np.arange(t_min, t_max + 0.5 * args.model_dt_s, args.model_dt_s)
    density_profile = build_density_profile(np.r_[t_head, t_model], trail_rows, old_params_shifted, t_ref)
    cov, cov_meta = build_measurement_covariance(obs, trail_rows, args)
    chol = whitening_from_covariance(cov)

    print(
        f"Fitting correlated ballistic model with {len(obs)} head observations, "
        f"{len(trail_rows)} trail positions, {cov_meta['n_residuals']} residuals",
        flush=True,
    )
    lower = np.r_[np.full(3, -30000.0), np.full(3, -4000.0)]
    upper = np.r_[np.full(3, 30000.0), np.full(3, 4000.0)]
    res = least_squares(
        whitened_delta_residuals,
        q0,
        args=(base_state, obs, trail_rows, t_ref, t_model, density_profile, args, chol),
        bounds=(lower, upper),
        x_scale=np.r_[np.full(3, 3000.0), np.full(3, 500.0)],
        diff_step=np.r_[np.full(3, 100.0), np.full(3, 10.0)],
        loss="linear",
        max_nfev=args.max_nfev,
        verbose=1,
    )
    xhat = parameters_from_delta(res.x, base_state, args)
    cov_delta, chi2_red = covariance_from_whitened_jacobian(res, len(res.x))
    cov_params = embed_state_covariance(cov_delta)

    print("Writing products", flush=True)
    payload = write_json(args.out_dir / "ballistic_head_trail_correlated_fit.json", res, xhat, obs, trail_rows, t_ref, t_model, density_profile, args)
    payload["parameter_order"] = [
        "x_gcrs_m",
        "y_gcrs_m",
        "z_gcrs_m",
        "vx_gcrs_mps",
        "vy_gcrs_mps",
        "vz_gcrs_mps",
        "fixed_log10_cd_area_over_mass_m2_kg",
    ]
    payload["fitted_parameter_order"] = payload["parameter_order"][:6]
    payload["fixed_parameter_order"] = payload["parameter_order"][6:]
    payload["fit_covariance"] = cov_params.tolist()
    payload["correlated_covariance"] = {
        **cov_meta,
        "reduced_chi2": chi2_red,
        "covariance_model": "Sigma blocks use triangular overlap autocorrelation; parameter covariance is the standard linearized GLS covariance (J^T Sigma^-1 J)^-1 for the six fitted inertial-state parameters.",
    }
    payload["fixed_body"] = {
        "diameter_m": float(args.body_diameter_m),
        "mass_kg": float(args.body_mass_kg),
        "cd": float(args.body_cd),
        "cd_area_over_mass_m2_kg": float(fixed_cd_area_over_mass(args)),
        "density_kg_m3": float(FIXED_BODY_DENSITY_KG_M3 if args.body_diameter_m == FIXED_BODY_DIAMETER_M and args.body_mass_kg == FIXED_BODY_MASS_KG else args.body_mass_kg / ((4.0 / 3.0) * math.pi * (0.5 * args.body_diameter_m) ** 3)),
        "note": "Atmospheric drag is fixed from this body model and is not estimated in the least-squares fit.",
    }
    payload["nominal_orbit_comparison"] = {
        "method": "Astropy built-in Earth/Sun ephemeris plus two-body heliocentric elements; diagnostic only.",
        "old": nominal_orbit_elements(old_payload),
        "new": nominal_orbit_elements(payload),
    }
    (args.out_dir / "ballistic_head_trail_correlated_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_initial_state_json(args.out_dir / "ballistic_initial_state_correlated.json", xhat, cov_params, t_ref)

    tx0, rot = transmitter_local_frame(geom)
    impact_plot = args.out_dir / "ballistic_head_trail_correlated_impact_map.png"
    impact_payload = plot_impact_map(impact_plot, payload, xhat, cov_params, obs, trail_rows, t_ref, density_profile, args)
    payload["fragment_impact"] = impact_payload
    (args.out_dir / "ballistic_head_trail_correlated_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overview_plot = args.out_dir / "station_measurement_ballistic_correlated_overview.png"
    range_doppler_plot = args.out_dir / "ballistic_head_trail_correlated_range_doppler_fit.png"
    altitude_plot = args.out_dir / "ballistic_head_trail_correlated_altitude_enu.png"
    lonlat_plot = args.out_dir / "ballistic_head_trail_correlated_lonlat_height.png"
    trail_error_plot = args.out_dir / "trail_position_correlated_error_vs_snr.png"
    trail_angle_plot = args.out_dir / "trail_aod_correlated_angle_error_vs_snr.png"
    plot_correlated_range_doppler(range_doppler_plot, xhat, cov_params, obs, t_ref, density_profile, args)
    plot_station_measurement_ballistic_overview(overview_plot, xhat, obs, trail_rows, geom, t_ref, density_profile, impact_payload, args)
    plot_altitude_enu(altitude_plot, payload, trail_rows, tx0, rot, t_ref, impact_payload=impact_payload, params=xhat, cov=cov_params, obs=obs, density_profile=density_profile, args=args)
    plot_lonlat_height(lonlat_plot, payload, xhat, cov_params, obs, trail_rows, t_ref, t_model, density_profile, args)
    plot_trail_position_error_debug(trail_error_plot, payload, trail_rows, t_ref, tx0, rot)
    plot_trail_aod_angle_error_debug(trail_angle_plot, payload, trail_rows, t_ref, tx0, rot)
    for plot in [impact_plot, overview_plot, range_doppler_plot, altitude_plot, lonlat_plot, trail_error_plot, trail_angle_plot]:
        shutil.copy2(plot, Path(args.report_fig_dir) / plot.name)

    write_appendix_summary(Path(args.report_table_dir) / "correlated_ballistic_fit_summary.tex", payload, old_payload, cov_meta)
    write_orbit_comparison_table(Path(args.report_table_dir) / "correlated_ballistic_orbit_comparison_table.tex", payload["nominal_orbit_comparison"])
    print(f"Wrote {args.out_dir / 'ballistic_head_trail_correlated_fit.json'}")
    print(f"reduced chi^2 {chi2_red:.3f}")
    print(f"fixed Cd*A/m {payload['cd_area_over_mass_m2_kg']:.4g} m^2/kg")
    print(f"Trail median residual {payload['trail_residual_m']['median']:.1f} m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-selection-csv", type=Path, default=DEFAULT_HEAD_SELECTION)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--trail-json", type=Path, default=DEFAULT_TRAIL)
    p.add_argument("--initial-ballistic-json", type=Path, default=ROOT / "products" / "ballistic_head_trail_fit" / "ballistic_head_trail_fit.json")
    p.add_argument("--combined-fit-json", type=Path, default=DEFAULT_COMBINED_FIT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-fig-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--report-table-dir", type=Path, default=DEFAULT_REPORT_TABLE_DIR)
    p.add_argument("--head-coherent-codes", type=int, default=16)
    p.add_argument("--trail-coherent-codes", type=int, default=8)
    p.add_argument("--code-period-s", type=float, default=0.01)
    p.add_argument("--trail-min-snr-db", type=float, default=8.0)
    p.add_argument("--max-aod-offset-deg", type=float, default=35.0)
    p.add_argument("--min-trail-alt-km", type=float, default=20.0)
    p.add_argument("--max-trail-alt-km", type=float, default=115.0)
    p.add_argument("--trail-timing-enable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trail-timing-max-alt-km", type=float, default=60.0)
    p.add_argument("--trail-timing-lag-s", type=float, default=1.0)
    p.add_argument("--trail-timing-sigma-s", type=float, default=1.0)
    p.add_argument("--model-padding-s", type=float, default=0.8)
    p.add_argument("--model-dt-s", type=float, default=0.10)
    p.add_argument("--fragment-breakup-alt-km", type=float, default=30.0)
    p.add_argument("--body-diameter-m", type=float, default=FIXED_BODY_DIAMETER_M)
    p.add_argument("--body-mass-kg", type=float, default=FIXED_BODY_MASS_KG)
    p.add_argument("--body-cd", type=float, default=FIXED_BODY_CD)
    p.add_argument("--fragment-diameter-m", type=float, default=DEFAULT_FRAGMENT_DIAMETER_M)
    p.add_argument("--fragment-diameter-min-m", type=float, default=0.01)
    p.add_argument("--fragment-diameter-max-m", type=float, default=1.0)
    p.add_argument("--fragment-density-kg-m3", type=float, default=DEFAULT_FRAGMENT_DENSITY_KG_M3)
    p.add_argument("--fragment-cd", type=float, default=FIXED_BODY_CD)
    p.add_argument("--fragment-breakup-search-s", type=float, default=30.0)
    p.add_argument("--fragment-impact-max-duration-s", type=float, default=3600.0)
    p.add_argument("--fragment-integration-dt-s", type=float, default=10.0)
    p.add_argument("--fragment-rtol", type=float, default=1e-5)
    p.add_argument("--fragment-atol", type=float, default=1e-1)
    p.add_argument("--impact-uncertainty-samples", type=int, default=60)
    p.add_argument("--impact-nominal-dt-s", type=float, default=2.0)
    p.add_argument("--impact-sample-dt-s", type=float, default=45.0)
    p.add_argument("--altitude-uncertainty-samples", type=int, default=80)
    p.add_argument("--range-doppler-uncertainty-samples", type=int, default=80)
    p.add_argument("--no-drag", action="store_true")
    p.add_argument("--max-nfev", type=int, default=60)
    return p.parse_args()


if __name__ == "__main__":
    main()
