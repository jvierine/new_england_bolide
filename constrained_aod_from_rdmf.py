#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from angle_departure_estimator import load_station_yaml, regular_direction_grid
from estimate_head_echo_trajectory import sample_fit_parameters
from station_config import short_station_name as station_label


C = 299792458.0
DEFAULT_MERGED_DIR = "products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_TRAJECTORY = "products/head_echo_trajectory_roi_fit/head_echo_trajectory.json"


def unit_to_az_el(unit_enu):
    unit_enu = np.asarray(unit_enu, dtype=float)
    az = np.rad2deg(np.arctan2(unit_enu[..., 0], unit_enu[..., 1]))
    az = (az + 360.0) % 360.0
    el = np.rad2deg(np.arcsin(np.clip(unit_enu[..., 2], -1.0, 1.0)))
    return az, el


def angular_distance_deg(a, b):
    dot = np.sum(np.asarray(a) * np.asarray(b), axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def trajectory_position_enu(fit, t_unix):
    p0 = np.array(fit["fit_parameters_enu_at_tx"]["position_ref_m"], dtype=float)
    v = np.array(fit["fit_parameters_enu_at_tx"]["velocity_mps"], dtype=float)
    return p0 + v * (float(t_unix) - float(fit["t_ref_unix"]))


def trajectory_position_enu_from_params(params, fit, t_unix):
    params = np.asarray(params, dtype=float)
    return params[:3] + params[3:6] * (np.asarray(t_unix, dtype=float) - float(fit["t_ref_unix"]))[:, None]


def steering_grid(site, directions, phase_sign):
    ant = np.array(site["antenna_pos_tx"], dtype=float)
    wavelength = C / float(site["frequency_hz"])
    phase = float(phase_sign) * 2.0 * np.pi / wavelength * (directions @ ant.T)
    return np.exp(1j * phase).astype(np.complex64)


def steering_vector(site, direction, phase_sign):
    return steering_grid(site, np.asarray(direction, dtype=float)[None, :], phase_sign)[0].astype(np.complex128)


def score_voltage(voltage, steering):
    y = np.asarray(voltage, dtype=np.complex64).copy()
    norms = np.sqrt(np.sum(np.abs(y) ** 2, axis=1))
    good = norms > 0
    if not np.any(good):
        return np.zeros(steering.shape[0], dtype=np.float32)
    y[good, :] /= norms[good, None]
    coh = y @ np.conj(steering).T
    score = np.sum(np.abs(coh[good, :]) ** 2, axis=0)
    score /= float(np.sum(good) * y.shape[1] ** 2)
    return score.astype(np.float32)


def tangent_basis(unit):
    unit = np.asarray(unit, dtype=float)
    unit = unit / np.linalg.norm(unit)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(unit, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, unit)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def unit_from_tangent(unit0, basis, delta):
    e1, e2 = basis
    unit = np.asarray(unit0, dtype=float) + float(delta[0]) * e1 + float(delta[1]) * e2
    return unit / np.linalg.norm(unit)


def initial_channel_scales(voltage, steering):
    denom = np.vdot(steering, steering)
    return np.array([np.vdot(steering, row) / denom for row in voltage], dtype=np.complex128)


def pack_params(delta, gamma):
    params = np.empty(2 + 2 * len(gamma), dtype=float)
    params[:2] = delta
    params[2::2] = gamma.real
    params[3::2] = gamma.imag
    return params


def unpack_params(params):
    params = np.asarray(params, dtype=float)
    gamma = params[2::2] + 1j * params[3::2]
    return params[:2], gamma


def aod_lstsq_refine(voltage, site, initial_unit, phase_sign, max_iter=30):
    y = np.asarray(voltage, dtype=np.complex128)
    norms = np.sqrt(np.sum(np.abs(y) ** 2, axis=1))
    good = norms > 0
    y = y[good, :]
    n_chan, n_tx = y.shape
    if n_chan == 0:
        return {
            "success": False,
            "message": "no nonzero receiver channels",
            "unit_enu": np.asarray(initial_unit, dtype=float).tolist(),
            "gamma": [],
            "channel_indices": [],
            "n_channels_used": 0,
            "n_tx": int(np.asarray(voltage).shape[-1]),
            "iterations": 0,
            "residual_norm": float("nan"),
            "residual_variance": float("nan"),
            "dof": 0,
            "direction_cov_tangent_rad2": [[float("nan"), float("nan")], [float("nan"), float("nan")]],
            "sigma_major_deg": float("nan"),
            "sigma_minor_deg": float("nan"),
            "sigma_angular_deg": float("nan"),
        }

    unit0 = np.asarray(initial_unit, dtype=float)
    unit0 /= np.linalg.norm(unit0)
    basis = tangent_basis(unit0)
    a0 = steering_vector(site, unit0, phase_sign)
    params = pack_params(np.zeros(2, dtype=float), initial_channel_scales(y, a0))

    def residual_for(p):
        delta, gamma = unpack_params(p)
        unit = unit_from_tangent(unit0, basis, delta)
        a = steering_vector(site, unit, phase_sign)
        err = gamma[:, None] * a[None, :] - y
        return np.r_[err.real.ravel(), err.imag.ravel()]

    def jacobian_for(p, r0):
        jac = np.empty((len(r0), len(p)), dtype=float)
        for i in range(len(p)):
            step = 1e-5 if i < 2 else 1e-6 * max(1.0, abs(float(p[i])))
            pp = p.copy()
            pp[i] += step
            jac[:, i] = (residual_for(pp) - r0) / step
        return jac

    damping = 1e-3
    success = False
    message = "maximum iterations reached"
    r = residual_for(params)
    sse = float(np.dot(r, r))
    initial_sse = sse
    iterations = 0
    for iterations in range(1, max_iter + 1):
        jac = jacobian_for(params, r)
        normal = jac.T @ jac
        rhs = -(jac.T @ r)
        scale = np.maximum(np.diag(normal), 1.0)
        accepted = False
        for _ in range(8):
            try:
                step = np.linalg.solve(normal + damping * np.diag(scale), rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(normal + damping * np.diag(scale)) @ rhs
            trial = params + step
            trial_r = residual_for(trial)
            trial_sse = float(np.dot(trial_r, trial_r))
            if np.isfinite(trial_sse) and trial_sse < sse:
                params = trial
                r = trial_r
                sse = trial_sse
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                break
            damping *= 10.0
        if not accepted:
            message = "no improving Levenberg-Marquardt step"
            break
        if np.linalg.norm(step[:2]) < 1e-8 and np.linalg.norm(step[2:]) < 1e-8 * max(1.0, np.linalg.norm(params[2:])):
            success = True
            message = "converged"
            break

    delta, gamma = unpack_params(params)
    unit = unit_from_tangent(unit0, basis, delta)
    final_r = residual_for(params)
    final_jac = jacobian_for(params, final_r)
    dof = max(1, len(final_r) - len(params))
    sigma2 = float(np.dot(final_r, final_r) / dof)
    cov = sigma2 * np.linalg.pinv(final_jac.T @ final_jac)
    direction_cov = cov[:2, :2]
    eig = np.maximum(np.linalg.eigvalsh(direction_cov), 0.0)
    sigma_minor, sigma_major = np.sqrt(eig)
    return {
        "success": bool(success or message == "converged" or sse < initial_sse),
        "message": message,
        "unit_enu": unit.tolist(),
        "gamma": [[float(g.real), float(g.imag)] for g in gamma],
        "channel_indices": np.flatnonzero(good).astype(int).tolist(),
        "n_channels_used": int(n_chan),
        "n_tx": int(n_tx),
        "iterations": int(iterations),
        "residual_norm": float(np.linalg.norm(final_r)),
        "residual_variance": sigma2,
        "dof": int(dof),
        "direction_cov_tangent_rad2": direction_cov.tolist(),
        "sigma_major_deg": float(np.rad2deg(sigma_major)),
        "sigma_minor_deg": float(np.rad2deg(sigma_minor)),
        "sigma_angular_deg": float(np.rad2deg(np.sqrt(np.trace(direction_cov)))),
    }


def load_rdmf_station(path):
    h = h5py.File(path, "r")
    sample_rate = float(h["sample_rate"][()])
    return {
        "handle": h,
        "station": h["station"][()].decode(),
        "time_unix": h["t_sample"][()] / sample_rate,
        "range_km": h["range_km"][()],
        "snr_db": 10.0 * np.log10(np.maximum(h["P"][()] / np.maximum(h["N"][()], 1e-12), 1e-12)),
    }


def pick_voltage(rdmf, ob, max_time_error_s, max_range_error_km):
    ti = int(np.argmin(np.abs(rdmf["time_unix"] - ob["unix_time"])))
    ri = int(np.argmin(np.abs(rdmf["range_km"] - ob["range_m"] / 1000.0)))
    dt = float(rdmf["time_unix"][ti] - ob["unix_time"])
    dr = float(rdmf["range_km"][ri] - ob["range_m"] / 1000.0)
    if abs(dt) > max_time_error_s or abs(dr) > max_range_error_km:
        return None
    voltage = rdmf["handle"]["voltage_peak"][ti, :, ri, :]
    return ti, ri, dt, dr, voltage, float(rdmf["snr_db"][ti, ri])


def estimate_one(voltage, site, dir_unit, dir_az, dir_el, prior_unit, cone_deg, phase_sign):
    keep = (dir_unit @ prior_unit) >= np.cos(np.deg2rad(cone_deg))
    if not np.any(keep):
        raise ValueError("No search-grid directions fell inside the prior cone")
    dirs = dir_unit[keep]
    steering = steering_grid(site, dirs, phase_sign)
    score = score_voltage(voltage, steering)
    imax = int(np.argmax(score))
    grid_unit = dirs[imax]
    refined = aod_lstsq_refine(voltage, site, grid_unit, phase_sign)
    best_unit = np.array(refined["unit_enu"], dtype=float) if refined["success"] else grid_unit
    best_az, best_el = unit_to_az_el(best_unit)
    grid_az, grid_el = unit_to_az_el(grid_unit)
    prior_az, prior_el = unit_to_az_el(prior_unit)
    return {
        "best_unit_enu": best_unit.tolist(),
        "best_az_deg": float(best_az),
        "best_el_deg": float(best_el),
        "grid_unit_enu": grid_unit.tolist(),
        "grid_az_deg": float(grid_az),
        "grid_el_deg": float(grid_el),
        "prior_az_deg": float(prior_az),
        "prior_el_deg": float(prior_el),
        "prior_offset_deg": float(angular_distance_deg(best_unit, prior_unit)),
        "score": float(score[imax]),
        "grid_score": float(score[imax]),
        "ls_success": bool(refined["success"]),
        "ls_message": refined["message"],
        "ls_iterations": refined["iterations"],
        "ls_residual_norm": refined["residual_norm"],
        "ls_residual_variance": refined["residual_variance"],
        "ls_dof": refined["dof"],
        "ls_channel_indices": refined["channel_indices"],
        "ls_channel_scales": refined["gamma"],
        "ls_direction_cov_tangent_rad2": refined["direction_cov_tangent_rad2"],
        "ls_sigma_major_deg": refined["sigma_major_deg"],
        "ls_sigma_minor_deg": refined["sigma_minor_deg"],
        "ls_sigma_angular_deg": refined["sigma_angular_deg"],
        "ls_grid_to_fit_offset_deg": float(angular_distance_deg(best_unit, grid_unit)),
        "n_search_directions": int(np.sum(keep)),
    }


def summarize(rows):
    out = {}
    for station in sorted(set(r["station"] for r in rows)):
        rr = [r for r in rows if r["station"] == station]
        score = np.array([r["score"] for r in rr], dtype=float)
        weights = np.maximum(score, 1e-6)
        best_unit = np.array([r["best_unit_enu"] for r in rr], dtype=float)
        prior_unit = np.array([r["prior_unit_enu"] for r in rr], dtype=float)
        mean_best = np.average(best_unit, axis=0, weights=weights)
        mean_prior = np.average(prior_unit, axis=0, weights=weights)
        mean_best /= np.linalg.norm(mean_best)
        mean_prior /= np.linalg.norm(mean_prior)
        best_az, best_el = unit_to_az_el(mean_best)
        prior_az, prior_el = unit_to_az_el(mean_prior)
        out[station] = {
            "n": len(rr),
            "weighted_best_az_deg": float(best_az),
            "weighted_best_el_deg": float(best_el),
            "weighted_prior_az_deg": float(prior_az),
            "weighted_prior_el_deg": float(prior_el),
            "weighted_prior_offset_deg": float(angular_distance_deg(mean_best, mean_prior)),
            "median_per_pick_prior_offset_deg": float(np.median([r["prior_offset_deg"] for r in rr])),
            "median_score": float(np.median(score)),
        }
    return out


def trajectory_covariance_samples(fit, n_samples=200):
    err = fit.get("linearized_fit_error", {})
    covariance = err.get("covariance")
    if covariance is None or n_samples <= 0:
        return np.empty((0, 6), dtype=float)
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    return sample_fit_parameters(params, np.array(covariance, dtype=float), n_samples=n_samples, seed=20260530)


def plot_results(rows, summary, out_png, fit=None, uncertainty_samples=None):
    stations = sorted(summary)
    fig, axes = plt.subplots(
        1,
        len(stations),
        figsize=(4.8 * len(stations), 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    all_az = []
    all_el = []
    for station in stations:
        rr = [r for r in rows if r["station"] == station]
        all_az.extend([r["prior_az_deg"] for r in rr])
        all_az.extend([r["best_az_deg"] for r in rr])
        all_el.extend([r["prior_el_deg"] for r in rr])
        all_el.extend([r["best_el_deg"] for r in rr])

    for ax, station in zip(axes, stations):
        rr = sorted([r for r in rows if r["station"] == station], key=lambda r: r["unix_time"])
        times = np.array([r["unix_time"] for r in rr], dtype=float)
        if fit is not None and uncertainty_samples is not None and len(uncertainty_samples) > 0:
            for si, params in enumerate(uncertainty_samples):
                pos = trajectory_position_enu_from_params(params, fit, times)
                unit = pos / np.linalg.norm(pos, axis=1)[:, None]
                az, el = unit_to_az_el(unit)
                ax.plot(
                    az,
                    el,
                    color="0.55",
                    lw=0.8,
                    alpha=0.22,
                    label="tri-static AoD samples" if si == 0 else None,
                    zorder=1,
                )
        ax.plot(
            [r["prior_az_deg"] for r in rr],
            [r["prior_el_deg"] for r in rr],
            "-",
            lw=2.2,
            color="0.05",
            alpha=0.9,
            label="tri-static predicted AoD",
            zorder=3,
        )
        sc = ax.scatter(
            [r["best_az_deg"] for r in rr],
            [r["best_el_deg"] for r in rr],
            c=[r["snr_db"] for r in rr],
            s=28,
            cmap="viridis",
            edgecolor="0.15",
            linewidth=0.25,
            alpha=0.9,
            label="RDMF voltage AoD",
            zorder=4,
        )
        ax.set_title(f"{station_label(station)} (n={len(rr)})")
        ax.set_xlabel("Azimuth from north (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Elevation (deg)")

    az_pad = max(1.0, 0.08 * (max(all_az) - min(all_az)))
    el_pad = max(1.0, 0.12 * (max(all_el) - min(all_el)))
    for ax in axes:
        ax.set_xlim(min(all_az) - az_pad, max(all_az) + az_pad)
        ax.set_ylim(min(all_el) - el_pad, max(all_el) + el_pad)
    cb = fig.colorbar(sc, ax=axes, shrink=0.86, pad=0.015)
    cb.set_label("SNR (dB)")
    fig.suptitle("Constrained Transmit AoD And Tri-Static Trajectory Prediction")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Constrain saved per-seed RDMF voltages with the head-echo trajectory prior.")
    p.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY)
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--stations", nargs="+", default=None)
    p.add_argument("--cone-deg", type=float, default=20.0)
    p.add_argument("--grid-size", type=int, default=361)
    p.add_argument("--max-zenith-angle", type=float, default=90.0)
    p.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    p.add_argument("--min-snr-db", type=float, default=5.0)
    p.add_argument("--max-time-error-s", type=float, default=0.01)
    p.add_argument("--max-range-error-km", type=float, default=1.0)
    p.add_argument("--uncertainty-samples", type=int, default=200)
    p.add_argument("--out-json", default="products/constrained_aod/constrained_aod_summary.json")
    p.add_argument("--out-png", default="products/constrained_aod/constrained_aod_summary.png")
    args = p.parse_args()

    fit = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    wanted = set(args.stations or fit["used_stations"])
    observations = [ob for ob in fit["observations"] if ob["station"] in wanted and ob["snr_db"] >= args.min_snr_db]
    if not observations:
        raise SystemExit("No trajectory observations matched the station/SNR filters")

    dir_unit, dir_az, dir_el = regular_direction_grid(args.grid_size, args.max_zenith_angle)
    sites = {station: load_station_yaml(args.data_dir, station) for station in wanted}
    rdmf = {}
    try:
        for station in wanted:
            path = Path(args.merged_dir) / f"rdmf_peak_{station}_merged.h5"
            if not path.exists():
                continue
            rdmf[station] = load_rdmf_station(path)
        rows = []
        skipped = 0
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
            est = estimate_one(voltage, sites[station], dir_unit, dir_az, dir_el, prior_unit, args.cone_deg, args.phase_sign)
            est.update(
                {
                    "station": station,
                    "unix_time": float(ob["unix_time"]),
                    "range_km": float(ob["range_m"] / 1000.0),
                    "snr_db": float(snr_db),
                    "time_index": ti,
                    "range_index": ri,
                    "time_error_s": dt,
                    "range_error_km": dr,
                    "prior_unit_enu": prior_unit.tolist(),
                }
            )
            rows.append(est)
    finally:
        for val in rdmf.values():
            val["handle"].close()

    if not rows:
        raise SystemExit("No RDMF voltage samples matched the trajectory observations")

    summary = summarize(rows)
    payload = {
        "schema": "new_england_bolide.constrained_aod.v1",
        "trajectory_json": args.trajectory_json,
        "merged_dir": args.merged_dir,
        "phase_sign": args.phase_sign,
        "cone_deg": args.cone_deg,
        "grid_size": args.grid_size,
        "min_snr_db": args.min_snr_db,
        "n_observations": len(observations),
        "n_used": len(rows),
        "n_skipped": skipped,
        "summary_by_station": summary,
        "estimates": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    uncertainty_samples = trajectory_covariance_samples(fit, n_samples=args.uncertainty_samples)
    plot_results(rows, summary, args.out_png, fit=fit, uncertainty_samples=uncertainty_samples)

    print(f"used={len(rows)} skipped={skipped} cone={args.cone_deg:g}deg phase_sign={args.phase_sign:g}")
    for station, stats in summary.items():
        print(
            f"{station}: n={stats['n']} "
            f"aod=az {stats['weighted_best_az_deg']:.2f} el {stats['weighted_best_el_deg']:.2f} deg "
            f"prior=az {stats['weighted_prior_az_deg']:.2f} el {stats['weighted_prior_el_deg']:.2f} deg "
            f"offset={stats['weighted_prior_offset_deg']:.2f} deg "
            f"median_pick_offset={stats['median_per_pick_prior_offset_deg']:.2f} deg"
        )
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
