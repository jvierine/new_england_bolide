#!/usr/bin/env python3
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml
from plot_head_echo_roi_points import load_points


C = 299792458.0
DEFAULT_ANNOTATIONS = "annotations/head_echo_rois.json"
DEFAULT_OUT_DIR = "products/head_echo_trajectory"


def iso_utc(t):
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


def ecef_to_geodetic_spherical(x):
    r = np.linalg.norm(x)
    lat = np.rad2deg(np.arcsin(x[2] / r))
    lon = np.rad2deg(np.arctan2(x[1], x[0]))
    alt = r - 6371000.0
    return lat, lon, alt


def load_rdmf(fname):
    with h5py.File(fname, "r") as h:
        sr = float(h["sample_rate"][()])
        return {
            "station": h["station"][()].decode(),
            "t_unix": h["t_sample"][()] / sr,
            "range_km": h["range_km"][()],
            "P": h["P"][()],
            "N": h["N"][()],
            "D": h["D"][()],
            "frequency_hz": float(h["frequency_hz"][()]),
        }


def roi_observations(annotation, snr_threshold_db=0.0):
    data = load_rdmf(annotation["source_file"])
    t = data["t_unix"]
    r = data["range_km"]
    t0 = annotation["time_range"]["start_unix"]
    t1 = annotation["time_range"]["end_unix"]
    r0 = annotation["range_km"]["min"]
    r1 = annotation["range_km"]["max"]
    keep_t = np.where((t >= min(t0, t1)) & (t <= max(t0, t1)))[0]
    keep_r = np.where((r >= min(r0, r1)) & (r <= max(r0, r1)))[0]
    out = []
    if len(keep_t) == 0 or len(keep_r) == 0:
        return out

    snr = data["P"] / np.maximum(data["N"], 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
    for ti in keep_t:
        row = snr_db[ti, keep_r]
        imax = int(np.nanargmax(row))
        if row[imax] < snr_threshold_db:
            continue
        ri = keep_r[imax]
        out.append(
            {
                "station": data["station"],
                "unix_time": float(t[ti]),
                "range_m": float(r[ri] * 1000.0),
                "snr_db": float(snr_db[ti, ri]),
                "doppler_hz": float(data["D"][ti, ri]),
            }
        )
    return out


def load_observations(annotation_file, snr_threshold_db, reject_outliers=True, delay_sigma=4.0, doppler_sigma=4.0):
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    points, rejected = load_points(
        annotation_file,
        snr_threshold_db,
        reject=reject_outliers,
        delay_sigma=delay_sigma,
        doppler_sigma=doppler_sigma,
    )
    obs = [
        {
            "station": p["station"],
            "unix_time": p["unix_time"],
            "range_m": p["range_km"] * 1000.0,
            "snr_db": p["snr_db"],
            "doppler_hz": p["doppler_hz"],
        }
        for p in points
    ]
    obs.sort(key=lambda x: x["unix_time"])
    used_stations = sorted(set(o["station"] for o in obs))
    return annotations, obs, used_stations, rejected


def station_geometry(data_dir, stations):
    geom = {}
    for station in stations:
        site = load_station_yaml(data_dir, station)
        geom[station] = {
            "tx": geodetic_to_ecef(*site["gps_tx"]),
            "rx": geodetic_to_ecef(*site["gps_rx"]),
            "frequency_hz": site["frequency_hz"],
            "gps_tx": site["gps_tx"].tolist(),
            "gps_rx": site["gps_rx"].tolist(),
        }
    return geom


def predict_ranges(params, obs, geom, t_ref, tx0, rot):
    p0_enu = params[:3]
    v_enu = params[3:6]
    pred = np.zeros(len(obs))
    for i, ob in enumerate(obs):
        pos = tx0 + rot @ (p0_enu + v_enu * (ob["unix_time"] - t_ref))
        g = geom[ob["station"]]
        pred[i] = np.linalg.norm(pos - g["tx"]) + np.linalg.norm(pos - g["rx"])
    return pred


def predict_dopplers(params, obs, geom, t_ref, tx0, rot):
    p0_enu = params[:3]
    v_enu = params[3:6]
    pred = np.zeros(len(obs))
    for i, ob in enumerate(obs):
        g = geom[ob["station"]]
        dt = ob["unix_time"] - t_ref
        pos = tx0 + rot @ (p0_enu + v_enu * dt)
        vel = rot @ v_enu
        u_tx = (pos - g["tx"]) / np.linalg.norm(pos - g["tx"])
        u_rx = (pos - g["rx"]) / np.linalg.norm(pos - g["rx"])
        drdt = float(np.dot(u_tx + u_rx, vel))
        # Bistatic Doppler convention used in the NM analysis:
        # f_D = (1 / lambda) d/dt (|x - tx| + |x - rx|).
        pred[i] = g["frequency_hz"] * drdt / C
    return pred


def residuals(params, obs, geom, t_ref, tx0, rot, sigma_range_m):
    pred = predict_ranges(params, obs, geom, t_ref, tx0, rot)
    meas = np.array([o["range_m"] for o in obs])
    return (pred - meas) / sigma_range_m


def doppler_residuals(params, obs, geom, t_ref, tx0, rot, frequency_hz, sigma_doppler_hz):
    pred = predict_dopplers(params, obs, geom, t_ref, tx0, rot)
    meas = np.array([o["doppler_hz"] for o in obs])
    return (pred - meas) / sigma_doppler_hz


def combined_residuals(params, obs, geom, t_ref, tx0, rot, sigma_range_m, use_doppler, sigma_doppler_hz):
    rr = residuals(params, obs, geom, t_ref, tx0, rot, sigma_range_m)
    if not use_doppler:
        return rr
    return np.concatenate([rr, doppler_residuals(params, obs, geom, t_ref, tx0, rot, None, sigma_doppler_hz)])


def static_position_residuals(p_enu, station_picks, geom, tx0, rot, sigma_range_m):
    pos = tx0 + rot @ p_enu
    res = []
    for ob in station_picks:
        g = geom[ob["station"]]
        pred = np.linalg.norm(pos - g["tx"]) + np.linalg.norm(pos - g["rx"])
        res.append((pred - ob["range_m"]) / sigma_range_m)
    return np.array(res)


def strongest_pick_per_station(obs):
    picks = []
    for station in sorted(set(o["station"] for o in obs)):
        station_obs = [o for o in obs if o["station"] == station]
        picks.append(max(station_obs, key=lambda o: o["snr_db"]))
    return picks


def initial_position_from_ranges(obs, geom, tx0, rot):
    picks = strongest_pick_per_station(obs)
    guesses = []
    for east_km in (-200, -100, 0, 100, 200):
        for north_km in (-200, -100, 0, 100, 200):
            for up_km in (50, 70, 90, 110):
                guesses.append(np.array([east_km * 1e3, north_km * 1e3, up_km * 1e3]))

    best = None
    bounds = (np.array([-500e3, -500e3, 20e3]), np.array([500e3, 500e3, 160e3]))
    for x0 in guesses:
        res = least_squares(
            static_position_residuals,
            x0,
            bounds=bounds,
            args=(picks, geom, tx0, rot, 750.0),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=500,
        )
        score = float(np.sum(res.fun**2))
        if best is None or score < best[0]:
            best = (score, res)
    return best[1].x, picks


def fit_trajectory(obs, geom, data_dir, use_doppler=True):
    tx_site = load_station_yaml(data_dir, obs[0]["station"])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    t_ref = float(np.median([o["unix_time"] for o in obs]))
    p_init, station_picks = initial_position_from_ranges(obs, geom, tx0, rot)

    guesses = []
    for speed in (12000.0, 20000.0, 30000.0, 40000.0):
        for az_deg in np.arange(0.0, 360.0, 45.0):
            az = np.deg2rad(az_deg)
            for vz in (-15000.0, -5000.0, 5000.0):
                vh = np.sqrt(max(speed * speed - vz * vz, 0.0))
                ve = vh * np.sin(az)
                vn = vh * np.cos(az)
                guesses.append(np.r_[p_init, [ve, vn, vz]])

    best = None
    bounds = (
        np.array([-500e3, -500e3, 20e3, -50000.0, -50000.0, -50000.0]),
        np.array([500e3, 500e3, 160e3, 50000.0, 50000.0, 50000.0]),
    )
    for x0 in guesses:
        res = least_squares(
            combined_residuals,
            x0,
            bounds=bounds,
            args=(obs, geom, t_ref, tx0, rot, 750.0, use_doppler, 500.0),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=1200,
        )
        score = np.sum(res.fun**2)
        if best is None or score < best[0]:
            best = (score, res)

    score, res = best
    pred = predict_ranges(res.x, obs, geom, t_ref, tx0, rot)
    pred_doppler = predict_dopplers(res.x, obs, geom, t_ref, tx0, rot)
    meas = np.array([o["range_m"] for o in obs])
    range_resid = pred - meas
    doppler_resid = pred_doppler - np.array([o["doppler_hz"] for o in obs])
    pos_ref = tx0 + rot @ res.x[:3]
    vel_ecef = rot @ res.x[3:6]
    lat, lon, alt = ecef_to_geodetic_spherical(pos_ref)
    return {
        "result": res,
        "score": score,
        "t_ref": t_ref,
        "tx0": tx0,
        "rot": rot,
        "range_residual_m": range_resid,
        "doppler_residual_hz": doppler_resid,
        "pred_range_m": pred,
        "pred_doppler_hz": pred_doppler,
        "position_ref_ecef_m": pos_ref,
        "velocity_ecef_mps": vel_ecef,
        "lat_ref_deg_spherical": lat,
        "lon_ref_deg_spherical": lon,
        "alt_ref_m_spherical": alt,
        "initial_station_picks": station_picks,
    }


def write_outputs(out_dir, annotations, obs, used_stations, geom, fit, rejected_points, use_doppler):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit["result"]
    speed = float(np.linalg.norm(res.x[3:6]))
    payload = {
        "schema": "new_england_bolide.head_echo_trajectory.v1",
        "source_annotations": annotations,
        "used_stations": used_stations,
        "n_observations": len(obs),
        "n_rejected_roi_points": len(rejected_points),
        "fit_uses_doppler": use_doppler,
        "doppler_model": "f_D = (f0/c) * (u_tx + u_rx) dot v, positive for increasing bistatic path length",
        "initial_station_picks": fit["initial_station_picks"],
        "t_ref_unix": fit["t_ref"],
        "t_ref_utc": iso_utc(fit["t_ref"]),
        "fit_parameters_enu_at_tx": {
            "position_ref_m": res.x[:3].tolist(),
            "velocity_mps": res.x[3:6].tolist(),
            "speed_mps": speed,
            "speed_kmps": speed / 1000.0,
        },
        "position_ref_ecef_m": fit["position_ref_ecef_m"].tolist(),
        "velocity_ecef_mps": fit["velocity_ecef_mps"].tolist(),
        "spherical_geodetic_ref": {
            "lat_deg": float(fit["lat_ref_deg_spherical"]),
            "lon_deg": float(fit["lon_ref_deg_spherical"]),
            "alt_m": float(fit["alt_ref_m_spherical"]),
        },
        "range_residual_m": {
            "rms": float(np.sqrt(np.mean(fit["range_residual_m"] ** 2))),
            "median": float(np.median(fit["range_residual_m"])),
            "p95_abs": float(np.percentile(np.abs(fit["range_residual_m"]), 95)),
        },
        "doppler_residual_hz": {
            "rms": float(np.sqrt(np.mean(fit["doppler_residual_hz"] ** 2))),
            "median": float(np.median(fit["doppler_residual_hz"])),
            "p95_abs": float(np.percentile(np.abs(fit["doppler_residual_hz"]), 95)),
        },
        "observations": obs,
    }
    (out_dir / "head_echo_trajectory.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    t = np.array([o["unix_time"] - fit["t_ref"] for o in obs])
    stations = sorted(set(o["station"] for o in obs))
    for station in stations:
        idx = np.array([i for i, o in enumerate(obs) if o["station"] == station])
        ax.scatter(t[idx], np.array([obs[i]["range_m"] for i in idx]) / 1000.0, s=18, label=f"{station} obs")
        ax.plot(t[idx], fit["pred_range_m"][idx] / 1000.0, lw=1.5, label=f"{station} fit")
    ax.set_xlabel(f"Seconds relative to {iso_utc(fit['t_ref'])}")
    ax.set_ylabel("Bistatic range (km)")
    ax.set_title("Head echo range observations and straight-line trajectory fit")
    ax.grid(True, color="0.9")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "head_echo_range_fit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    p0 = res.x[:3] / 1000.0
    v = res.x[3:6] / 1000.0
    ts = np.linspace(min(t), max(t), 100)
    traj = p0[None, :] + ts[:, None] * v[None, :]
    ax.plot(traj[:, 0], traj[:, 1], lw=2, label="fit")
    ax.scatter(p0[0], p0[1], c="k", s=40, label="t_ref")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East from Haystack TX (km)")
    ax.set_ylabel("North from Haystack TX (km)")
    ax.set_title(f"Horizontal trajectory, speed={speed/1000.0:.1f} km/s")
    ax.grid(True, color="0.9")
    ax.legend()
    fig.savefig(out_dir / "head_echo_horizontal_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return payload


def show_interactive_fit(obs, fit):
    t_ref = fit["t_ref"]
    t = np.array([o["unix_time"] - t_ref for o in obs])
    stations = sorted(set(o["station"] for o in obs))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)
    for station in stations:
        idx = np.array([i for i, o in enumerate(obs) if o["station"] == station])
        order = idx[np.argsort(t[idx])]
        delay_ms = np.array([obs[i]["range_m"] for i in order]) / C * 1e3
        fit_delay_ms = fit["pred_range_m"][order] / C * 1e3
        doppler_khz = np.array([obs[i]["doppler_hz"] for i in order]) / 1e3
        fit_doppler_khz = fit["pred_doppler_hz"][order] / 1e3
        axes[0].scatter(t[order], delay_ms, s=26, label=f"{station} obs")
        axes[0].plot(t[order], fit_delay_ms, lw=1.8, label=f"{station} fit")
        axes[1].scatter(t[order], doppler_khz, s=26, label=f"{station} obs")
        axes[1].plot(t[order], fit_doppler_khz, lw=1.8, label=f"{station} fit")

    range_axis = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda delay_ms: delay_ms * C / 1e6,
            lambda range_km: range_km * 1e6 / C,
        ),
    )
    range_axis.set_ylabel("Bistatic range (km)")
    axes[0].set_ylabel("Delay (ms)")
    axes[0].set_title("Joint head echo trajectory fit: delay/range and bistatic Doppler")
    axes[0].grid(True, color="0.9")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel(f"Seconds relative to {iso_utc(t_ref)}")
    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].grid(True, color="0.9")
    axes[1].legend(ncol=2, fontsize=8)
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Estimate a straight-line head-echo trajectory from manual RDMF ROIs.")
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--snr-threshold-db", type=float, default=5.0)
    p.add_argument("--no-use-doppler", action="store_true")
    p.add_argument("--no-reject-outliers", action="store_true")
    p.add_argument("--delay-sigma", type=float, default=4.0)
    p.add_argument("--doppler-sigma", type=float, default=4.0)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    annotations, obs, used_stations, rejected = load_observations(
        args.annotations,
        args.snr_threshold_db,
        reject_outliers=not args.no_reject_outliers,
        delay_sigma=args.delay_sigma,
        doppler_sigma=args.doppler_sigma,
    )
    if len(used_stations) < 3:
        raise SystemExit(f"Need at least three head-echo station ROIs; got {used_stations}")
    geom = station_geometry(args.data_dir, used_stations)
    use_doppler = not args.no_use_doppler
    fit = fit_trajectory(obs, geom, args.data_dir, use_doppler=use_doppler)
    payload = write_outputs(args.out_dir, annotations, obs, used_stations, geom, fit, rejected, use_doppler)
    print(f"Used stations: {', '.join(used_stations)}")
    print(f"Observations: {payload['n_observations']}")
    print(f"Rejected ROI outliers: {payload['n_rejected_roi_points']}")
    print(f"Reference time: {payload['t_ref_utc']}")
    print(f"Speed: {payload['fit_parameters_enu_at_tx']['speed_kmps']:.2f} km/s")
    print(f"Reference spherical lat/lon/alt: {payload['spherical_geodetic_ref']}")
    print(f"Range residual RMS: {payload['range_residual_m']['rms']:.1f} m")
    print(f"Doppler residual RMS: {payload['doppler_residual_hz']['rms']:.1f} Hz")
    print(f"Wrote {Path(args.out_dir) / 'head_echo_trajectory.json'}")
    if not args.no_show:
        show_interactive_fit(obs, fit)


if __name__ == "__main__":
    main()
