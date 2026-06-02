#!/usr/bin/env python3
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.spatial import ConvexHull

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


def apparent_radiant(fit, tx0, rot, params):
    obs = fit["observations"]
    t_first = min(o["unix_time"] for o in obs)
    first_state = model_state_series(params, [t_first], fit["t_ref_unix"], tx0, rot)
    first_lat, first_lon, first_alt_m = first_state["lat_lon_alt"][0]
    v_enu_tx = np.array(fit["fit_parameters_enu_at_tx"]["velocity_mps"], dtype=float)
    v_ecef = rot @ v_enu_tx
    radiant_ecef = -v_ecef / np.linalg.norm(v_ecef)
    local_rot = enu_to_ecef_matrix(first_lat, first_lon)
    radiant_enu = local_rot.T @ radiant_ecef
    az_deg = (np.rad2deg(np.arctan2(radiant_enu[0], radiant_enu[1])) + 360.0) % 360.0
    el_deg = np.rad2deg(np.arcsin(np.clip(radiant_enu[2], -1.0, 1.0)))
    loc = EarthLocation(lat=first_lat * u.deg, lon=first_lon * u.deg, height=first_alt_m * u.m)
    altaz = SkyCoord(
        az=az_deg * u.deg,
        alt=el_deg * u.deg,
        frame=AltAz(obstime=Time(t_first, format="unix"), location=loc),
    )
    icrs = altaz.icrs
    return {
        "az_deg": float(az_deg),
        "el_deg": float(el_deg),
        "ra_deg": float(icrs.ra.deg),
        "dec_deg": float(icrs.dec.deg),
    }


def plot_altitude_summary(fit, tx0, rot, params, samples):
    obs = fit["observations"]
    t_ref = fit["t_ref_unix"]
    t_obs = np.array([o["unix_time"] for o in obs], dtype=float)
    t_grid = np.linspace(float(np.min(t_obs)), float(np.max(t_obs)), 220)
    model = model_state_series(params, t_grid, t_ref, tx0, rot)
    sample_states = [model_state_series(p, t_grid, t_ref, tx0, rot) for p in samples]
    sample_alt_km = np.array([s["lat_lon_alt"][:, 2] / 1000.0 for s in sample_states])
    alt_lo, alt_hi = np.percentile(sample_alt_km, [2.5, 97.5], axis=0)
    radiant = apparent_radiant(fit, tx0, rot, params)
    speed_kms = fit["fit_parameters_enu_at_tx"]["speed_kmps"]

    fig, ax = plt.subplots(figsize=(12.0, 6.6), constrained_layout=True)
    tr = model["time_relative_s"]
    ax.fill_between(tr, alt_lo, alt_hi, color="C0", alpha=0.2, linewidth=0, label="95% fit uncertainty")
    ax.plot(tr, model["lat_lon_alt"][:, 2] / 1000.0, color="C0", lw=2.2, label="fit")
    ax.scatter([0.0], [fit["geodetic_ref_wgs84"]["alt_m"] / 1000.0], color="k", s=32, zorder=4, label="t_ref")
    ax.set_xlabel(f"Seconds relative to {fit['t_ref_utc']}")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Best-fit head echo altitude")
    ax.grid(True, color="0.9", linewidth=0.6)
    text = (
        f"Speed = {speed_kms:.2f} km/s\n"
        f"Radiant RA = {radiant['ra_deg']:.2f} deg, Dec = {radiant['dec_deg']:.2f} deg\n"
        f"Radiant Az = {radiant['az_deg']:.2f} deg, El = {radiant['el_deg']:.2f} deg\n"
        "No zenithal-attraction correction"
    )
    ax.text(
        0.03,
        0.06,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.65", alpha=0.92),
    )
    ax.legend(loc="upper right")

    out = FIG_DIR / "head_echo_altitude_velocity.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _sample_hull(sample_points):
    cloud = np.reshape(np.array(sample_points), (-1, 2))
    good = np.all(np.isfinite(cloud), axis=1)
    cloud = cloud[good]
    if len(cloud) < 3:
        return None
    center = np.median(cloud, axis=0)
    dist = np.linalg.norm(cloud - center[None, :], axis=1)
    cloud95 = cloud[dist <= np.percentile(dist, 95.0)]
    if len(cloud95) < 3:
        return None
    return cloud95[ConvexHull(cloud95).vertices]


def _annotate_altitudes(ax, lon, lat, alt_km, data_crs):
    levels = np.array([320.0, 260.0, 200.0, 140.0, 100.0, 40.0, 10.0])
    order = np.argsort(alt_km)
    alt_sorted = alt_km[order]
    lon_sorted = lon[order]
    lat_sorted = lat[order]
    _, unique_idx = np.unique(np.round(alt_sorted, 6), return_index=True)
    unique_idx = np.sort(unique_idx)
    alt_unique = alt_sorted[unique_idx]
    lon_unique = lon_sorted[unique_idx]
    lat_unique = lat_sorted[unique_idx]
    offsets = [(-18, -24), (-10, -24), (0, -24), (10, -24), (18, -24), (26, -24), (34, -24)]
    for level, (dx, dy) in zip(levels, offsets, strict=True):
        if level < alt_unique[0] or level > alt_unique[-1]:
            continue
        x = float(np.interp(level, alt_unique, lon_unique))
        y = float(np.interp(level, alt_unique, lat_unique))
        ax.annotate(
            f"{level:.0f} km",
            xy=(x, y),
            xycoords=data_crs._as_mpl_transform(ax),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="0.1",
            zorder=10,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.45, shrinkA=0, shrinkB=1.5),
        )


def plot_cartopy_report_figure(fit, geom, tx0, rot, params, samples):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as exc:
        print(f"Skipping Cartopy map: {exc}")
        return None

    obs = fit["observations"]
    t_ref = fit["t_ref_unix"]
    t_obs = np.array([o["unix_time"] for o in obs], dtype=float)
    t_first = float(np.min(t_obs))
    t_last = float(np.max(t_obs))
    t_all = np.linspace(t_first - 10.0, t_last + 3.0, 420)

    model = model_state_series(params, t_all, t_ref, tx0, rot)
    lon = model["lat_lon_alt"][:, 1]
    lat = model["lat_lon_alt"][:, 0]
    alt_km = model["lat_lon_alt"][:, 2] / 1000.0
    pre = t_all < t_first
    detect = (t_all >= t_first) & (t_all <= t_last)
    post = t_all > t_last

    sample_lonlat = []
    for sample in samples:
        sample_model = model_state_series(sample, t_all, t_ref, tx0, rot)
        sample_lonlat.append(sample_model["lat_lon_alt"][:, [1, 0]])
    sample_lonlat = np.array(sample_lonlat)

    station_lons = []
    station_lats = []
    for g in geom.values():
        rx_lat, rx_lon, _ = g["gps_rx"]
        tx_lat, tx_lon, _ = g["gps_tx"]
        station_lons.extend([rx_lon, tx_lon])
        station_lats.extend([rx_lat, tx_lat])

    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9.2, 7.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    cape_cod_extent = [-71.8, -69.2, 41.2, 42.9]

    hull_poly = _sample_hull(sample_lonlat)
    extra_lon = hull_poly[:, 0] if hull_poly is not None else np.array([])
    extra_lat = hull_poly[:, 1] if hull_poly is not None else np.array([])
    all_lon = np.concatenate([lon, extra_lon, np.array(station_lons)])
    all_lat = np.concatenate([lat, extra_lat, np.array(station_lats)])
    pad_lon = max(0.15, 0.2 * (np.nanmax(all_lon) - np.nanmin(all_lon)))
    pad_lat = max(0.15, 0.2 * (np.nanmax(all_lat) - np.nanmin(all_lat)))
    extent = [
        min(cape_cod_extent[0], np.nanmin(all_lon) - pad_lon),
        max(cape_cod_extent[1], np.nanmax(all_lon) + pad_lon),
        min(cape_cod_extent[2], np.nanmin(all_lat) - pad_lat),
        max(cape_cod_extent[3], np.nanmax(all_lat) + pad_lat),
    ]
    ax.set_extent(extent, crs=data_crs)
    ax.set_facecolor("0.96")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none", zorder=0)
    ax.coastlines(resolution="10m", linewidth=0.9, color="0.15", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.35", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.7", alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False
    if hull_poly is not None:
        ax.fill(
            hull_poly[:, 0],
            hull_poly[:, 1],
            color="C0",
            alpha=0.18,
            transform=data_crs,
            label="sampled 95% covariance region",
            zorder=2,
        )
    ax.plot(lon[pre], lat[pre], "--", color="C1", lw=2.2, transform=data_crs, label="-10 s extrapolation", zorder=4)
    ax.plot(lon[detect], lat[detect], "-", color="C3", lw=2.7, transform=data_crs, label="fit over detections", zorder=5)
    ax.plot(lon[post], lat[post], "--", color="C2", lw=2.2, transform=data_crs, label="+3 s extrapolation", zorder=4)
    ax.scatter(lon[detect][0], lat[detect][0], s=45, color="k", transform=data_crs, zorder=6, label="first detection")
    ax.scatter(lon[detect][-1], lat[detect][-1], s=45, facecolor="white", edgecolor="k", transform=data_crs, zorder=6, label="last detection")
    _annotate_altitudes(ax, lon, lat, alt_km, data_crs)

    tx_seen = set()
    receiver_labeled = False
    for station, g in geom.items():
        rx_lat, rx_lon, _ = g["gps_rx"]
        tx_lat, tx_lon, _ = g["gps_tx"]
        tx_key = (round(tx_lat, 5), round(tx_lon, 5))
        ax.scatter(
            rx_lon,
            rx_lat,
            marker="^",
            s=58,
            color="0.1",
            transform=data_crs,
            zorder=7,
            label=None if receiver_labeled else "receivers",
        )
        receiver_labeled = True
        ax.text(rx_lon + 0.025, rx_lat + 0.015, station, fontsize=8, transform=data_crs, zorder=8, clip_on=True)
        if tx_key not in tx_seen:
            ax.scatter(tx_lon, tx_lat, marker="*", s=110, color="gold", edgecolor="0.2", transform=data_crs, zorder=7, label="transmitter")
            ax.text(tx_lon + 0.025, tx_lat + 0.015, "TX", fontsize=8, weight="bold", transform=data_crs, zorder=8, clip_on=True)
            tx_seen.add(tx_key)
    ax.set_title("Head-echo trajectory fit with straight-line extrapolation")
    ax.legend(loc="best", fontsize=8)

    out = FIG_DIR / "head_echo_cartopy_trajectory_uncertainty.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def write_trajectory_stats(fit, tx0, rot, params):
    obs = fit["observations"]
    t_first = min(o["unix_time"] for o in obs)
    first_state = model_state_series(params, [t_first], fit["t_ref_unix"], tx0, rot)
    first_lat, first_lon, first_alt_m = first_state["lat_lon_alt"][0]
    ref_lat, ref_lon, ref_alt_m = ecef_to_geodetic_wgs84(np.array(fit["position_ref_ecef_m"]))
    v_enu = np.array(fit["fit_parameters_enu_at_tx"]["velocity_mps"])
    speed = np.linalg.norm(v_enu)
    radiant = apparent_radiant(fit, tx0, rot, params)
    first_utc = datetime.fromtimestamp(t_first, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    lines = [
        f"\\newcommand{{\\TrajectoryNObservations}}{{{fit['n_observations']}}}\n",
        f"\\newcommand{{\\TrajectoryNRejected}}{{{fit['n_rejected_roi_points']}}}\n",
        f"\\newcommand{{\\TrajectoryPomfretStartTrimS}}{{{fit['station_start_trims_s'].get('pomfret', 0.0):.2f}}}\n",
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
        f"\\newcommand{{\\TrajectoryRadiantRADeg}}{{{radiant['ra_deg']:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryRadiantDecDeg}}{{{radiant['dec_deg']:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryRadiantAzDeg}}{{{radiant['az_deg']:.2f}}}\n",
        f"\\newcommand{{\\TrajectoryRadiantElDeg}}{{{radiant['el_deg']:.2f}}}\n",
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
    altitude_fig = plot_altitude_summary(fit, tx0, rot, params, samples)
    cartopy_fig = plot_cartopy_report_figure(fit, geom, tx0, rot, params, samples)
    stats = write_trajectory_stats(fit, tx0, rot, params)
    print(measurement_fig)
    print(altitude_fig)
    print(cartopy_fig)
    print(stats)


if __name__ == "__main__":
    main()
