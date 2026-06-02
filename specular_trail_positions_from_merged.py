#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import numpy as np
from scipy.spatial import ConvexHull

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml, regular_direction_grid
from aod_position_from_merged import (
    aod_delay_position,
    enu_series_to_geodetic,
    position_uncertainty_mc,
    station_label,
    trajectory_covariance_samples,
    trajectory_position_enu_from_params,
    tx_angle_sigma_rad,
)
from constrained_aod_from_rdmf import estimate_one, trajectory_position_enu
from estimate_head_echo_trajectory import ecef_to_geodetic_wgs84
from range_doppler_mf_bolide import detrend_window, get_codes, interpolate_rect, read_channel_files, rf_file_for_time


DEFAULT_MERGED_DIR = "products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_TRAJECTORY = "products/head_echo_trajectory_roi_fit/head_echo_trajectory.json"
DEFAULT_OUT_DIR = "products/specular_trail_positions"
DEFAULT_ANNOTATIONS = "annotations/trail_echo_rois.json"
DEFAULT_REPORT_FIG = "reports/figures/head_echo_trajectory_trail_heatmap.png"
DEFAULT_ALIAS_BOUNDS = (-70.2, -69.8, 42.1, 42.4)
RANGE_WINDOWS_KM = {
    "beverly": (180.0, 275.0),
    "clinton": (220.0, 280.0),
    "pomfret": (350.0, 420.0),
    "siena": (400.0, 500.0),
    "amherstnh": (200.0, 350.0),
}


def in_alias_cluster(row, bounds=DEFAULT_ALIAS_BOUNDS):
    lon_min, lon_max, lat_min, lat_max = bounds
    return lon_min <= float(row["lon_deg"]) <= lon_max and lat_min <= float(row["lat_deg"]) <= lat_max


def load_rdmf(path):
    h = h5py.File(path, "r")
    sr = float(h["sample_rate"][()])
    t0_sample = float(h["t0_sample"][()])
    start_second = float(h.attrs.get("start_second", 0.0))
    range_gate_width_m = float(h["range_resolution_m"][()]) / float(h["range_interp"][()])
    return {
        "handle": h,
        "station": h["station"][()].decode(),
        "time_unix": h["t_sample"][()] / sr,
        "raw_second": start_second + (h["t_sample"][()] - t0_sample) / sr,
        "range_km": h["range_km"][()],
        "P_rti": h["P_rti"][()],
        "D_rti": h["D_rti"][()],
        "N": h["N"][()],
        "range_gate_width_m": range_gate_width_m,
        "n_coh": int(h["n_coh"][()]),
        "range_interp": int(h["range_interp"][()]),
        "channels": h["channels"][()].astype(int).tolist(),
    }


def select_specular_picks(rdmf, min_snr_db, max_doppler_hz, min_time_spacing_s, max_picks, range_window_km=None):
    snr_db = 10.0 * np.log10(np.maximum(rdmf["P_rti"] / np.maximum(rdmf["N"], 1e-12), 1e-12))
    mask = (np.abs(rdmf["D_rti"]) < max_doppler_hz) & (snr_db > min_snr_db)
    if range_window_km is not None:
        r0, r1 = range_window_km
        keep_r = (rdmf["range_km"] >= min(r0, r1)) & (rdmf["range_km"] <= max(r0, r1))
        mask &= keep_r[None, :]
    candidates = []
    for ti in np.where(np.any(mask, axis=1))[0]:
        row = np.where(mask[ti], snr_db[ti], -np.inf)
        ri = int(np.argmax(row))
        candidates.append(
            {
                "time_index": int(ti),
                "range_index": int(ri),
                "unix_time": float(rdmf["time_unix"][ti]),
                "raw_second": float(rdmf["raw_second"][ti]),
                "range_km": float(rdmf["range_km"][ri]),
                "doppler_hz": float(rdmf["D_rti"][ti, ri]),
                "snr_db": float(snr_db[ti, ri]),
            }
        )
    candidates.sort(key=lambda x: x["snr_db"], reverse=True)
    picked = []
    for cand in candidates:
        if all(abs(cand["unix_time"] - p["unix_time"]) >= min_time_spacing_s for p in picked):
            picked.append(cand)
        if len(picked) >= max_picks:
            break
    picked.sort(key=lambda x: x["unix_time"])
    return picked


def select_trail_roi_picks(rdmf, roi, min_snr_db, max_doppler_hz, min_time_spacing_s, max_picks):
    if not roi or not roi.get("has_trail_echo", False):
        return []
    roi_selection = roi.get("selection", {})
    threshold = max(float(min_snr_db), float(roi_selection.get("snr_threshold_db", min_snr_db)))
    max_abs_doppler = min(float(max_doppler_hz), float(roi_selection.get("max_abs_doppler_hz", max_doppler_hz)))
    t0 = float(roi["time_range"]["start_unix"])
    t1 = float(roi["time_range"]["end_unix"])
    r0 = float(roi["range_km"]["min"])
    r1 = float(roi["range_km"]["max"])
    snr_db = 10.0 * np.log10(np.maximum(rdmf["P_rti"] / np.maximum(rdmf["N"], 1e-12), 1e-12))
    mask = (
        (rdmf["time_unix"][:, None] >= t0)
        & (rdmf["time_unix"][:, None] <= t1)
        & (rdmf["range_km"][None, :] >= min(r0, r1))
        & (rdmf["range_km"][None, :] <= max(r0, r1))
        & (np.abs(rdmf["D_rti"]) <= max_abs_doppler)
        & (snr_db >= threshold)
    )
    candidates = []
    for ti in np.where(np.any(mask, axis=1))[0]:
        row = np.where(mask[ti], snr_db[ti], -np.inf)
        ri = int(np.argmax(row))
        candidates.append(
            {
                "time_index": int(ti),
                "range_index": int(ri),
                "unix_time": float(rdmf["time_unix"][ti]),
                "raw_second": float(rdmf["raw_second"][ti]),
                "range_km": float(rdmf["range_km"][ri]),
                "doppler_hz": float(rdmf["D_rti"][ti, ri]),
                "snr_db": float(snr_db[ti, ri]),
            }
        )
    candidates.sort(key=lambda x: x["snr_db"], reverse=True)
    picked = []
    for cand in candidates:
        if all(abs(cand["unix_time"] - p["unix_time"]) >= min_time_spacing_s for p in picked):
            picked.append(cand)
        if len(picked) >= max_picks:
            break
    picked.sort(key=lambda x: x["unix_time"])
    return picked


def raw_complex_tx_response(site, data_dir, iso_hour, channels, raw_second, range_km, doppler_hz, n_coh, range_interp, spike_sigma):
    code_len_i = site["code_len"] * range_interp
    sample_rate_i = None
    win_len = n_coh * code_len_i
    gate_km = site["range_resolution_m"] / range_interp / 1000.0
    range_idx = int(round(range_km / gate_km))
    read_len_i = win_len + range_idx + 1
    read_seconds = read_len_i / (100000.0 * range_interp) + 0.05
    codes = get_codes(site["seeds"], site["code_len"], n_coh, range_interp=range_interp)
    responses = []
    station_dir = Path(data_dir) / site["station"]
    for ch in channels:
        files = rf_file_for_time(station_dir, ch, iso_hour)
        z, sr, _ = read_channel_files(files, start_second=raw_second, seconds=read_seconds)
        if sample_rate_i is None:
            sample_rate_i = sr * range_interp
        z = interpolate_rect(z, range_interp)
        z0 = detrend_window(z[:read_len_i], spike_sigma=spike_sigma)
        segment = z0[range_idx : range_idx + win_len]
        fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
        fi = int(np.argmin(np.abs(fvec - doppler_hz)))
        y = np.zeros(len(codes), dtype=np.complex64)
        for ci, code in enumerate(codes):
            spec = np.fft.fftshift(np.fft.fft(segment * np.conj(code)))
            y[ci] = spec[fi]
        responses.append(y)
    return np.array(responses), float(fvec[fi]), range_idx


def prior_from_trajectory_delay(fit, site, tx_ecef, rot, pick, n_grid=1200):
    t_obs = float(pick["unix_time"])
    t0 = float(fit["t_ref_unix"]) - 2.5
    t1 = t_obs
    if t1 <= t0:
        t1 = float(fit["t_ref_unix"]) + 2.5
    ts = np.linspace(t0, t1, n_grid)
    pos_enu = np.array([trajectory_position_enu(fit, t) for t in ts])
    pos_ecef = tx_ecef[None, :] + pos_enu @ rot.T
    rx_ecef = geodetic_to_ecef(*site["gps_rx"])
    bistatic = np.linalg.norm(pos_ecef - tx_ecef[None, :], axis=1) + np.linalg.norm(pos_ecef - rx_ecef[None, :], axis=1)
    im = int(np.argmin(np.abs(bistatic - pick["range_km"] * 1000.0)))
    prior_vec = pos_enu[im]
    return prior_vec / np.linalg.norm(prior_vec), float(ts[im]), float(bistatic[im] - pick["range_km"] * 1000.0)


def sample_hull(lonlat_samples):
    points = np.reshape(np.asarray(lonlat_samples, dtype=float), (-1, 2))
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 4:
        return None
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center[None, :], axis=1)
    keep = radius <= np.percentile(radius, 95.0)
    points = points[keep]
    if len(points) < 4:
        return None
    hull = ConvexHull(points)
    poly = points[hull.vertices]
    return np.vstack((poly, poly[0]))


def trail_uncertainty_lonlat_samples(rows, tx_ecef, rot, n_per_position=80):
    rng = np.random.default_rng(20260530)
    enu_samples = []
    for row in rows:
        unc = row.get("uncertainty") or {}
        cov = unc.get("position_cov_enu_m2")
        mean = np.asarray(row["position_enu_m"], dtype=float)
        if cov is None:
            enu_samples.append(mean[None, :])
            continue
        cov = np.asarray(cov, dtype=float)
        try:
            samples = rng.multivariate_normal(mean, cov, size=n_per_position, check_valid="ignore")
        except Exception:
            samples = mean[None, :]
        enu_samples.append(samples)
    if not enu_samples:
        return np.empty((0, 2), dtype=float)
    enu_samples = np.vstack(enu_samples)
    _ecef, llh = enu_series_to_geodetic(tx_ecef, rot, enu_samples)
    return llh[:, [1, 0]]


def annotate_trajectory_altitudes(ax, lon, lat, alt_km, transform_kw):
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
    if len(alt_unique) < 2:
        return
    offsets = [(-18, -24), (-10, -24), (0, -24), (10, -24), (18, -24), (26, -24), (34, -24)]
    for level, (dx, dy) in zip(levels, offsets, strict=True):
        if level < alt_unique[0] or level > alt_unique[-1]:
            continue
        x = float(np.interp(level, alt_unique, lon_unique))
        y = float(np.interp(level, alt_unique, lat_unique))
        annotate_kwargs = dict(
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="0.1",
            zorder=11,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.45, shrinkA=0, shrinkB=1.5),
        )
        if transform_kw.get("transform") is not None:
            annotate_kwargs["xycoords"] = transform_kw["transform"]._as_mpl_transform(ax)
        ax.annotate(f"{level:.0f} km", **annotate_kwargs)


def plot_positions(rows, fit, tx_ecef, rot, sites, out_png, uncertainty_samples):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        data_crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=(9.2, 7.2), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection=data_crs)
    except Exception:
        cfeature = None
        data_crs = None
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 7.2), constrained_layout=True)
    transform_kw = {"transform": data_crs} if data_crs is not None else {}
    stations = sorted(set(r["station"] for r in rows))

    obs_t = np.array([o["unix_time"] for o in fit["observations"]], dtype=float)
    t_first = float(np.min(obs_t))
    t_last = float(np.max(obs_t))
    t = np.linspace(t_first - 10.0, t_last + 3.0, 420)
    tri_enu = np.array([trajectory_position_enu(fit, ti) for ti in t])
    _, tri_llh = enu_series_to_geodetic(tx_ecef, rot, tri_enu)
    tri_lon = tri_llh[:, 1]
    tri_lat = tri_llh[:, 0]
    tri_alt_km = tri_llh[:, 2] / 1000.0
    pre = t < t_first
    detect = (t >= t_first) & (t <= t_last)
    post = t > t_last

    sample_llh = []
    for params in uncertainty_samples:
        sample_enu = trajectory_position_enu_from_params(params, fit, t)
        _, llh = enu_series_to_geodetic(tx_ecef, rot, sample_enu)
        sample_llh.append(llh)
    sample_lonlat = np.array([llh[:, [1, 0]] for llh in sample_llh]) if sample_llh else np.empty((0, 0, 2))
    hull_poly = sample_hull(sample_lonlat) if sample_lonlat.size else None

    trail_lonlat = trail_uncertainty_lonlat_samples(rows, tx_ecef, rot)
    echo_lons = [r["lon_deg"] for r in rows] + trail_lonlat[:, 0].tolist() + tri_lon.tolist()
    echo_lats = [r["lat_deg"] for r in rows] + trail_lonlat[:, 1].tolist() + tri_lat.tolist()
    if hull_poly is not None:
        echo_lons.extend(hull_poly[:, 0].tolist())
        echo_lats.extend(hull_poly[:, 1].tolist())
    first_site = sites[stations[0]]
    tx_lat, tx_lon, _ = first_site["gps_tx"]
    context_lons = list(echo_lons) + [tx_lon]
    context_lats = list(echo_lats) + [tx_lat]
    for station in stations:
        rx_lat, rx_lon, _ = sites[station]["gps_rx"]
        context_lons.append(rx_lon)
        context_lats.append(rx_lat)

    cape_cod_extent = [-71.8, -69.2, 41.2, 42.9]
    pad_lon = max(0.15, 0.2 * (np.nanmax(context_lons) - np.nanmin(context_lons)))
    pad_lat = max(0.15, 0.2 * (np.nanmax(context_lats) - np.nanmin(context_lats)))
    extent = [
        min(cape_cod_extent[0], np.nanmin(context_lons) - pad_lon),
        max(cape_cod_extent[1], np.nanmax(context_lons) + pad_lon),
        min(cape_cod_extent[2], np.nanmin(context_lats) - pad_lat),
        max(cape_cod_extent[3], np.nanmax(context_lats) + pad_lat),
    ]
    if data_crs is not None and hasattr(ax, "set_extent"):
        ax.set_extent(extent, crs=data_crs)
        ax.set_facecolor("0.96")
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.94", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.86", edgecolor="none", zorder=0)
        ax.coastlines(resolution="10m", linewidth=0.9, color="0.15", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.35", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.7", alpha=0.8)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

    if hull_poly is not None:
        ax.fill(
            hull_poly[:, 0],
            hull_poly[:, 1],
            color="C0",
            alpha=0.16,
            label="sampled 95% trajectory covariance",
            zorder=2,
            **transform_kw,
        )

    if len(trail_lonlat) > 0:
        h, lon_edges, lat_edges = np.histogram2d(
            trail_lonlat[:, 0],
            trail_lonlat[:, 1],
            bins=(180, 160),
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
        )
        h = np.ma.masked_where(h <= 0, h)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            h.T,
            cmap=cmap,
            norm=LogNorm(vmin=1.0, vmax=float(np.nanmax(h))),
            alpha=0.62,
            zorder=4,
            **transform_kw,
        )
        fig.colorbar(mesh, ax=ax, label="Trail echo position density")

    ax.plot(tri_lon[pre], tri_lat[pre], "--", color="C1", lw=2.2, label="-10 s extrapolation", zorder=5, **transform_kw)
    ax.plot(tri_lon[detect], tri_lat[detect], "-", color="C3", lw=2.7, label="fit over detections", zorder=6, **transform_kw)
    ax.plot(tri_lon[post], tri_lat[post], "--", color="C2", lw=2.2, label="+3 s extrapolation", zorder=5, **transform_kw)
    ax.scatter(
        [r["lon_deg"] for r in rows],
        [r["lat_deg"] for r in rows],
        s=10,
        color="white",
        edgecolor="0.1",
        linewidth=0.25,
        alpha=0.65,
        label="AoD + range position solves",
        zorder=8,
        **transform_kw,
    )
    annotate_trajectory_altitudes(ax, tri_lon, tri_lat, tri_alt_km, transform_kw)
    ax.scatter(tx_lon, tx_lat, marker="*", s=110, color="gold", edgecolor="0.2", label="transmitter", zorder=9, **transform_kw)
    receiver_labeled = False
    for station in stations:
        rx_lat, rx_lon, _ = sites[station]["gps_rx"]
        ax.scatter(rx_lon, rx_lat, marker="^", s=58, color="0.1", label=None if receiver_labeled else "receivers", zorder=9, **transform_kw)
        receiver_labeled = True
        ax.text(rx_lon + 0.025, rx_lat + 0.015, station_label(station), fontsize=8, zorder=10, clip_on=True, **transform_kw)
    ax.text(tx_lon + 0.025, tx_lat + 0.015, "TX", fontsize=8, weight="bold", zorder=10, clip_on=True, **transform_kw)
    ax.set_title("Head-echo trajectory with non-specular trail AoD positions")
    ax.legend(loc="best", fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Estimate non-specular trail positions from saved near-zero-Doppler ROIs.")
    p.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--stations", nargs="+", default=["amherstnh", "beverly", "clinton", "pomfret", "siena"])
    p.add_argument("--min-snr-db", type=float, default=8.0)
    p.add_argument("--max-doppler-hz", type=float, default=100.0)
    p.add_argument("--use-report-range-windows", action="store_true", default=True)
    p.add_argument("--no-report-range-windows", action="store_false", dest="use_report_range_windows")
    p.add_argument("--min-time-spacing-s", type=float, default=0.08)
    p.add_argument("--max-picks-per-station", type=int, default=45)
    p.add_argument("--cone-deg", type=float, default=25.0)
    p.add_argument("--grid-size", type=int, default=361)
    p.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=-1.0)
    p.add_argument("--spike-sigma", type=float, default=0.0)
    p.add_argument("--aod-phase-error-deg", type=float, default=15.0)
    p.add_argument("--position-uncertainty-mc", type=int, default=200)
    p.add_argument("--trajectory-uncertainty-samples", type=int, default=200)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--report-fig", default=DEFAULT_REPORT_FIG)
    p.add_argument("--exclude-alias-cluster", action="store_true", default=True)
    p.add_argument("--no-exclude-alias-cluster", action="store_false", dest="exclude_alias_cluster")
    args = p.parse_args()

    fit = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    rois = annotations.get("stations", {})
    sites = {station: load_station_yaml(args.data_dir, station) for station in args.stations}
    tx_ecef = geodetic_to_ecef(*sites[args.stations[0]]["gps_tx"])
    rot = enu_to_ecef_matrix(sites[args.stations[0]]["gps_tx"][0], sites[args.stations[0]]["gps_tx"][1])
    dir_unit, dir_az, dir_el = regular_direction_grid(args.grid_size, 90.0)
    rng = np.random.default_rng(20260530)

    rows = []
    skipped = 0
    for station in args.stations:
        path = Path(args.merged_dir) / f"rdmf_peak_{station}_merged.h5"
        rdmf = load_rdmf(path)
        try:
            if station in rois:
                picks = select_trail_roi_picks(
                    rdmf,
                    rois[station],
                    args.min_snr_db,
                    args.max_doppler_hz,
                    args.min_time_spacing_s,
                    args.max_picks_per_station,
                )
                roi = rois[station]
                range_text = f"{roi['range_km']['min']:.1f}-{roi['range_km']['max']:.1f} km ROI"
            else:
                range_window = RANGE_WINDOWS_KM.get(station) if args.use_report_range_windows else None
                picks = select_specular_picks(
                    rdmf,
                    args.min_snr_db,
                    args.max_doppler_hz,
                    args.min_time_spacing_s,
                    args.max_picks_per_station,
                    range_window_km=range_window,
                )
                range_text = "all ranges" if range_window is None else f"{range_window[0]:.0f}-{range_window[1]:.0f} km"
            print(f"{station_label(station)}: selected {len(picks)} near-zero-Doppler trail picks in {range_text}", flush=True)
            for pi, pick in enumerate(picks, start=1):
                prior_unit, prior_t, prior_delay_resid = prior_from_trajectory_delay(fit, sites[station], tx_ecef, rot, pick)
                try:
                    voltage, doppler_bin_hz, raw_range_idx = raw_complex_tx_response(
                        sites[station],
                        args.data_dir,
                        args.iso_hour,
                        rdmf["channels"],
                        pick["raw_second"],
                        pick["range_km"],
                        pick["doppler_hz"],
                        rdmf["n_coh"],
                        rdmf["range_interp"],
                        args.spike_sigma,
                    )
                except Exception:
                    skipped += 1
                    continue
                aod = estimate_one(voltage, sites[station], dir_unit, dir_az, dir_el, prior_unit, args.cone_deg, args.phase_sign)
                rx_ecef = geodetic_to_ecef(*sites[station]["gps_rx"])
                solved = aod_delay_position(tx_ecef, rx_ecef, aod["best_unit_enu"], rot, pick["range_km"] * 1000.0)
                if solved is None:
                    skipped += 1
                    continue
                pos_ecef, tx_range_m, delay_residual_m = solved
                pos_enu = rot.T @ (pos_ecef - tx_ecef)
                lat, lon, alt_m = ecef_to_geodetic_wgs84(pos_ecef)
                angle_sigma_rad = tx_angle_sigma_rad(sites[station], args.aod_phase_error_deg)
                unc = position_uncertainty_mc(tx_ecef, rx_ecef, rot, aod["best_unit_enu"], pick["range_km"] * 1000.0, rdmf["range_gate_width_m"], angle_sigma_rad, args.position_uncertainty_mc, rng)
                if unc is not None:
                    unc["aod_phase_error_deg"] = float(args.aod_phase_error_deg)
                rows.append(
                    {
                        "station": station,
                        "unix_time": pick["unix_time"],
                        "raw_second": pick["raw_second"],
                        "range_km": pick["range_km"],
                        "doppler_hz": pick["doppler_hz"],
                        "doppler_bin_hz": doppler_bin_hz,
                        "snr_db": pick["snr_db"],
                        "time_index": pick["time_index"],
                        "range_index": pick["range_index"],
                        "raw_range_index": raw_range_idx,
                        "prior_traj_time_unix": prior_t,
                        "prior_delay_residual_m": prior_delay_resid,
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
                        "uncertainty": unc,
                    }
                )
        finally:
            rdmf["handle"].close()

    if not rows:
        raise SystemExit("No non-specular trail positions solved")

    excluded_alias_rows = []
    if args.exclude_alias_cluster:
        kept_rows = []
        for row in rows:
            if in_alias_cluster(row):
                excluded_alias_rows.append(row)
            else:
                kept_rows.append(row)
        rows = kept_rows
        print(f"Excluded {len(excluded_alias_rows)} AoD alias-cluster trail positions north of Cape Cod", flush=True)

    if not rows:
        raise SystemExit("No non-specular trail positions left after alias filtering")

    by_station = {}
    for station in sorted(set(r["station"] for r in rows)):
        rr = [r for r in rows if r["station"] == station]
        by_station[station] = {
            "n": len(rr),
            "median_snr_db": float(np.median([r["snr_db"] for r in rr])),
            "median_abs_doppler_hz": float(np.median([abs(r["doppler_hz"]) for r in rr])),
            "median_aod_prior_offset_deg": float(np.median([r["aod_prior_offset_deg"] for r in rr])),
            "median_alt_km": float(np.median([r["alt_m"] / 1000.0 for r in rr])),
            "median_position_sigma_3d_m": float(np.median([r["uncertainty"]["position_sigma_3d_m"] for r in rr if r.get("uncertainty")])),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "specular_trail_position_summary.json"
    out_png = out_dir / "specular_trail_positions_map.png"
    payload = {
        "schema": "new_england_bolide.specular_trail_positions.v1",
        "merged_dir": args.merged_dir,
        "trajectory_json": args.trajectory_json,
        "annotations": args.annotations,
        "selection": {
            "min_snr_db": args.min_snr_db,
            "max_doppler_hz": args.max_doppler_hz,
            "roi_annotation_schema": annotations.get("schema"),
            "exclude_alias_cluster": args.exclude_alias_cluster,
            "alias_cluster_bounds_lonlat": list(DEFAULT_ALIAS_BOUNDS),
            "n_excluded_alias_cluster": len(excluded_alias_rows),
            "use_report_range_windows": args.use_report_range_windows,
            "range_windows_km": {k: list(v) for k, v in RANGE_WINDOWS_KM.items()},
            "min_time_spacing_s": args.min_time_spacing_s,
            "max_picks_per_station": args.max_picks_per_station,
        },
        "aod_phase_error_deg": args.aod_phase_error_deg,
        "n_used": len(rows),
        "n_skipped": skipped,
        "excluded_alias_cluster_positions": excluded_alias_rows,
        "summary_by_station": by_station,
        "positions": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    traj_samples = trajectory_covariance_samples(fit, args.trajectory_uncertainty_samples)
    plot_positions(rows, fit, tx_ecef, rot, sites, out_png, traj_samples)
    if args.report_fig:
        plot_positions(rows, fit, tx_ecef, rot, sites, args.report_fig, traj_samples)

    print(f"used={len(rows)} skipped={skipped} |f|<{args.max_doppler_hz:g}Hz snr>{args.min_snr_db:g}dB", flush=True)
    for station, stats in by_station.items():
        print(
            f"{station_label(station)}: n={stats['n']} median_alt={stats['median_alt_km']:.1f}km "
            f"median_abs_dop={stats['median_abs_doppler_hz']:.1f}Hz "
            f"median_aod_offset={stats['median_aod_prior_offset_deg']:.2f}deg "
            f"median_sigma_3d={stats['median_position_sigma_3d_m']/1000.0:.2f}km",
            flush=True,
        )
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_png}", flush=True)
    if args.report_fig:
        print(f"Wrote {args.report_fig}", flush=True)


if __name__ == "__main__":
    main()
