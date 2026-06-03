#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from station_config import short_station_name

from range_doppler_mf_bolide import (
    detrend_window,
    get_codes,
    interpolate_rect,
    read_channel_files,
    rf_file_for_time,
)


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
C = 299792458.0


def load_station_yaml(data_dir, station):
    cfg_path = Path(data_dir) / f"analysis-config-millstone_{station}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    system = cfg["System"]
    coded = cfg["Decoding"]["coded-cw"]
    return {
        "config_path": str(cfg_path),
        "station": station,
        "name_rx": system["name_rx"],
        "name_tx": system["name_tx"],
        "gps_rx": np.array(system["gps_rx"], dtype=float),
        "gps_tx": np.array(system["gps_tx"], dtype=float),
        "antenna_pos_rx": np.array(system["antenna_pos_rx"], dtype=float),
        "antenna_pos_tx": np.array(system["antenna_pos_tx"], dtype=float),
        "frequency_hz": float(system["frequency"]),
        "seeds": list(coded["seeds"]),
        "code_len": int(coded["n_code"]),
        "range_resolution_m": float(system.get("range_resolution", 3000.0)),
    }


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def enu_to_ecef_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.column_stack((east, north, up))


def tx_antenna_ecef(site):
    origin = geodetic_to_ecef(*site["gps_tx"])
    rot = enu_to_ecef_matrix(site["gps_tx"][0], site["gps_tx"][1])
    return origin[None, :] + site["antenna_pos_tx"] @ rot.T


def unit_vectors_enu(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    east, north = np.broadcast_arrays(np.sin(az) * np.cos(el), np.cos(az) * np.cos(el))
    up = np.broadcast_to(np.sin(el), east.shape)
    return np.stack((east, north, up), axis=-1)


def regular_direction_grid(grid_size=121, max_zenith_angle=90.0):
    """
    Regular direction-cosine grid like the PANSY interferometry u/v search.

    Returns ENU unit vectors with east/north sampled uniformly in horizontal
    direction cosine and up derived from the unit sphere. Points outside the
    requested zenith-angle cap are removed.
    """
    umax = np.sin(np.deg2rad(max_zenith_angle))
    u = np.linspace(-umax, umax, int(grid_size), dtype=np.float64)
    east, north = np.meshgrid(u, u)
    h2 = east * east + north * north
    keep = h2 <= umax * umax
    up = np.sqrt(np.maximum(0.0, 1.0 - h2))
    unit = np.column_stack((east[keep], north[keep], up[keep]))
    az = np.rad2deg(np.arctan2(unit[:, 0], unit[:, 1]))
    az = (az + 360.0) % 360.0
    el = np.rad2deg(np.arcsin(unit[:, 2]))
    return unit.astype(np.float32), az.astype(np.float32), el.astype(np.float32)


def read_pick_from_rdmf(fname, snr_threshold=0.0):
    with h5py.File(fname, "r") as h:
        P = h["P"][()]
        N = h["N"][()]
        D = h["D"][()]
        t_sample = h["t_sample"][()]
        sr = float(h["sample_rate"][()])
        t0_sample = float(h["t0_sample"][()]) if "t0_sample" in h else float(t_sample[0])
        station = h["station"][()].decode()
        if "range_km" in h:
            range_km = h["range_km"][()]
        else:
            range_km = h["range_idx"][()] * 3.0
        snr = P / np.maximum(N, 1e-12)
        snr_search = snr.copy()
        if snr_threshold > 0:
            snr_search[snr_search < snr_threshold] = np.nan
        ti, ri = np.unravel_index(np.nanargmax(snr_search), snr.shape)
        return {
            "station": station,
            "time_second": float((t_sample[ti] - t0_sample) / sr),
            "unix_second": float(t_sample[ti] / sr),
            "range_km": float(range_km[ri]),
            "doppler_hz": float(D[ti, ri]),
            "snr": float(snr[ti, ri]),
        }


def complex_tx_response(site, args, time_second, range_km, doppler_hz):
    code_len_i = site["code_len"] * args.range_interp
    sample_rate_i = None
    win_len = args.n_coh * code_len_i
    range_gate_km = site["range_resolution_m"] / args.range_interp / 1000.0
    range_idx = int(round(range_km / range_gate_km))
    read_len_i = win_len + range_idx + 1
    read_seconds = read_len_i / (100000.0 * args.range_interp) + 0.05

    codes = get_codes(site["seeds"], site["code_len"], args.n_coh, range_interp=args.range_interp)
    responses = []
    station_dir = Path(args.data_dir) / site["station"]
    sample_rate = None

    for ch in args.channels:
        files = rf_file_for_time(station_dir, ch, args.iso_hour)
        z, sr, _ = read_channel_files(files, start_second=time_second, seconds=read_seconds)
        if sample_rate is None:
            sample_rate = sr
            sample_rate_i = sr * args.range_interp
        z = interpolate_rect(z, args.range_interp)
        if len(z) < read_len_i:
            raise ValueError("Not enough samples for requested time/range/coherent length")

        z0 = detrend_window(z[:read_len_i], spike_sigma=args.spike_sigma)
        segment = z0[range_idx : range_idx + win_len]
        fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
        fi = int(np.argmin(np.abs(fvec - doppler_hz)))
        y = np.zeros(len(codes), dtype=np.complex64)
        for ci, code in enumerate(codes):
            spec = np.fft.fftshift(np.fft.fft(segment * np.conj(code)))
            y[ci] = spec[fi]
        responses.append(y)

    return {
        "responses": np.array(responses),
        "range_idx": range_idx,
        "range_km": range_idx * range_gate_km,
        "doppler_bin_hz": float(fvec[fi]),
        "sample_rate": sample_rate,
    }


def estimate_aod(site, responses, az_grid, el_grid, phase_sign=1.0):
    ant = site["antenna_pos_tx"]
    wavelength = C / site["frequency_hz"]
    k = 2.0 * np.pi / wavelength
    score = np.zeros((len(el_grid), len(az_grid)), dtype=np.float64)

    # Normalize each channel vector so channels with more power do not dominate solely by gain.
    ych = responses.copy()
    norms = np.sqrt(np.sum(np.abs(ych) ** 2, axis=1))
    good = norms > 0
    ych[good, :] /= norms[good, None]

    for ei, el in enumerate(el_grid):
        u = unit_vectors_enu(az_grid, el)
        phase = phase_sign * k * (u @ ant.T)
        steering = np.exp(1j * phase)
        for y in ych:
            score[ei, :] += np.abs(np.sum(steering.conj() * y[None, :], axis=1)) ** 2

    score /= max(1, responses.shape[0])
    score /= responses.shape[1] ** 2
    ei, ai = np.unravel_index(np.argmax(score), score.shape)
    return {
        "score": score,
        "az_deg": float(az_grid[ai]),
        "el_deg": float(el_grid[ei]),
        "unit_enu": unit_vectors_enu(np.array([az_grid[ai]]), np.array([el_grid[ei]]))[0],
        "peak_score": float(score[ei, ai]),
    }


def write_result(ofname, site, pick, response_info, estimate, az_grid, el_grid):
    ofname = Path(ofname)
    ofname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(ofname, "w") as h:
        h["responses"] = response_info["responses"]
        h["az_grid_deg"] = az_grid
        h["el_grid_deg"] = el_grid
        h["score"] = estimate["score"]
        h["best_az_deg"] = estimate["az_deg"]
        h["best_el_deg"] = estimate["el_deg"]
        h["best_unit_enu"] = estimate["unit_enu"]
        h["peak_score"] = estimate["peak_score"]
        h["range_idx"] = response_info["range_idx"]
        h["range_km"] = response_info["range_km"]
        h["doppler_bin_hz"] = response_info["doppler_bin_hz"]
        h["frequency_hz"] = site["frequency_hz"]
        h["gps_tx"] = site["gps_tx"]
        h["gps_rx"] = site["gps_rx"]
        h["antenna_pos_tx_enu_m"] = site["antenna_pos_tx"]
        h["antenna_pos_tx_ecef_m"] = tx_antenna_ecef(site)
        h["seeds"] = np.array(site["seeds"], dtype=np.int64)
        h.attrs["station"] = site["station"]
        h.attrs["config_path"] = site["config_path"]
        for key, val in pick.items():
            h.attrs[key] = val


def plot_score(ofname, az_grid, el_grid, score, title):
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    pcm = ax.pcolormesh(az_grid, el_grid, score, shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Normalized beam score")
    ax.set_xlabel("Azimuth from north (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(title)
    fig.savefig(ofname, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Estimate transmit angle of departure from coded-CW TX antenna phases.")
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--station", default=None)
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0])
    p.add_argument("--time-second", type=float)
    p.add_argument("--range-km", type=float)
    p.add_argument("--doppler-hz", type=float)
    p.add_argument("--rdmf-file")
    p.add_argument("--pick-max", action="store_true")
    p.add_argument("--pick-snr-threshold", type=float, default=0.0)
    p.add_argument("--n-coh", type=int, default=8)
    p.add_argument("--range-interp", type=int, default=2)
    p.add_argument("--spike-sigma", type=float, default=0.0)
    p.add_argument("--az-step", type=float, default=1.0)
    p.add_argument("--el-step", type=float, default=1.0)
    p.add_argument("--el-min", type=float, default=0.0)
    p.add_argument("--el-max", type=float, default=90.0)
    p.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    p.add_argument("--out", default="products/aod/aod_estimate.h5")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    pick = {}
    if args.rdmf_file:
        pick = read_pick_from_rdmf(args.rdmf_file, snr_threshold=args.pick_snr_threshold)
        if args.station is None:
            args.station = pick["station"]
        if args.pick_max:
            args.time_second = pick["time_second"]
            args.range_km = pick["range_km"]
            args.doppler_hz = pick["doppler_hz"]

    if args.station is None:
        raise ValueError("Provide --station or --rdmf-file")
    if args.time_second is None or args.range_km is None or args.doppler_hz is None:
        raise ValueError("Provide --time-second, --range-km, and --doppler-hz, or use --rdmf-file --pick-max")

    site = load_station_yaml(args.data_dir, args.station)
    response_info = complex_tx_response(site, args, args.time_second, args.range_km, args.doppler_hz)
    az_grid = np.arange(0.0, 360.0 + 0.5 * args.az_step, args.az_step)
    el_grid = np.arange(args.el_min, args.el_max + 0.5 * args.el_step, args.el_step)
    estimate = estimate_aod(site, response_info["responses"], az_grid, el_grid, phase_sign=args.phase_sign)

    pick.update({
        "time_second": float(args.time_second),
        "range_km": float(args.range_km),
        "doppler_hz": float(args.doppler_hz),
    })
    write_result(args.out, site, pick, response_info, estimate, az_grid, el_grid)

    print(f"station={site['station']} tx={site['name_tx']} seeds={site['seeds']}")
    print(f"time_second={args.time_second:.6f} range_km={response_info['range_km']:.3f} doppler_bin_hz={response_info['doppler_bin_hz']:.2f}")
    print(f"best_aod_az={estimate['az_deg']:.2f} deg best_aod_el={estimate['el_deg']:.2f} deg score={estimate['peak_score']:.3f}")
    print(f"unit_enu={estimate['unit_enu']}")
    print(f"Wrote {args.out}")

    if args.plot:
        png = str(Path(args.out).with_suffix("")) + "_score.png"
        title = f"{short_station_name(site['station'])} AoD score, t={args.time_second:.3f}s r={response_info['range_km']:.1f}km"
        plot_score(png, az_grid, el_grid, estimate["score"], title)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
