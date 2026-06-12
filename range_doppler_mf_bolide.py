#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

C = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def parse_time_second(value, iso_hour):
    """Parse seconds from the hour for numeric, HH:MM:SS, or ISO timestamps."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass

    hour0 = datetime.fromisoformat(str(iso_hour).replace("Z", "+00:00"))
    if hour0.tzinfo is None:
        hour0 = hour0.replace(tzinfo=timezone.utc)
    if "T" in text:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=hour0.tzinfo)
    else:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"Cannot parse time {value!r}; use seconds, HH:MM:SS, or ISO timestamp")
        hh, mm, ss = int(parts[0]), int(parts[1]), float(parts[2])
        t = hour0.replace(hour=hh, minute=mm, second=int(ss), microsecond=int(round((ss % 1.0) * 1e6)))
    return (t - hour0).total_seconds()


def apply_t0_t1_args(args):
    t0 = parse_time_second(args.t0, args.iso_hour)
    t1 = parse_time_second(args.t1, args.iso_hour)
    if t0 is not None:
        args.start_second = float(t0)
    if t1 is not None:
        args.seconds = float(t1) - float(args.start_second)
        if args.seconds <= 0:
            raise ValueError(f"t1 must be after t0/start_second; got start={args.start_second:g}, t1={t1:g}")


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * np.cos(lon)
    y = (n + alt_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_to_ecef_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return np.column_stack((east, north, up))


def bpsk(seed, length):
    np.random.seed(seed)
    code = np.random.random(length)
    code = np.exp(2.0 * np.pi * 1.0j * code)
    code = np.angle(code)
    code = -1.0 * np.sign(code)
    return np.array(code, dtype=np.complex64)


def interpolate_rect(z, interp):
    if interp == 1:
        return z
    z_rep = np.repeat(z, interp)
    wf = np.repeat(1.0 / float(interp), interp)
    zi = np.fft.ifft(np.fft.fft(z_rep) * np.fft.fft(wf, len(z_rep)))
    zi = np.roll(zi, -int(interp / 2))
    return zi.astype(np.complex64)


def get_codes(seeds, code_len, n_coh, range_interp=1):
    codes = []
    for seed in seeds:
        code = bpsk(seed, code_len)
        code = interpolate_rect(code, range_interp)
        code = code / np.max(np.abs(code))
        codes.append(np.tile(code, n_coh).astype(np.complex64))
    return codes


def read_config(data_dir, station):
    cfg_path = Path(data_dir) / f"analysis-config-millstone_{station}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dec = cfg["Decoding"]["coded-cw"]
    system = cfg["System"]
    return {
        "config_path": str(cfg_path),
        "seeds": list(dec["seeds"]),
        "code_len": int(dec["n_code"]),
        "rmin_idx": int(dec["rmin_idx"]),
        "rmax_idx": int(dec["rmax_idx"]),
        "frequency_hz": float(system["frequency"]),
        "gps_rx": system["gps_rx"],
        "gps_tx": system["gps_tx"],
        "antenna_pos_tx": np.array(system["antenna_pos_tx"], dtype=np.float64),
        "name_rx": system["name_rx"],
        "name_tx": system["name_tx"],
    }


def load_trajectory(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = payload["model"]
    return {key: np.asarray(model[key], dtype=np.float64) for key in ("time_unix", "lat_deg", "lon_deg", "alt_m")}


def target_ecef(traj, time_unix):
    time_unix = np.asarray(time_unix, dtype=np.float64)
    lat = np.interp(time_unix, traj["time_unix"], traj["lat_deg"])
    lon = np.interp(time_unix, traj["time_unix"], traj["lon_deg"])
    alt = np.interp(time_unix, traj["time_unix"], traj["alt_m"])
    return np.stack([geodetic_to_ecef(a, b, c) for a, b, c in zip(lat, lon, alt, strict=True)], axis=0)


def tx_aod_steering_vector(config, traj, time_unix, phase_sign=-1.0):
    tx_ecef = geodetic_to_ecef(*config["gps_tx"])
    rot = enu_to_ecef_matrix(config["gps_tx"][0], config["gps_tx"][1])
    vec_enu = (target_ecef(traj, np.array([time_unix], dtype=np.float64))[0] - tx_ecef) @ rot
    unit = vec_enu / max(float(np.linalg.norm(vec_enu)), 1e-30)
    wavelength = C / float(config["frequency_hz"])
    phase = float(phase_sign) * (2.0 * np.pi / wavelength) * (
        unit @ np.asarray(config["antenna_pos_tx"], dtype=np.float64).T
    )
    return np.exp(1.0j * phase).astype(np.complex64), unit.astype(np.float32)


def trajectory_range_index_bounds(config, traj, time_unix, range_resolution_m, range_interp, pad_pixels):
    times = np.asarray(time_unix, dtype=np.float64)
    target = target_ecef(traj, times)
    tx_ecef = geodetic_to_ecef(*config["gps_tx"])
    rx_ecef = geodetic_to_ecef(*config["gps_rx"])
    range_m = np.linalg.norm(target - tx_ecef[None, :], axis=1) + np.linalg.norm(target - rx_ecef[None, :], axis=1)
    gate_m = float(range_resolution_m) / float(range_interp)
    idx = range_m / gate_m
    rmin = int(np.floor(np.nanmin(idx))) - int(pad_pixels)
    rmax = int(np.ceil(np.nanmax(idx))) + int(pad_pixels)
    return max(0, rmin), max(0, rmax)


def combine_matched_filter_images(code_images, good_head_dop, steering=None, polarization_combine=False):
    """
    Combine channel/code matched-filter voltage images.

    code_images has shape (rx_channel, tx_code, range, doppler). With TX
    steering enabled, the code dimension is coherently steered using the
    trajectory-implied transmit-array phase. The receive-polarization vector is
    estimated once from the peak pixel in the AoD-steered, incoherent-pol image.
    """
    n_channels, n_codes = code_images.shape[:2]
    tx_norm = float(n_codes)
    pol_gamma = np.full(n_channels, np.nan + 1.0j * np.nan, dtype=np.complex64)
    peak_range_idx = np.int32(-1)
    peak_doppler_idx = np.int32(-1)

    if steering is None:
        if polarization_combine:
            raise ValueError("polarization_combine requires tx_aod_beamform so TX voltages have been coherently steered")
        power = np.sum(np.abs(code_images) ** 2, axis=(0, 1)).astype(np.float32)
        return power, pol_gamma, peak_range_idx, peak_doppler_idx

    if len(steering) != n_codes:
        raise ValueError(f"steering vector has {len(steering)} TX channels, expected {n_codes}")

    tx_beam = np.einsum("ckrf,k->crf", code_images, np.conj(steering), optimize=True)
    selection_power = (np.sum(np.abs(tx_beam) ** 2, axis=0) / tx_norm).astype(np.float32)
    if not polarization_combine:
        return selection_power, pol_gamma, peak_range_idx, peak_doppler_idx

    search = selection_power.copy()
    search[:, ~good_head_dop] = -np.inf
    peak_range_idx, peak_doppler_idx = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_vec = tx_beam[:, peak_range_idx, peak_doppler_idx]
    norm = float(np.linalg.norm(peak_vec))
    if norm <= 0.0 or not np.isfinite(norm):
        power = selection_power
    else:
        pol_gamma = (peak_vec / norm).astype(np.complex64)
        combined = np.einsum("c,crf->rf", np.conj(pol_gamma), tx_beam, optimize=True)
        power = (np.abs(combined) ** 2 / tx_norm).astype(np.float32)
    return power, pol_gamma, np.int32(peak_range_idx), np.int32(peak_doppler_idx)


def save_rdmf_frame(out_dir, station, ti, unix_time, range_km, fvec, power, vmin_db=None, vmax_db=None):
    out_dir = Path(out_dir) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    img_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    if vmin_db is None:
        finite = img_db[np.isfinite(img_db)]
        vmin_db = float(np.nanpercentile(finite, 5.0)) if finite.size else 0.0
    if vmax_db is None:
        finite = img_db[np.isfinite(img_db)]
        vmax_db = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    mesh = ax.pcolormesh(fvec, range_km, img_db, shading="auto", cmap="inferno", vmin=vmin_db, vmax=vmax_db)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Matched-filter power (dB)")
    ax.set_xlabel("Doppler frequency (Hz)")
    ax.set_ylabel("Bistatic group range (km)")
    ax.set_title(f"{station} RDMF step {ti:04d}, unix {unix_time:.3f}")
    fig.savefig(out_dir / f"{station}_rdmf_{ti:04d}_{unix_time:.3f}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def rf_file_for_time(station_dir, channel, iso_hour):
    files = sorted((Path(station_dir) / f"ch{channel:03d}" / iso_hour).glob("rf@*.h5"))
    if not files:
        raise FileNotFoundError(f"No RF files under {station_dir}/ch{channel:03d}/{iso_hour}")
    return files


def estimate_complex_dc(z):
    return np.complex64(float(np.mean(z.real)) + 1.0j * float(np.mean(z.imag)))


def read_channel_files(files, start_second=0.0, seconds=None, *, dc_subtract="mean"):
    chunks = []
    sample_rate = None
    abs_index0 = None

    for fn in files:
        with h5py.File(fn, "r") as h:
            ds = h["rf_data"]
            sr = float(ds.attrs["sample_rate_numerator"]) / float(ds.attrs["sample_rate_denominator"])
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise ValueError(f"Sample-rate mismatch in {fn}: {sr} != {sample_rate}")
            if abs_index0 is None:
                abs_index0 = int(h["rf_data_index"][0, 0])
            chunks.append(ds[:, 0].astype(np.complex64))

    z = np.concatenate(chunks)
    i0 = int(round(start_second * sample_rate))
    if seconds is None:
        i1 = len(z)
    else:
        i1 = min(len(z), i0 + int(round(seconds * sample_rate)))
    if i0 >= len(z) or i1 <= i0:
        raise ValueError("Requested time span is outside the available RF data")
    out = z[i0:i1]
    if dc_subtract == "mean":
        out = out - estimate_complex_dc(out)
    elif dc_subtract != "none":
        raise ValueError(f"Unknown dc_subtract method {dc_subtract!r}")
    return out.astype(np.complex64, copy=False), sample_rate, abs_index0 + i0


def detrend_window(z, spike_sigma=0.0):
    # Remove complex mean DC and optionally remove spikes.
    z = z - estimate_complex_dc(z)
    if spike_sigma and spike_sigma > 0:
        med = np.median(np.abs(z))
        if med > 0:
            z = z.copy()
            z[np.abs(z) > spike_sigma * med] = 0.0
    return z


def estimate_range_gate_dc(channel_data, ridx, sample_rate_i, seconds):
    if seconds <= 0:
        return None
    n_cal = int(round(seconds * sample_rate_i))
    if n_cal <= 0:
        return None
    offsets = np.zeros((len(channel_data), len(ridx)), dtype=np.complex64)
    for chi, z in enumerate(channel_data):
        max_stop = int(ridx[-1]) + n_cal
        if max_stop > len(z):
            raise ValueError("Not enough data to estimate range-gate DC offsets")
        cs = np.concatenate(([0.0j], np.cumsum(z[:max_stop], dtype=np.complex128)))
        offsets[chi, :] = ((cs[ridx + n_cal] - cs[ridx]) / float(n_cal)).astype(np.complex64)
    return offsets


def range_doppler_bank(
    channel_data,
    sample_rate,
    abs_index0,
    codes,
    code_len,
    rmin_idx,
    rmax_idx,
    range_resolution_m,
    range_interp,
    n_coh,
    codes_per_step,
    fmin,
    fmax,
    ignore_freq,
    head_min_doppler_hz=250.0,
    dc_cancel=False,
    dc_cancel_seconds=5.0,
    dc_calibration_data=None,
    spike_sigma=0.0,
    doppler_whiten=False,
    doppler_zero_pad_factor=4,
    tx_aod_beamform=False,
    tx_aod_trajectory=None,
    tx_aod_config=None,
    tx_aod_phase_sign=-1.0,
    polarization_combine=False,
    rdmf_frame_dir=None,
    rdmf_frame_station=None,
    rdmf_frame_vmin_db=None,
    rdmf_frame_vmax_db=None,
    rdmf_frame_every=1,
    progress_every=1,
    checkpoint_every=0,
    checkpoint_func=None,
):
    code_len_i = code_len * range_interp
    sample_rate_i = sample_rate * range_interp
    win_len = n_coh * code_len_i
    step_len = codes_per_step * code_len_i
    original_step_len = codes_per_step * code_len
    read_len = win_len + rmax_idx + 1
    n_steps = 1 + (min(map(len, channel_data)) - read_len) // step_len
    if n_steps <= 0:
        raise ValueError("Requested coherent window/range span is longer than the data")

    ridx = np.arange(rmin_idx, rmax_idx + 1, dtype=np.int64)
    fftlen = int(doppler_zero_pad_factor) * win_len
    if fftlen < win_len:
        raise ValueError("doppler_zero_pad_factor must be at least 1")
    fvec = np.fft.fftshift(np.fft.fftfreq(fftlen, d=1.0 / sample_rate_i))
    fidx = np.where((fvec >= fmin) & (fvec <= fmax))[0]
    fvec0 = fvec[fidx]
    good_rti_dop = np.abs(fvec0) >= ignore_freq
    good_head_dop = np.abs(fvec0) >= head_min_doppler_hz
    if not np.any(good_head_dop):
        raise ValueError(
            f"No Doppler bins satisfy head_min_doppler_hz={head_min_doppler_hz:g} "
            f"within [{fmin:g}, {fmax:g}] Hz"
        )

    P_rti = np.zeros((n_steps, len(ridx)), dtype=np.float32)
    D_rti = np.zeros_like(P_rti)
    P_head = np.zeros_like(P_rti)
    D_head = np.zeros_like(P_rti)
    N = np.zeros_like(P_rti)
    pol_gamma = np.full((n_steps, len(channel_data)), np.nan + 1.0j * np.nan, dtype=np.complex64)
    pol_peak_range_idx = np.full(n_steps, -1, dtype=np.int32)
    pol_peak_doppler_idx = np.full(n_steps, -1, dtype=np.int32)
    tx_aod_unit_enu = np.full((n_steps, 3), np.nan, dtype=np.float32)
    t_sample = abs_index0 + np.arange(n_steps, dtype=np.float64) * original_step_len
    t_start = time.time()
    dc_source = channel_data if dc_calibration_data is None else dc_calibration_data
    dc_offsets = estimate_range_gate_dc(dc_source, ridx, sample_rate_i, dc_cancel_seconds) if dc_cancel else None
    if dc_offsets is not None:
        print(f"subtracting per-range DC offsets estimated from {dc_cancel_seconds:g}s", flush=True)
    if polarization_combine and not tx_aod_beamform:
        raise ValueError("polarization_combine currently requires tx_aod_beamform")
    if tx_aod_beamform and (tx_aod_trajectory is None or tx_aod_config is None):
        raise ValueError("tx_aod_beamform requires tx_aod_trajectory and tx_aod_config")

    # for all time steps, compute the matched filter for all codes and ranges,
    # store matched filter power, best Doppler bin, and noise floor estimate. Optionally apply Doppler whitening.
    for ti in range(n_steps):
        start = ti * step_len
        code_images = np.zeros((len(channel_data), len(codes), len(ridx), len(fidx)), dtype=np.complex64)
        for chi, z in enumerate(channel_data):
            z0 = detrend_window(z[start : start + read_len], spike_sigma=spike_sigma)
            windows = np.lib.stride_tricks.sliding_window_view(z0, win_len)[ridx, :]
            if dc_offsets is not None:
                windows = windows - dc_offsets[chi, :, None]
            for code_i, code in enumerate(codes):
                Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), n=fftlen, axis=1), axes=1)
                code_images[chi, code_i, :, :] = Z[:, fidx]

        steering = None
        if tx_aod_beamform:
            unix_time = float(t_sample[ti] / sample_rate)
            steering, unit_enu = tx_aod_steering_vector(
                tx_aod_config,
                tx_aod_trajectory,
                unix_time,
                phase_sign=tx_aod_phase_sign,
            )
            tx_aod_unit_enu[ti, :] = unit_enu
        S, gamma, peak_ri, peak_di = combine_matched_filter_images(
            code_images,
            good_head_dop=good_head_dop,
            steering=steering,
            polarization_combine=polarization_combine,
        )
        pol_gamma[ti, :] = gamma
        pol_peak_range_idx[ti] = peak_ri
        pol_peak_doppler_idx[ti] = peak_di
        if rdmf_frame_dir is not None and rdmf_frame_every > 0 and ti % int(rdmf_frame_every) == 0:
            save_rdmf_frame(
                rdmf_frame_dir,
                rdmf_frame_station or "station",
                ti,
                float(t_sample[ti] / sample_rate),
                (ridx * (range_resolution_m / range_interp) / 1000.0).astype(np.float32),
                fvec0.astype(np.float32),
                S,
                vmin_db=rdmf_frame_vmin_db,
                vmax_db=rdmf_frame_vmax_db,
            )

        if doppler_whiten:
            # Normalize each Doppler bin by its median power over range before
            # peak picking. This suppresses Doppler-dependent clutter/leakage,
            # so a detection means "bright relative to other ranges at this
            # Doppler" rather than simply "bright in absolute power." In this
            # bolide data, the head echo is often not observable without this
            # whitening step because strong background structure dominates the
            # absolute-power peak picker.
            dop_floor = np.median(S, axis=0)
            S_pick = S / np.maximum(dop_floor[None, :], 1e-12)
            floor = np.median(S_pick, axis=1)
        else:
            # Use absolute matched-filter power. This preserves raw power
            # differences but lets strong Doppler-dependent clutter dominate
            # the peak picker.
            S_pick = S
            floor = np.median(S_pick, axis=1)

        # RTI products keep the strongest bin, including non-specular trail
        # echoes. Head products use a stricter Doppler gate for dynamic
        # head-echo measurements.
        S_rti_search = S_pick.copy()
        if np.any(~good_rti_dop):
            S_rti_search[:, ~good_rti_dop] = -np.inf
        imax_rti = np.argmax(S_rti_search, axis=1)
        P_rti[ti, :] = S_pick[np.arange(len(ridx)), imax_rti]
        D_rti[ti, :] = fvec0[imax_rti]

        S_head_search = S_pick.copy()
        S_head_search[:, ~good_head_dop] = -np.inf
        imax_head = np.argmax(S_head_search, axis=1)
        P_head[ti, :] = S_pick[np.arange(len(ridx)), imax_head]
        D_head[ti, :] = fvec0[imax_head]
        N[ti, :] = floor

        snr = P_head[ti, :] / np.maximum(N[ti, :], 1e-12)
        best = int(np.argmax(snr))
        n_done = ti + 1
        if progress_every > 0 and (n_done == 1 or n_done == n_steps or n_done % progress_every == 0):
            elapsed = time.time() - t_start
            steps_per_sec = n_done / max(elapsed, 1e-9)
            eta = (n_steps - n_done) / max(steps_per_sec, 1e-9)
            print(
                f"PROGRESS {n_done}/{n_steps} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s rate={steps_per_sec:.3f}steps/s "
                f"t={((abs_index0 + ti * original_step_len) / sample_rate):.3f} "
                f"range_idx={ridx[best]} head_dop={D_head[ti, best]:.1f}Hz snr={snr[best]:.2f}",
                flush=True,
            )

        if checkpoint_func and checkpoint_every > 0 and (n_done == n_steps or n_done % checkpoint_every == 0):
            checkpoint_func(
                {
                    "P_rti": P_rti[:n_done, :],
                    "D_rti": D_rti[:n_done, :],
                    "P_head": P_head[:n_done, :],
                    "D_head": D_head[:n_done, :],
                    "P": P_head[:n_done, :],
                    "D": D_head[:n_done, :],
                    "N": N[:n_done, :],
                    "pol_gamma": pol_gamma[:n_done, :],
                    "pol_peak_range_idx": pol_peak_range_idx[:n_done],
                    "pol_peak_doppler_idx": pol_peak_doppler_idx[:n_done],
                    "tx_aod_unit_enu": tx_aod_unit_enu[:n_done, :],
                    "t_sample": t_sample[:n_done],
                    "range_idx": ridx.astype(np.int32),
                    "range_km": (ridx * (range_resolution_m / range_interp) / 1000.0).astype(np.float32),
                    "fvec": fvec0.astype(np.float32),
                },
                n_done,
                n_steps,
            )

    return {
        "P_rti": P_rti,
        "D_rti": D_rti,
        "P_head": P_head,
        "D_head": D_head,
        "P": P_head,
        "D": D_head,
        "N": N,
        "pol_gamma": pol_gamma,
        "pol_peak_range_idx": pol_peak_range_idx,
        "pol_peak_doppler_idx": pol_peak_doppler_idx,
        "tx_aod_unit_enu": tx_aod_unit_enu,
        "t_sample": t_sample,
        "range_idx": ridx.astype(np.int32),
        "range_km": (ridx * (range_resolution_m / range_interp) / 1000.0).astype(np.float32),
        "fvec": fvec0.astype(np.float32),
        "doppler_zero_pad_factor": np.array(int(doppler_zero_pad_factor), dtype=np.int32),
    }


def write_output(ofname, result, config, args, sample_rate, abs_index0):
    ofname = Path(ofname)
    ofname.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ofname.with_suffix(ofname.suffix + ".tmp")
    with h5py.File(tmp_name, "w") as h:
        for key, val in result.items():
            h[key] = val

        def write_once(key, val):
            if key not in h:
                h[key] = val

        write_once("sample_rate", sample_rate)
        write_once("t0_sample", abs_index0)
        write_once("station", np.bytes_(args.station))
        write_once("channels", np.array(args.channels, dtype=np.int32))
        write_once("seeds", np.array(config["seeds"], dtype=np.int64))
        write_once("code_len", config["code_len"])
        write_once("n_coh", args.n_coh)
        write_once("codes_per_step", args.codes_per_step)
        write_once("range_interp", args.range_interp)
        write_once("doppler_zero_pad_factor", args.doppler_zero_pad_factor)
        write_once("range_resolution_m", args.range_resolution_m)
        write_once("ignore_freq", args.ignore_freq)
        write_once("head_min_doppler_hz", args.head_min_doppler_hz)
        write_once("dc_cancel", int(args.dc_cancel))
        write_once("dc_cancel_seconds", args.dc_cancel_seconds)
        write_once("dc_cancel_start_second", args.dc_cancel_start_second)
        write_once("doppler_whiten", int(args.doppler_whiten))
        write_once("tx_aod_beamform", int(args.tx_aod_beamform))
        write_once("tx_aod_phase_sign", float(args.tx_aod_phase_sign))
        write_once("polarization_combine", int(args.polarization_combine))
        write_once("frequency_hz", config["frequency_hz"])
        write_once("gps_rx", np.array(config["gps_rx"], dtype=np.float64))
        write_once("gps_tx", np.array(config["gps_tx"], dtype=np.float64))
        write_once("antenna_pos_tx", np.array(config["antenna_pos_tx"], dtype=np.float64))
        h.attrs["config_path"] = config["config_path"]
        h.attrs["iso_hour"] = args.iso_hour
        h.attrs["name_rx"] = config["name_rx"]
        h.attrs["name_tx"] = config["name_tx"]
        h.attrs["tx_aod_trajectory_json"] = str(args.tx_aod_trajectory_json)
        h.attrs["polarization_method"] = "peak_pixel_unit_vector" if args.polarization_combine else "incoherent"
        if hasattr(args, "n_done"):
            h.attrs["n_done"] = int(args.n_done)
        if hasattr(args, "n_steps"):
            h.attrs["n_steps"] = int(args.n_steps)
    tmp_name.replace(ofname)


def main():
    p = argparse.ArgumentParser(description="Simple range-Doppler matched filter bank for the Cape Cod Bay bolide data.")
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--station", default="pomfret")
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--start-second", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--t0", default=None, help="Start time as seconds from iso-hour, HH:MM:SS, or ISO timestamp.")
    p.add_argument("--t1", default=None, help="Stop time as seconds from iso-hour, HH:MM:SS, or ISO timestamp.")
    p.add_argument("--n-coh", type=int, default=8)
    p.add_argument("--codes-per-step", type=int, default=8)
    p.add_argument("--range-interp", type=int, default=1)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=4)
    p.add_argument("--range-resolution-m", type=float, default=3000.0)
    p.add_argument("--max-range-km", type=float)
    p.add_argument("--rmin-idx", type=int)
    p.add_argument("--rmax-idx", type=int)
    p.add_argument(
        "--trajectory-range-json",
        default=None,
        help="Trajectory JSON used to restrict evaluated range gates around the predicted bistatic range.",
    )
    p.add_argument("--trajectory-range-pad-pixels", type=int, default=None)
    p.add_argument("--fmin", type=float, default=-40000.0)
    p.add_argument("--fmax", type=float, default=40000.0)
    p.add_argument("--ignore-freq", type=float, default=150.0)
    p.add_argument("--head-min-doppler-hz", type=float, default=250.0)
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--dc-cancel-seconds", type=float, default=5.0)
    p.add_argument("--dc-cancel-start-second", type=float, default=75.0)
    p.add_argument("--spike-sigma", type=float, default=0.0)
    p.add_argument("--doppler-whiten", action="store_true", default=False)
    p.add_argument("--no-doppler-whiten", action="store_false", dest="doppler_whiten")
    p.add_argument("--tx-aod-beamform", action="store_true")
    p.add_argument(
        "--tx-aod-trajectory-json",
        default="products/ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5/ballistic_head_trail_correlated_fit.json",
    )
    p.add_argument("--tx-aod-phase-sign", type=float, default=-1.0)
    p.add_argument("--polarization-combine", action="store_true")
    p.add_argument("--save-rdmf-frames", action="store_true", help="Write one range-frequency RDMF PNG for each processed time step.")
    p.add_argument("--rdmf-frame-dir", type=Path, default=None, help="Directory for per-time-step RDMF PNGs.")
    p.add_argument("--rdmf-frame-vmin-db", type=float, default=None)
    p.add_argument("--rdmf-frame-vmax-db", type=float, default=None)
    p.add_argument("--rdmf-frame-every", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    apply_t0_t1_args(args)

    config = read_config(args.data_dir, args.station)
    rmin_idx = config["rmin_idx"] if args.rmin_idx is None else args.rmin_idx
    rmax_idx = config["rmax_idx"] if args.rmax_idx is None else args.rmax_idx
    if args.max_range_km is not None:
        range_gate_km = args.range_resolution_m / args.range_interp / 1000.0
        rmax_idx = int(round(args.max_range_km / range_gate_km))

    channel_data = []
    sample_rate = None
    abs_index0 = None
    station_dir = Path(args.data_dir) / args.station
    for ch in args.channels:
        files = rf_file_for_time(station_dir, ch, args.iso_hour)
        z, sr, idx0 = read_channel_files(files, start_second=args.start_second, seconds=args.seconds)
        z = interpolate_rect(z, args.range_interp)
        channel_data.append(z)
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0

    if args.trajectory_range_pad_pixels is not None:
        traj_path = args.trajectory_range_json or args.tx_aod_trajectory_json
        range_traj = load_trajectory(traj_path)
        seconds = float(args.seconds) if args.seconds is not None else max(0.0, (min(map(len, channel_data)) / (sample_rate * args.range_interp)))
        t_start = float(abs_index0) / float(sample_rate)
        t_stop = t_start + seconds
        t_grid = np.linspace(t_start, t_stop, max(2, int(np.ceil(seconds / 0.01)) + 1))
        r0, r1 = trajectory_range_index_bounds(
            config,
            range_traj,
            t_grid,
            args.range_resolution_m,
            args.range_interp,
            args.trajectory_range_pad_pixels,
        )
        rmin_idx = max(rmin_idx, r0)
        rmax_idx = min(rmax_idx, r1)
        print(
            f"trajectory range gate window idx={rmin_idx}:{rmax_idx} "
            f"pad={args.trajectory_range_pad_pixels} trajectory={traj_path}",
            flush=True,
        )
        if rmax_idx < rmin_idx:
            raise ValueError("Trajectory range gate window is empty")

    codes = get_codes(config["seeds"], config["code_len"], args.n_coh, range_interp=args.range_interp)
    tx_aod_trajectory = load_trajectory(args.tx_aod_trajectory_json) if args.tx_aod_beamform else None
    ofname = args.out
    if ofname is None:
        ofname = f"rdmf_{args.station}_{args.iso_hour}_{int(args.start_second):04d}s.h5"
    rdmf_frame_dir = None
    if args.save_rdmf_frames:
        rdmf_frame_dir = args.rdmf_frame_dir
        if rdmf_frame_dir is None:
            rdmf_frame_dir = Path(ofname).with_suffix("").parent / f"{Path(ofname).with_suffix('').name}_frames"

    def checkpoint(result_part, n_done, n_steps):
        args.n_done = n_done
        args.n_steps = n_steps
        write_output(ofname, result_part, config, args, sample_rate, abs_index0)
        print(f"CHECKPOINT {ofname} {n_done}/{n_steps}", flush=True)

    dc_calibration_data = None
    if args.dc_cancel:
        sample_rate_i = sample_rate * args.range_interp
        cal_seconds = args.dc_cancel_seconds + (float(rmax_idx) + 1.0) / sample_rate_i
        dc_calibration_data = []
        for ch in args.channels:
            files = rf_file_for_time(station_dir, ch, args.iso_hour)
            z, _sr, _idx0 = read_channel_files(
                files,
                start_second=args.dc_cancel_start_second,
                seconds=cal_seconds,
            )
            dc_calibration_data.append(interpolate_rect(z, args.range_interp))

    result = range_doppler_bank(
        channel_data=channel_data,
        sample_rate=sample_rate,
        abs_index0=abs_index0,
        codes=codes,
        code_len=config["code_len"],
        rmin_idx=rmin_idx,
        rmax_idx=rmax_idx,
        range_resolution_m=args.range_resolution_m,
        range_interp=args.range_interp,
        n_coh=args.n_coh,
        codes_per_step=args.codes_per_step,
        doppler_zero_pad_factor=args.doppler_zero_pad_factor,
        tx_aod_beamform=args.tx_aod_beamform,
        tx_aod_trajectory=tx_aod_trajectory,
        tx_aod_config=config,
        tx_aod_phase_sign=args.tx_aod_phase_sign,
        polarization_combine=args.polarization_combine,
        rdmf_frame_dir=rdmf_frame_dir,
        rdmf_frame_station=args.station,
        rdmf_frame_vmin_db=args.rdmf_frame_vmin_db,
        rdmf_frame_vmax_db=args.rdmf_frame_vmax_db,
        rdmf_frame_every=args.rdmf_frame_every,
        fmin=args.fmin,
        fmax=args.fmax,
        ignore_freq=args.ignore_freq,
        head_min_doppler_hz=args.head_min_doppler_hz,
        dc_cancel=args.dc_cancel,
        dc_cancel_seconds=args.dc_cancel_seconds,
        dc_calibration_data=dc_calibration_data,
        spike_sigma=args.spike_sigma,
        doppler_whiten=args.doppler_whiten,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_func=checkpoint,
    )

    args.n_done = result["P"].shape[0]
    args.n_steps = result["P"].shape[0]
    write_output(ofname, result, config, args, sample_rate, abs_index0)
    print(f"Wrote {ofname}")


if __name__ == "__main__":
    main()
