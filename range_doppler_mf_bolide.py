#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


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
        "name_rx": system["name_rx"],
        "name_tx": system["name_tx"],
    }


def rf_file_for_time(station_dir, channel, iso_hour):
    files = sorted((Path(station_dir) / f"ch{channel:03d}" / iso_hour).glob("rf@*.h5"))
    if not files:
        raise FileNotFoundError(f"No RF files under {station_dir}/ch{channel:03d}/{iso_hour}")
    return files


def read_channel_files(files, start_second=0.0, seconds=None):
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
    return z[i0:i1], sample_rate, abs_index0 + i0


def detrend_window(z, spike_sigma=0.0):
    # remove median and optionally remove spikes
    z = z - np.median(z)
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
    fftlen = win_len
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
    t_sample = abs_index0 + np.arange(n_steps, dtype=np.float64) * original_step_len
    t_start = time.time()
    dc_source = channel_data if dc_calibration_data is None else dc_calibration_data
    dc_offsets = estimate_range_gate_dc(dc_source, ridx, sample_rate_i, dc_cancel_seconds) if dc_cancel else None
    if dc_offsets is not None:
        print(f"subtracting per-range DC offsets estimated from {dc_cancel_seconds:g}s", flush=True)

    # for all time steps, compute the matched filter for all codes and ranges,
    # store matched filter power, best Doppler bin, and noise floor estimate. Optionally apply Doppler whitening.
    for ti in range(n_steps):
        start = ti * step_len
        S = np.zeros((len(ridx), len(fidx)), dtype=np.float32)
        for chi, z in enumerate(channel_data):
            z0 = detrend_window(z[start : start + read_len], spike_sigma=spike_sigma)
            windows = np.lib.stride_tricks.sliding_window_view(z0, win_len)[ridx, :]
            if dc_offsets is not None:
                windows = windows - dc_offsets[chi, :, None]
            for code in codes:
                Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)
                S += np.abs(Z[:, fidx]) ** 2

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
        "t_sample": t_sample,
        "range_idx": ridx.astype(np.int32),
        "range_km": (ridx * (range_resolution_m / range_interp) / 1000.0).astype(np.float32),
        "fvec": fvec0.astype(np.float32),
    }


def write_output(ofname, result, config, args, sample_rate, abs_index0):
    ofname = Path(ofname)
    ofname.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ofname.with_suffix(ofname.suffix + ".tmp")
    with h5py.File(tmp_name, "w") as h:
        for key, val in result.items():
            h[key] = val
        h["sample_rate"] = sample_rate
        h["t0_sample"] = abs_index0
        h["station"] = np.bytes_(args.station)
        h["channels"] = np.array(args.channels, dtype=np.int32)
        h["seeds"] = np.array(config["seeds"], dtype=np.int64)
        h["code_len"] = config["code_len"]
        h["n_coh"] = args.n_coh
        h["codes_per_step"] = args.codes_per_step
        h["range_interp"] = args.range_interp
        h["range_resolution_m"] = args.range_resolution_m
        h["ignore_freq"] = args.ignore_freq
        h["head_min_doppler_hz"] = args.head_min_doppler_hz
        h["dc_cancel"] = int(args.dc_cancel)
        h["dc_cancel_seconds"] = args.dc_cancel_seconds
        h["dc_cancel_start_second"] = args.dc_cancel_start_second
        h["doppler_whiten"] = int(args.doppler_whiten)
        h["frequency_hz"] = config["frequency_hz"]
        h["gps_rx"] = np.array(config["gps_rx"], dtype=np.float64)
        h["gps_tx"] = np.array(config["gps_tx"], dtype=np.float64)
        h.attrs["config_path"] = config["config_path"]
        h.attrs["iso_hour"] = args.iso_hour
        h.attrs["name_rx"] = config["name_rx"]
        h.attrs["name_tx"] = config["name_tx"]
        if hasattr(args, "n_done"):
            h.attrs["n_done"] = int(args.n_done)
        if hasattr(args, "n_steps"):
            h.attrs["n_steps"] = int(args.n_steps)
    tmp_name.replace(ofname)


def main():
    p = argparse.ArgumentParser(description="Simple range-Doppler matched filter bank for the New England bolide data.")
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--station", default="pomfret")
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--start-second", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--n-coh", type=int, default=8)
    p.add_argument("--codes-per-step", type=int, default=8)
    p.add_argument("--range-interp", type=int, default=1)
    p.add_argument("--range-resolution-m", type=float, default=3000.0)
    p.add_argument("--max-range-km", type=float)
    p.add_argument("--rmin-idx", type=int)
    p.add_argument("--rmax-idx", type=int)
    p.add_argument("--fmin", type=float, default=-40000.0)
    p.add_argument("--fmax", type=float, default=40000.0)
    p.add_argument("--ignore-freq", type=float, default=150.0)
    p.add_argument("--head-min-doppler-hz", type=float, default=250.0)
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--dc-cancel-seconds", type=float, default=5.0)
    p.add_argument("--dc-cancel-start-second", type=float, default=75.0)
    p.add_argument("--spike-sigma", type=float, default=0.0)
    p.add_argument("--doppler-whiten", action="store_true", default=True)
    p.add_argument("--no-doppler-whiten", action="store_false", dest="doppler_whiten")
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--out", default=None)
    args = p.parse_args()

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

    codes = get_codes(config["seeds"], config["code_len"], args.n_coh, range_interp=args.range_interp)
    ofname = args.out
    if ofname is None:
        ofname = f"rdmf_{args.station}_{args.iso_hour}_{int(args.start_second):04d}s.h5"

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
