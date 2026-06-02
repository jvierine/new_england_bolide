#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from angle_departure_estimator import load_station_yaml
from range_doppler_mf_bolide import (
    detrend_window,
    get_codes,
    interpolate_rect,
    read_channel_files,
    rf_file_for_time,
)


DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def estimate_range_gate_dc(channel_data, range_idx, sample_rate_i, seconds):
    if seconds <= 0:
        return None
    n_cal = int(round(seconds * sample_rate_i))
    if n_cal <= 0:
        return None
    offsets = np.zeros((len(channel_data), len(range_idx)), dtype=np.complex64)
    for chi, z in enumerate(channel_data):
        max_stop = int(range_idx[-1]) + n_cal
        if max_stop > len(z):
            raise ValueError("Not enough data to estimate range-gate DC offsets")
        cs = np.concatenate(([0.0j], np.cumsum(z[:max_stop], dtype=np.complex128)))
        offsets[chi, :] = ((cs[range_idx + n_cal] - cs[range_idx]) / float(n_cal)).astype(np.complex64)
    return offsets


def read_dc_calibration_channels(args, site, range_idx, sample_rate_i):
    cal_seconds = args.dc_cancel_seconds + (float(range_idx[-1]) + 1.0) / sample_rate_i
    old_start = args.start_second
    old_seconds = args.seconds
    try:
        args.start_second = args.dc_cancel_start_second
        args.seconds = cal_seconds
        zc, _sample_rate, _abs_index0 = read_station_channels(args, site)
    finally:
        args.start_second = old_start
        args.seconds = old_seconds
    return zc


def read_station_channels(args, site):
    station_dir = Path(args.data_dir) / site["station"]
    zc = []
    sr = None
    idx0 = None
    for ch in args.channels:
        files = rf_file_for_time(station_dir, ch, args.iso_hour)
        z, sr0, idx00 = read_channel_files(files, start_second=args.start_second, seconds=args.seconds)
        zc.append(interpolate_rect(z, args.range_interp))
        sr = sr0 if sr is None else sr
        idx0 = idx00 if idx0 is None else idx0
    return zc, sr, idx0


def station_sample_rate(data_dir, station, channel, iso_hour):
    files = rf_file_for_time(Path(data_dir) / station, channel, iso_hour)
    with h5py.File(files[0], "r") as h:
        ds = h["rf_data"]
        return float(ds.attrs["sample_rate_numerator"]) / float(ds.attrs["sample_rate_denominator"])


def make_output(args, site, n_steps, range_idx, fvec):
    ofile = Path(args.out)
    if ofile.exists() and not args.overwrite:
        raise FileExistsError(f"{ofile} exists; refusing to overwrite")
    ofile.parent.mkdir(parents=True, exist_ok=True)

    n_ch = len(args.channels)
    n_range = len(range_idx)
    n_tx = len(site["seeds"])
    range_km = range_idx * site["range_resolution_m"] / args.range_interp / 1000.0
    chunks_2d = (min(64, n_steps), min(256, n_range))
    chunks_v = (1, n_ch, min(256, n_range), n_tx)

    h = h5py.File(ofile, "w")
    h["station"] = np.bytes_(site["station"])
    h["channels"] = np.array(args.channels, dtype=np.int32)
    h["seeds"] = np.array(site["seeds"], dtype=np.int64)
    h["code_len"] = int(site["code_len"])
    h["frequency_hz"] = float(site["frequency_hz"])
    h["gps_rx"] = np.array(site["gps_rx"], dtype=np.float64)
    h["gps_tx"] = np.array(site["gps_tx"], dtype=np.float64)
    h["range_idx"] = range_idx.astype(np.int32)
    h["range_km"] = range_km.astype(np.float32)
    h["fvec"] = fvec.astype(np.float32)
    h["sample_rate"] = float(args.sample_rate)
    h["t0_sample"] = float(args.abs_index0)
    h["n_coh"] = int(args.n_coh)
    h["codes_per_step"] = int(args.codes_per_step)
    h["range_interp"] = int(args.range_interp)
    h["range_resolution_m"] = float(site["range_resolution_m"])
    h["ignore_freq"] = float(args.ignore_freq)
    h["head_min_doppler_hz"] = float(args.head_min_doppler_hz)
    h["dc_cancel"] = int(args.dc_cancel)
    h["dc_cancel_seconds"] = float(args.dc_cancel_seconds)
    h["dc_cancel_start_second"] = float(args.dc_cancel_start_second)
    h["doppler_whiten"] = int(args.doppler_whiten)
    h.attrs["config_path"] = site["config_path"]
    h.attrs["iso_hour"] = args.iso_hour
    h.attrs["start_second"] = float(args.start_second)
    h.attrs["seconds"] = -1.0 if args.seconds is None else float(args.seconds)
    h.attrs["complete"] = 0
    h.attrs["n_done"] = 0
    h.create_dataset("t_sample", (n_steps,), dtype="f8")
    h.create_dataset("N", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("P_rti", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("D_rti", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("P_head", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("D_head", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    # Backward-compatible aliases for older scripts: P/D are the head-echo
    # picker, while RTI plots should use P_rti/D_rti when present.
    h.create_dataset("P", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("D", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("voltage_peak", (n_steps, n_ch, n_range, n_tx), dtype="c8", chunks=chunks_v)
    return h


def run_station(args):
    site = load_station_yaml(args.data_dir, args.station)
    zc, sample_rate, abs_index0 = read_station_channels(args, site)
    args.sample_rate = sample_rate
    args.abs_index0 = abs_index0

    code_len_i = site["code_len"] * args.range_interp
    sample_rate_i = sample_rate * args.range_interp
    win_len = args.n_coh * code_len_i
    step_len = args.codes_per_step * code_len_i
    original_step_len = args.codes_per_step * site["code_len"]
    gate_km = site["range_resolution_m"] / args.range_interp / 1000.0
    rmin = int(np.ceil(args.range_min_km / gate_km))
    rmax = int(np.floor(args.range_max_km / gate_km))
    range_idx = np.arange(rmin, rmax + 1, dtype=np.int64)
    read_len = win_len + int(range_idx[-1]) + 1
    n_steps = 1 + (min(map(len, zc)) - read_len) // step_len
    if n_steps <= 0:
        raise ValueError("No processing steps available for requested interval/ranges")
    if args.max_steps is not None:
        n_steps = min(n_steps, int(args.max_steps))

    fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    fidx = np.where((fvec >= args.fmin) & (fvec <= args.fmax))[0]
    fvec0 = fvec[fidx]
    good_rti_dop = np.abs(fvec0) >= args.ignore_freq
    good_head_dop = np.abs(fvec0) >= args.head_min_doppler_hz
    if not np.any(good_head_dop):
        raise ValueError(
            f"No Doppler bins satisfy --head-min-doppler-hz={args.head_min_doppler_hz:g} "
            f"within [{args.fmin:g}, {args.fmax:g}] Hz"
        )
    codes = get_codes(site["seeds"], site["code_len"], args.n_coh, range_interp=args.range_interp)
    if args.dc_cancel:
        dc_zc = read_dc_calibration_channels(args, site, range_idx, sample_rate_i)
        dc_offsets = estimate_range_gate_dc(dc_zc, range_idx, sample_rate_i, args.dc_cancel_seconds)
    else:
        dc_offsets = None
    if dc_offsets is not None:
        print(
            f"{args.station}: subtracting per-range DC offsets estimated from "
            f"{args.dc_cancel_seconds:g}s at t={args.dc_cancel_start_second:g}s",
            flush=True,
        )

    h = make_output(args, site, n_steps, range_idx, fvec0)
    t0 = time.time()
    try:
        for ti in range(n_steps):
            start = ti * step_len
            S = np.zeros((len(range_idx), len(fidx)), dtype=np.float32)
            Z_store = np.zeros((len(args.channels), len(range_idx), len(codes), len(fidx)), dtype=np.complex64)
            for chi, z in enumerate(zc):
                z0 = detrend_window(z[start : start + read_len], spike_sigma=args.spike_sigma)
                windows = np.lib.stride_tricks.sliding_window_view(z0, win_len)[range_idx, :]
                if dc_offsets is not None:
                    windows = windows - dc_offsets[chi, :, None]
                for ci, code in enumerate(codes):
                    Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
                    Z_store[chi, :, ci, :] = Z
                    S += np.abs(Z) ** 2

            S /= float(len(args.channels) * len(codes))
            if args.doppler_whiten:
                # Match range_doppler_mf_bolide.py: divide each Doppler bin by
                # its median range power before peak picking. This reduces
                # Doppler-dependent clutter/leakage and makes the picker look
                # for range-localized excess power at each Doppler. In this
                # bolide data, the head echo is often hidden without this step.
                dop_floor = np.median(S, axis=0)
                S_pick = S / np.maximum(dop_floor[None, :], 1e-12)
            else:
                # Use absolute matched-filter power. This is simpler, but
                # strong clutter in a Doppler bin can dominate the selected peak.
                S_pick = S
            noise = np.median(S_pick, axis=1)

            # RTI products keep the strongest bin, including non-specular trail
            # echoes. Head products use a stricter Doppler gate for dynamic
            # head-echo measurements.
            S_rti_search = S_pick.copy()
            if np.any(~good_rti_dop):
                S_rti_search[:, ~good_rti_dop] = -np.inf
            imax_rti = np.argmax(S_rti_search, axis=1)
            peak_rti = S_pick[np.arange(len(range_idx)), imax_rti]
            dop_rti = fvec0[imax_rti]

            S_head_search = S_pick.copy()
            S_head_search[:, ~good_head_dop] = -np.inf
            imax_head = np.argmax(S_head_search, axis=1)
            peak_head = S_pick[np.arange(len(range_idx)), imax_head]
            dop_head = fvec0[imax_head]

            voltage = np.zeros((len(args.channels), len(range_idx), len(codes)), dtype=np.complex64)
            for ri, fi in enumerate(imax_head):
                voltage[:, ri, :] = Z_store[:, ri, :, fi]

            h["t_sample"][ti] = abs_index0 + ti * original_step_len
            h["N"][ti, :] = noise
            h["P_rti"][ti, :] = peak_rti
            h["D_rti"][ti, :] = dop_rti
            h["P_head"][ti, :] = peak_head
            h["D_head"][ti, :] = dop_head
            h["P"][ti, :] = peak_head
            h["D"][ti, :] = dop_head
            h["voltage_peak"][ti, :, :, :] = voltage

            n_done = ti + 1
            h.attrs["n_done"] = n_done
            if n_done % args.flush_every == 0 or n_done == n_steps:
                h.flush()
            if n_done == 1 or n_done == n_steps or n_done % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta = (n_steps - n_done) / max(rate, 1e-9)
                snr = peak_head / np.maximum(noise, 1e-12)
                best = int(np.argmax(snr))
                print(
                    f"PROGRESS {n_done}/{n_steps} elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"rate={rate:.3f}steps/s best_range_km={range_idx[best] * gate_km:.1f} "
                    f"head_dop={dop_head[best]:.1f}Hz "
                    f"head_snr_db={10*np.log10(max(snr[best], 1e-12)):.1f}",
                    flush=True,
                )
        h.attrs["complete"] = 1
    finally:
        h.flush()
        h.close()
    print(f"Wrote {args.out}", flush=True)


def chunk_starts(start_second, total_seconds, chunk_seconds):
    n = int(np.ceil(total_seconds / chunk_seconds))
    for i in range(n):
        t0 = start_second + i * chunk_seconds
        sec = min(chunk_seconds, start_second + total_seconds - t0)
        if sec > 0:
            yield i, t0, sec


def run_batch(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is non-empty; choose a new --out-dir or pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for station in args.stations:
        site = load_station_yaml(args.data_dir, station)
        sr = station_sample_rate(args.data_dir, station, args.channels[0], args.iso_hour)
        code_len_i = site["code_len"] * args.range_interp
        step_len = args.codes_per_step * code_len_i
        original_step_seconds = args.codes_per_step * site["code_len"] / sr
        win_len = args.n_coh * code_len_i
        gate_km = site["range_resolution_m"] / args.range_interp / 1000.0
        rmax = int(np.floor(args.range_max_km / gate_km))
        read_len = win_len + rmax + 1
        total_samples_i = int(round(args.total_seconds * sr)) * args.range_interp
        total_steps = 1 + (total_samples_i - read_len) // step_len
        if total_steps <= 0:
            raise ValueError(f"No processing steps for {station}")
        chunk_steps = args.chunk_steps
        if chunk_steps is None:
            chunk_steps = max(1, int(np.floor(args.chunk_seconds / original_step_seconds)))
        read_extra_seconds = read_len / (sr * args.range_interp)
        for chunk_i, step0 in enumerate(range(0, total_steps, chunk_steps)):
            n_this = min(chunk_steps, total_steps - step0)
            t0 = args.start_second + step0 * original_step_seconds
            sec = n_this * original_step_seconds + read_extra_seconds
            out = out_dir / f"rdmf_peak_{station}_{int(round(t0 * 1000)):09d}ms_{int(round(sec * 1000)):06d}ms.h5"
            cmd = [
                sys.executable, __file__, "station",
                "--data-dir", args.data_dir,
                "--station", station,
                "--iso-hour", args.iso_hour,
                "--channels", *[str(ch) for ch in args.channels],
                "--start-second", f"{t0:.6f}",
                "--seconds", f"{sec:.6f}",
                "--max-steps", str(n_this),
                "--n-coh", str(args.n_coh),
                "--codes-per-step", str(args.codes_per_step),
                "--range-interp", str(args.range_interp),
                "--range-min-km", str(args.range_min_km),
                "--range-max-km", str(args.range_max_km),
                "--fmin", str(args.fmin),
                "--fmax", str(args.fmax),
                "--ignore-freq", str(args.ignore_freq),
                "--head-min-doppler-hz", str(args.head_min_doppler_hz),
                "--dc-cancel-seconds", str(args.dc_cancel_seconds),
                "--dc-cancel-start-second", str(args.dc_cancel_start_second),
                "--spike-sigma", str(args.spike_sigma),
                "--progress-every", str(args.progress_every),
                "--flush-every", str(args.flush_every),
                "--out", str(out),
            ]
            if args.doppler_whiten:
                cmd.append("--doppler-whiten")
            if station in set(args.dc_cancel_stations):
                cmd.append("--dc-cancel")
            if args.overwrite:
                cmd.append("--overwrite")
            tasks.append((station, chunk_i, t0, sec, out, cmd))

    print(f"Queued {len(tasks)} station/chunk jobs with max_parallel={args.max_parallel}", flush=True)
    running = {}
    task_i = 0
    complete = 0
    failed = []
    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    while task_i < len(tasks) or running:
        while task_i < len(tasks) and len(running) < args.max_parallel:
            station, chunk_i, t0, sec, out, cmd = tasks[task_i]
            log = out.with_suffix(".log")
            lf = open(log, "w")
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
            running[proc.pid] = {
                "proc": proc,
                "log": lf,
                "station": station,
                "chunk_i": chunk_i,
                "t0": t0,
                "sec": sec,
                "out": out,
            }
            print(f"START pid={proc.pid} {station} chunk={chunk_i} t0={t0:.3f}s sec={sec:.3f}s out={out}", flush=True)
            task_i += 1

        time.sleep(args.poll_seconds)
        for pid, info in list(running.items()):
            rc = info["proc"].poll()
            if rc is None:
                continue
            info["log"].close()
            del running[pid]
            complete += 1
            elapsed = time.time() - started
            rate = complete / max(elapsed, 1e-9)
            eta = (len(tasks) - complete) / max(rate, 1e-9)
            status = "OK" if rc == 0 else f"FAILED rc={rc}"
            print(
                f"DONE {status} {complete}/{len(tasks)} pid={pid} {info['station']} "
                f"chunk={info['chunk_i']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            if rc != 0:
                failed.append(info)
    if failed:
        print("Failed jobs:", flush=True)
        for info in failed:
            print(f"  {info['station']} chunk={info['chunk_i']} t0={info['t0']:.3f} out={info['out']}", flush=True)
        raise SystemExit(1)


def merge_station(indir, station, out_file, overwrite=False):
    files = sorted(Path(indir).glob(f"rdmf_peak_{station}_*.h5"))
    rows = []
    for fn in files:
        try:
            with h5py.File(fn, "r") as h:
                if int(h.attrs.get("complete", 0)) != 1:
                    continue
                rows.append((float(h["t_sample"][0]), fn))
        except OSError:
            continue
    if not rows:
        raise FileNotFoundError(f"No complete chunk files for {station} in {indir}")
    rows.sort()
    files = [fn for _, fn in rows]
    out_file = Path(out_file)
    if out_file.exists() and not overwrite:
        raise FileExistsError(f"{out_file} exists; refusing to overwrite")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(files[0], "r") as h0:
        n_total = 0
        for fn in files:
            with h5py.File(fn, "r") as h:
                n_total += h["P"].shape[0]
        n_range = h0["P"].shape[1]
        n_ch = h0["voltage_peak"].shape[1]
        n_tx = h0["voltage_peak"].shape[3]
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")
        with h5py.File(tmp, "w") as out:
            for key in (
                "station", "channels", "seeds", "code_len", "frequency_hz", "gps_rx", "gps_tx",
                "range_idx", "range_km", "fvec", "sample_rate", "t0_sample", "n_coh",
                "codes_per_step", "range_interp", "range_resolution_m", "ignore_freq",
                "head_min_doppler_hz", "dc_cancel", "dc_cancel_seconds", "dc_cancel_start_second", "doppler_whiten",
            ):
                if key in h0:
                    out[key] = h0[key][()]
            for k, v in h0.attrs.items():
                out.attrs[k] = v
            out.attrs["merged_from"] = str(Path(indir))
            out.attrs["complete"] = 1
            out.attrs["n_chunks"] = len(files)
            chunks_2d = (min(128, n_total), min(256, n_range))
            chunks_v = (1, n_ch, min(256, n_range), n_tx)
            out.create_dataset("t_sample", (n_total,), dtype="f8")
            out.create_dataset("N", (n_total, n_range), dtype="f4", chunks=chunks_2d)
            for key in ("P_rti", "D_rti", "P_head", "D_head", "P", "D"):
                out.create_dataset(key, (n_total, n_range), dtype="f4", chunks=chunks_2d)
            out.create_dataset("voltage_peak", (n_total, n_ch, n_range, n_tx), dtype="c8", chunks=chunks_v)

            pos = 0
            for fn in files:
                with h5py.File(fn, "r") as h:
                    n = h["P"].shape[0]
                    sl = slice(pos, pos + n)
                    out["t_sample"][sl] = h["t_sample"][()]
                    out["N"][sl, :] = h["N"][()]
                    if "P_rti" in h:
                        p_rti = h["P_rti"][()]
                        d_rti = h["D_rti"][()]
                        p_head = h["P_head"][()]
                        d_head = h["D_head"][()]
                    else:
                        p_rti = h["P"][()]
                        d_rti = h["D"][()]
                        p_head = p_rti
                        d_head = d_rti
                    out["P_rti"][sl, :] = p_rti
                    out["D_rti"][sl, :] = d_rti
                    out["P_head"][sl, :] = p_head
                    out["D_head"][sl, :] = d_head
                    out["P"][sl, :] = p_head
                    out["D"][sl, :] = d_head
                    out["voltage_peak"][sl, :, :, :] = h["voltage_peak"][()]
                    pos += n
        tmp.replace(out_file)
    print(f"{station}: merged {len(files)} chunks -> {out_file}", flush=True)


def run_merge(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in args.stations:
        merge_station(args.in_dir, station, out_dir / f"rdmf_peak_{station}_merged.h5", overwrite=args.overwrite)


def parser():
    p = argparse.ArgumentParser(description="Parallel simple range-Doppler matched filter with peak voltage storage.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--data-dir", default="2026_05_30_bolide")
        q.add_argument("--iso-hour", default="2026-05-30T18-00-00")
        q.add_argument("--channels", nargs="+", type=int, default=[0, 1])
        q.add_argument("--start-second", type=float, default=80.0)
        q.add_argument("--seconds", type=float, default=None)
        q.add_argument("--max-steps", type=int)
        q.add_argument("--n-coh", type=int, default=8)
        q.add_argument("--codes-per-step", type=int, default=1)
        q.add_argument("--range-interp", type=int, default=2)
        q.add_argument("--range-min-km", type=float, default=0.0)
        q.add_argument("--range-max-km", type=float, default=600.0)
        q.add_argument("--fmin", type=float, default=-50000.0)
        q.add_argument("--fmax", type=float, default=50000.0)
        q.add_argument("--ignore-freq", type=float, default=0.0)
        q.add_argument("--head-min-doppler-hz", type=float, default=250.0)
        q.add_argument("--dc-cancel", action="store_true")
        q.add_argument("--dc-cancel-seconds", type=float, default=1.0)
        q.add_argument("--dc-cancel-start-second", type=float, default=0.0)
        q.add_argument("--spike-sigma", type=float, default=0.0)
        q.add_argument("--doppler-whiten", action="store_true", default=True)
        q.add_argument("--no-doppler-whiten", action="store_false", dest="doppler_whiten")
        q.add_argument("--progress-every", type=int, default=25)
        q.add_argument("--flush-every", type=int, default=10)
        q.add_argument("--overwrite", action="store_true")

    ps = sub.add_parser("station")
    common(ps)
    ps.add_argument("--station", required=True)
    ps.add_argument("--out", required=True)

    pb = sub.add_parser("batch")
    common(pb)
    pb.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    pb.add_argument("--dc-cancel-stations", nargs="+", default=["clinton"])
    pb.add_argument("--total-seconds", type=float, default=100.0)
    pb.add_argument("--chunk-seconds", type=float, default=0.1)
    pb.add_argument("--chunk-steps", type=int)
    pb.add_argument("--max-parallel", type=int, default=128)
    pb.add_argument("--poll-seconds", type=float, default=2.0)
    pb.add_argument("--out-dir", required=True)

    pm = sub.add_parser("merge")
    pm.add_argument("--in-dir", required=True)
    pm.add_argument("--out-dir", required=True)
    pm.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    pm.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.cmd == "station":
        run_station(args)
    elif args.cmd == "batch":
        run_batch(args)
    elif args.cmd == "merge":
        run_merge(args)


if __name__ == "__main__":
    main()
