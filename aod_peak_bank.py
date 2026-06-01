#!/usr/bin/env python3
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from angle_departure_estimator import estimate_aod, load_station_yaml, regular_direction_grid
from range_doppler_mf_bolide import (
    detrend_window,
    get_codes,
    interpolate_rect,
    read_channel_files,
    rf_file_for_time,
)


RANGE_WINDOWS_KM = {
    "beverly": (0.0, 300.0),
    "clinton": (0.0, 300.0),
    "pomfret": (300.0, 450.0),
    "siena": (400.0, 500.0),
    "amherstnh": (0.0, 350.0),
}
DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


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


def make_output(args, site, n_steps, range_idx, fvec, dir_unit, dir_az, dir_el):
    ofile = Path(args.out)
    if ofile.exists() and not args.overwrite:
        raise FileExistsError(f"{ofile} exists; refusing to overwrite")
    ofile.parent.mkdir(parents=True, exist_ok=True)
    range_km = range_idx * site["range_resolution_m"] / args.range_interp / 1000.0
    n_ch = len(args.channels)
    n_range = len(range_idx)
    n_tx = len(site["seeds"])
    chunks_2d = (min(64, n_steps), min(128, n_range))
    chunks_v = (1, n_ch, min(128, n_range), n_tx)

    h = h5py.File(ofile, "w")
    h["station"] = np.bytes_(site["station"])
    h["channels"] = np.array(args.channels, dtype=np.int32)
    h["seeds"] = np.array(site["seeds"], dtype=np.int64)
    h["frequency_hz"] = site["frequency_hz"]
    h["gps_tx"] = site["gps_tx"]
    h["gps_rx"] = site["gps_rx"]
    h["antenna_pos_tx_enu_m"] = site["antenna_pos_tx"]
    h["range_idx"] = range_idx.astype(np.int32)
    h["range_km"] = range_km.astype(np.float32)
    h["fvec"] = fvec.astype(np.float32)
    h["direction_unit_enu"] = dir_unit.astype(np.float32)
    h["direction_az_deg"] = dir_az.astype(np.float32)
    h["direction_el_deg"] = dir_el.astype(np.float32)
    h["sample_rate"] = float(args.sample_rate)
    h["t0_sample"] = float(args.abs_index0)
    h["n_coh"] = args.n_coh
    h["codes_per_step"] = args.codes_per_step
    h["range_interp"] = args.range_interp
    h["snr_threshold_db"] = args.snr_threshold_db
    h.attrs["config_path"] = site["config_path"]
    h.attrs["start_second"] = args.start_second
    h.attrs["seconds"] = -1.0 if args.seconds is None else args.seconds
    h.attrs["complete"] = 0
    h.attrs["n_done"] = 0
    h.create_dataset("t_sample", (n_steps,), dtype="f8")
    h.create_dataset("peak_power", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("noise_floor", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("snr", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("peak_doppler_hz", (n_steps, n_range), dtype="f4", chunks=chunks_2d)
    h.create_dataset("aod_az_deg", (n_steps, n_range), dtype="f4", chunks=chunks_2d, fillvalue=np.nan)
    h.create_dataset("aod_el_deg", (n_steps, n_range), dtype="f4", chunks=chunks_2d, fillvalue=np.nan)
    h.create_dataset("aod_score", (n_steps, n_range), dtype="f4", chunks=chunks_2d, fillvalue=np.nan)
    h.create_dataset("voltage_mf", (n_steps, n_ch, n_range, n_tx), dtype="c8", chunks=chunks_v)
    h.create_dataset("voltage_lstsq", (n_steps, n_ch, n_range, n_tx), dtype="c8", chunks=chunks_v)
    return h


def steering_grid(site, dir_unit, phase_sign):
    ant = site["antenna_pos_tx"]
    wavelength = 299792458.0 / site["frequency_hz"]
    k = 2.0 * np.pi / wavelength
    phase = phase_sign * k * (dir_unit @ ant.T)
    return np.exp(1j * phase).astype(np.complex64)


def estimate_many_aod(voltages, snr_row, threshold_linear, steering, dir_az, dir_el):
    n_range = voltages.shape[1]
    az = np.full(n_range, np.nan, dtype=np.float32)
    el = np.full(n_range, np.nan, dtype=np.float32)
    score = np.full(n_range, np.nan, dtype=np.float32)
    good = np.where(snr_row >= threshold_linear)[0]
    if len(good) == 0:
        return az, el, score

    y = voltages[:, good, :].astype(np.complex64)
    norm = np.sqrt(np.sum(np.abs(y) ** 2, axis=2))
    norm[norm == 0] = 1.0
    y = y / norm[:, :, None]
    # scores: channel, selected_range, direction
    coh = np.einsum("crt,dt->crd", y, np.conj(steering), optimize=True)
    sc = np.sum(np.abs(coh) ** 2, axis=0) / (voltages.shape[0] * voltages.shape[2] ** 2)
    imax = np.argmax(sc, axis=1)
    az[good] = dir_az[imax]
    el[good] = dir_el[imax]
    score[good] = sc[np.arange(len(good)), imax]
    return az, el, score


def solve_lstsq_voltages(segment, codes, doppler_hz, sample_rate_i):
    n = len(segment)
    t = np.arange(n, dtype=np.float64) / sample_rate_i
    phasor = np.exp(2.0j * np.pi * doppler_hz * t).astype(np.complex64)
    A = np.column_stack([code * phasor for code in codes])
    x, *_ = np.linalg.lstsq(A, segment, rcond=None)
    return x.astype(np.complex64)


def run_station(args):
    site = load_station_yaml(args.data_dir, args.station)
    if args.range_min_km is None or args.range_max_km is None:
        args.range_min_km, args.range_max_km = RANGE_WINDOWS_KM[args.station]
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

    fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    fidx = np.where((fvec >= args.fmin) & (fvec <= args.fmax))[0]
    fvec0 = fvec[fidx]
    codes = get_codes(site["seeds"], site["code_len"], args.n_coh, range_interp=args.range_interp)
    codes_arr = np.array(codes)
    threshold_linear = 10.0 ** (args.snr_threshold_db / 10.0)
    dir_unit, dir_az, dir_el = regular_direction_grid(
        grid_size=args.grid_size,
        max_zenith_angle=args.max_zenith_angle,
    )
    steering = steering_grid(site, dir_unit, args.phase_sign)

    h = make_output(args, site, n_steps, range_idx, fvec0, dir_unit, dir_az, dir_el)
    t0 = time.time()
    try:
        for ti in range(n_steps):
            start = ti * step_len
            S = np.zeros((len(range_idx), len(fidx)), dtype=np.float32)
            Z_store = np.zeros((len(args.channels), len(range_idx), len(codes), len(fidx)), dtype=np.complex64)

            for chi, z in enumerate(zc):
                z0 = detrend_window(z[start : start + read_len], spike_sigma=args.spike_sigma)
                windows = np.lib.stride_tricks.sliding_window_view(z0, win_len)[range_idx, :]
                for ci, code in enumerate(codes):
                    Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
                    Z_store[chi, :, ci, :] = Z
                    S += np.abs(Z) ** 2

            S /= float(len(args.channels) * len(codes))
            noise = np.median(S, axis=1)
            imax = np.argmax(S, axis=1)
            peak = S[np.arange(len(range_idx)), imax]
            snr = peak / np.maximum(noise, 1e-12)
            dop = fvec0[imax]

            vmf = np.zeros((len(args.channels), len(range_idx), len(codes)), dtype=np.complex64)
            vls = np.zeros_like(vmf)
            for ri, fi in enumerate(imax):
                vmf[:, ri, :] = Z_store[:, ri, :, fi]

            # Doppler de-mixed TX-code amplitudes at the same picked Doppler/range.
            for chi, z in enumerate(zc):
                z0 = detrend_window(z[start : start + read_len], spike_sigma=args.spike_sigma)
                for ri, rg in enumerate(range_idx):
                    segment = z0[rg : rg + win_len]
                    vls[chi, ri, :] = solve_lstsq_voltages(segment, codes_arr, dop[ri], sample_rate_i)

            az, el, score = estimate_many_aod(vls, snr, threshold_linear, steering, dir_az, dir_el)

            h["t_sample"][ti] = abs_index0 + ti * original_step_len
            h["peak_power"][ti, :] = peak
            h["noise_floor"][ti, :] = noise
            h["snr"][ti, :] = snr
            h["peak_doppler_hz"][ti, :] = dop
            h["aod_az_deg"][ti, :] = az
            h["aod_el_deg"][ti, :] = el
            h["aod_score"][ti, :] = score
            h["voltage_mf"][ti, :, :, :] = vmf
            h["voltage_lstsq"][ti, :, :, :] = vls

            n_done = ti + 1
            h.attrs["n_done"] = n_done
            if n_done % args.flush_every == 0 or n_done == n_steps:
                h.flush()
            if n_done == 1 or n_done == n_steps or n_done % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta = (n_steps - n_done) / max(rate, 1e-9)
                n_det = int(np.sum(snr >= threshold_linear))
                best = int(np.argmax(snr))
                print(
                    f"PROGRESS {n_done}/{n_steps} elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"rate={rate:.3f}steps/s det_ranges={n_det} "
                    f"best_range_km={range_idx[best] * gate_km:.1f} "
                    f"dop={dop[best]:.1f}Hz snr_db={10*np.log10(max(snr[best], 1e-12)):.1f}",
                    flush=True,
                )
        h.attrs["complete"] = 1
    finally:
        h.flush()
        h.close()
    print(f"Wrote {args.out}", flush=True)


def reader_thread(station, proc, outq):
    for line in proc.stdout:
        outq.put((station, line.rstrip()))
    outq.put((station, None))


def run_parallel(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists and is non-empty; choose a new --out-dir or pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = {}
    outq = queue.Queue()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for station in args.stations:
        r0, r1 = RANGE_WINDOWS_KM[station]
        out = out_dir / f"aod_peak_{station}_{int(args.start_second):04d}s.h5"
        cmd = [
            sys.executable, __file__, "station",
            "--data-dir", args.data_dir,
            "--station", station,
            "--iso-hour", args.iso_hour,
            "--channels", *[str(ch) for ch in args.channels],
            "--start-second", str(args.start_second),
            "--n-coh", str(args.n_coh),
            "--codes-per-step", str(args.codes_per_step),
            "--range-interp", str(args.range_interp),
            "--range-min-km", str(r0),
            "--range-max-km", str(r1),
            "--fmin", str(args.fmin),
            "--fmax", str(args.fmax),
            "--snr-threshold-db", str(args.snr_threshold_db),
            "--grid-size", str(args.grid_size),
            "--max-zenith-angle", str(args.max_zenith_angle),
            "--phase-sign", str(args.phase_sign),
            "--progress-every", str(args.progress_every),
            "--flush-every", str(args.flush_every),
            "--out", str(out),
        ]
        if args.seconds is not None:
            cmd += ["--seconds", str(args.seconds)]
        if args.overwrite:
            cmd += ["--overwrite"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        procs[station] = proc
        threading.Thread(target=reader_thread, args=(station, proc, outq), daemon=True).start()
        print(f"[{station}] START pid={proc.pid} out={out}", flush=True)

    active = set(procs)
    while active:
        station, line = outq.get()
        if line is None:
            active.remove(station)
            rc = procs[station].wait()
            print(f"[{station}] DONE rc={rc}", flush=True)
            continue
        print(f"[{station}] {line}", flush=True)

    failed = [s for s, p in procs.items() if p.returncode != 0]
    if failed:
        raise SystemExit(f"failed stations: {failed}")


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
        r0, r1 = RANGE_WINDOWS_KM[station]
        for chunk_i, t0, sec in chunk_starts(args.start_second, args.total_seconds, args.chunk_seconds):
            out = out_dir / f"aod_peak_{station}_{int(round(t0 * 1000)):09d}ms_{int(round(sec * 1000)):06d}ms.h5"
            cmd = [
                sys.executable, __file__, "station",
                "--data-dir", args.data_dir,
                "--station", station,
                "--iso-hour", args.iso_hour,
                "--channels", *[str(ch) for ch in args.channels],
                "--start-second", f"{t0:.6f}",
                "--seconds", f"{sec:.6f}",
                "--n-coh", str(args.n_coh),
                "--codes-per-step", str(args.codes_per_step),
                "--range-interp", str(args.range_interp),
                "--range-min-km", str(r0),
                "--range-max-km", str(r1),
                "--fmin", str(args.fmin),
                "--fmax", str(args.fmax),
                "--snr-threshold-db", str(args.snr_threshold_db),
                "--grid-size", str(args.grid_size),
                "--max-zenith-angle", str(args.max_zenith_angle),
                "--phase-sign", str(args.phase_sign),
                "--progress-every", str(args.progress_every),
                "--flush-every", str(args.flush_every),
                "--out", str(out),
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            tasks.append((station, chunk_i, t0, sec, out, cmd))

    print(f"Queued {len(tasks)} station/chunk jobs with max_parallel={args.max_parallel}", flush=True)
    running = {}
    task_i = 0
    complete = 0
    failed = []
    t_batch = time.time()
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
            elapsed = time.time() - t_batch
            rate = complete / max(elapsed, 1e-9)
            eta = (len(tasks) - complete) / max(rate, 1e-9)
            status = "OK" if rc == 0 else f"FAILED rc={rc}"
            print(
                f"DONE {status} {complete}/{len(tasks)} pid={pid} "
                f"{info['station']} chunk={info['chunk_i']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            if rc != 0:
                failed.append(info)

    if failed:
        print("Failed jobs:", flush=True)
        for info in failed:
            print(f"  {info['station']} chunk={info['chunk_i']} t0={info['t0']:.3f} out={info['out']}", flush=True)
        raise SystemExit(1)


def parser():
    p = argparse.ArgumentParser(description="AoD peak-Doppler matched-filter bank with complex voltage storage.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(q):
        q.add_argument("--data-dir", default="2026_05_30_bolide")
        q.add_argument("--iso-hour", default="2026-05-30T18-00-00")
        q.add_argument("--channels", nargs="+", type=int, default=[0, 1])
        q.add_argument("--start-second", type=float, default=80.0)
        q.add_argument("--seconds", type=float, default=None)
        q.add_argument("--n-coh", type=int, default=8)
        q.add_argument("--codes-per-step", type=int, default=1)
        q.add_argument("--range-interp", type=int, default=2)
        q.add_argument("--fmin", type=float, default=-50000.0)
        q.add_argument("--fmax", type=float, default=50000.0)
        q.add_argument("--snr-threshold-db", type=float, default=6.0)
        q.add_argument("--grid-size", type=int, default=121)
        q.add_argument("--max-zenith-angle", type=float, default=90.0)
        q.add_argument("--phase-sign", type=float, choices=[-1.0, 1.0], default=1.0)
        q.add_argument("--spike-sigma", type=float, default=0.0)
        q.add_argument("--progress-every", type=int, default=25)
        q.add_argument("--flush-every", type=int, default=10)
        q.add_argument("--overwrite", action="store_true")

    ps = sub.add_parser("station")
    add_common(ps)
    ps.add_argument("--station", required=True)
    ps.add_argument("--range-min-km", type=float)
    ps.add_argument("--range-max-km", type=float)
    ps.add_argument("--out", required=True)

    pr = sub.add_parser("parallel")
    add_common(pr)
    pr.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    pr.add_argument("--out-dir", required=True)

    pb = sub.add_parser("batch")
    add_common(pb)
    pb.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    pb.add_argument("--total-seconds", type=float, default=100.0)
    pb.add_argument("--chunk-seconds", type=float, default=1.0)
    pb.add_argument("--max-parallel", type=int, default=128)
    pb.add_argument("--poll-seconds", type=float, default=2.0)
    pb.add_argument("--out-dir", required=True)
    return p


def main():
    args = parser().parse_args()
    if args.cmd == "station":
        run_station(args)
    elif args.cmd == "parallel":
        run_parallel(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
