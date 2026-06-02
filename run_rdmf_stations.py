#!/usr/bin/env python3
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def reader_thread(station, proc, outq):
    for line in proc.stdout:
        outq.put((station, line.rstrip()))
    outq.put((station, None))


def build_worker_cmd(args, station, out_file):
    cmd = [
        sys.executable,
        "range_doppler_mf_bolide.py",
        "--data-dir",
        args.data_dir,
        "--station",
        station,
        "--iso-hour",
        args.iso_hour,
        "--channels",
        *[str(ch) for ch in args.channels],
        "--start-second",
        str(args.start_second),
        "--n-coh",
        str(args.n_coh),
        "--codes-per-step",
        str(args.codes_per_step),
        "--range-interp",
        str(args.range_interp),
        "--range-resolution-m",
        str(args.range_resolution_m),
        "--fmin",
        str(args.fmin),
        "--fmax",
        str(args.fmax),
        "--ignore-freq",
        str(args.ignore_freq),
        "--head-min-doppler-hz",
        str(args.head_min_doppler_hz),
        "--dc-cancel-seconds",
        str(args.dc_cancel_seconds),
        "--dc-cancel-start-second",
        str(args.dc_cancel_start_second),
        "--spike-sigma",
        str(args.spike_sigma),
        "--progress-every",
        str(args.progress_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--out",
        str(out_file),
    ]
    if args.seconds is not None:
        cmd.extend(["--seconds", str(args.seconds)])
    if args.rmin_idx is not None:
        cmd.extend(["--rmin-idx", str(args.rmin_idx)])
    if args.rmax_idx is not None:
        cmd.extend(["--rmax-idx", str(args.rmax_idx)])
    if args.max_range_km is not None:
        cmd.extend(["--max-range-km", str(args.max_range_km)])
    if args.doppler_whiten:
        cmd.append("--doppler-whiten")
    if station in set(args.dc_cancel_stations):
        cmd.append("--dc-cancel")
    return cmd


def maybe_plot(args, station, out_file, force=False):
    if not out_file.exists():
        return False
    now = time.time()
    last = args._last_plot_time.get(station, 0.0)
    mtime = out_file.stat().st_mtime
    last_mtime = args._last_plot_mtime.get(station, 0.0)
    if not force and (now - last < args.plot_every or mtime <= last_mtime):
        return False

    cmd = [
        sys.executable,
        "plot_rdmf_bolide.py",
        str(out_file),
        "--snr-threshold",
        str(args.snr_threshold),
    ]
    try:
        res = subprocess.run(cmd, cwd=args.cwd, text=True, capture_output=True, check=False)
        if res.returncode == 0:
            args._last_plot_time[station] = now
            args._last_plot_mtime[station] = mtime
            print(f"[{station}] PLOT refreshed {out_file.with_suffix('')}_rti.png", flush=True)
            return True
        print(f"[{station}] PLOT failed: {res.stderr.strip() or res.stdout.strip()}", flush=True)
    except Exception as exc:
        print(f"[{station}] PLOT failed: {exc}", flush=True)
    return False


def main():
    p = argparse.ArgumentParser(description="Run one range-Doppler matched-filter worker per station.")
    p.add_argument("--data-dir", default="2026_05_30_bolide")
    p.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
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
    p.add_argument("--fmin", type=float, default=-12000.0)
    p.add_argument("--fmax", type=float, default=12000.0)
    p.add_argument("--ignore-freq", type=float, default=250.0)
    p.add_argument("--head-min-doppler-hz", type=float, default=250.0)
    p.add_argument("--dc-cancel-stations", nargs="+", default=["clinton"])
    p.add_argument("--dc-cancel-seconds", type=float, default=1.0)
    p.add_argument("--dc-cancel-start-second", type=float, default=0.0)
    p.add_argument("--spike-sigma", type=float, default=0.0)
    p.add_argument("--doppler-whiten", action="store_true", default=True)
    p.add_argument("--no-doppler-whiten", action="store_false", dest="doppler_whiten")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--plot-every", type=float, default=60.0)
    p.add_argument("--snr-threshold", type=float, default=3.5)
    p.add_argument("--out-dir", default="products/full_run")
    args = p.parse_args()
    args.cwd = Path.cwd()
    args._last_plot_time = {}
    args._last_plot_mtime = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outq = queue.Queue()
    procs = {}
    out_files = {}
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for station in args.stations:
        start_tag = f"{int(args.start_second):04d}s"
        dur_tag = "all" if args.seconds is None else f"{int(args.seconds)}s"
        out_file = out_dir / f"rdmf_{station}_{start_tag}_{dur_tag}.h5"
        out_files[station] = out_file
        cmd = build_worker_cmd(args, station, out_file)
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        procs[station] = proc
        threading.Thread(target=reader_thread, args=(station, proc, outq), daemon=True).start()
        print(f"[{station}] START pid={proc.pid} out={out_file}", flush=True)

    active = set(procs)
    t0 = time.time()
    while active:
        try:
            station, line = outq.get(timeout=2.0)
        except queue.Empty:
            for st in list(active):
                maybe_plot(args, st, out_files[st])
            continue

        if line is None:
            active.discard(station)
            rc = procs[station].wait()
            status = "OK" if rc == 0 else f"FAILED rc={rc}"
            print(f"[{station}] DONE {status}", flush=True)
            maybe_plot(args, station, out_files[station], force=True)
            continue

        print(f"[{station}] {line}", flush=True)
        if line.startswith("CHECKPOINT"):
            maybe_plot(args, station, out_files[station])

    elapsed = time.time() - t0
    failed = [st for st, proc in procs.items() if proc.returncode != 0]
    print(f"All station workers finished in {elapsed:.1f}s", flush=True)
    if failed:
        print(f"Failed stations: {', '.join(failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
