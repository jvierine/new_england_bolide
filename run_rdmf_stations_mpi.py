#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def add_opt(cmd, flag, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend(str(v) for v in value)
        return
    cmd.extend([flag, str(value)])


def worker_cmd(config, station):
    output = Path(config["out_dir"]) / f"rdmf_{station}_{config.get('tag', 'mpi')}.h5"
    cmd = [
        sys.executable,
        "range_doppler_mf_bolide.py",
        "--station",
        station,
        "--out",
        str(output),
    ]
    scalar_flags = {
        "data_dir": "--data-dir",
        "iso_hour": "--iso-hour",
        "start_second": "--start-second",
        "seconds": "--seconds",
        "t0": "--t0",
        "t1": "--t1",
        "n_coh": "--n-coh",
        "codes_per_step": "--codes-per-step",
        "range_interp": "--range-interp",
        "doppler_zero_pad_factor": "--doppler-zero-pad-factor",
        "range_resolution_m": "--range-resolution-m",
        "max_range_km": "--max-range-km",
        "rmin_idx": "--rmin-idx",
        "rmax_idx": "--rmax-idx",
        "trajectory_range_json": "--trajectory-range-json",
        "trajectory_range_pad_pixels": "--trajectory-range-pad-pixels",
        "fmin": "--fmin",
        "fmax": "--fmax",
        "ignore_freq": "--ignore-freq",
        "head_min_doppler_hz": "--head-min-doppler-hz",
        "dc_cancel_seconds": "--dc-cancel-seconds",
        "dc_cancel_start_second": "--dc-cancel-start-second",
        "spike_sigma": "--spike-sigma",
        "tx_aod_trajectory_json": "--tx-aod-trajectory-json",
        "tx_aod_phase_sign": "--tx-aod-phase-sign",
        "rdmf_frame_dir": "--rdmf-frame-dir",
        "rdmf_frame_vmin_db": "--rdmf-frame-vmin-db",
        "rdmf_frame_vmax_db": "--rdmf-frame-vmax-db",
        "rdmf_frame_every": "--rdmf-frame-every",
        "progress_every": "--progress-every",
        "checkpoint_every": "--checkpoint-every",
    }
    for key, flag in scalar_flags.items():
        add_opt(cmd, flag, config.get(key))
    add_opt(cmd, "--channels", config.get("channels"))
    if config.get("doppler_whiten", False):
        cmd.append("--doppler-whiten")
    else:
        cmd.append("--no-doppler-whiten")
    for key, flag in (
        ("tx_aod_beamform", "--tx-aod-beamform"),
        ("polarization_combine", "--polarization-combine"),
        ("save_rdmf_frames", "--save-rdmf-frames"),
    ):
        if config.get(key, False):
            cmd.append(flag)
    dc_cancel = set(config.get("dc_cancel_stations", []))
    if station in dc_cancel:
        cmd.append("--dc-cancel")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="MPI launcher for station RDMF workers.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stations = list(config["stations"])
    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    if config.get("rdmf_frame_dir"):
        Path(config["rdmf_frame_dir"]).mkdir(parents=True, exist_ok=True)

    failures = []
    for station in stations[rank::size]:
        cmd = worker_cmd(config, station)
        print(f"[rank {rank}] START {station}: {' '.join(cmd)}", flush=True)
        res = subprocess.run(cmd, cwd=Path.cwd(), text=True, check=False)
        if res.returncode != 0:
            failures.append((station, res.returncode))
        print(f"[rank {rank}] DONE {station} rc={res.returncode}", flush=True)

    all_failures = comm.gather(failures, root=0)
    if rank == 0:
        flat = [item for sub in all_failures for item in sub]
        if flat:
            raise SystemExit(f"RDMF MPI failures: {flat}")


if __name__ == "__main__":
    main()
