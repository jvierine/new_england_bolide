#!/usr/bin/env python3
"""Run station-window RDMF products for head-echo measurement selection.

The output products are intentionally one code long, stepped by one code, with
16x Doppler zero padding. Range gates are restricted to the station intervals
used by the RCS memo / station RTI plots.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from range_doppler_mf_bolide import read_config  # noqa: E402

STATION_RANGES_KM = {
    "beverly": (180.0, 275.0),
    "clinton": (220.0, 280.0),
    "pomfret": (350.0, 420.0),
}

DEFAULT_OUT_DIR = ROOT / "products" / "rdmf_txsteer_pol_zp16_1code_station_ranges"
DEFAULT_TRAJECTORY = ROOT / "products" / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5" / "ballistic_head_trail_correlated_fit.json"


def range_indices(data_dir: Path, station: str, rmin_km: float, rmax_km: float, range_resolution_m: float, range_interp: int) -> tuple[int, int]:
    config = read_config(data_dir, station)
    gate_km = float(range_resolution_m) / float(range_interp) / 1000.0
    config_rmin_idx = int(config["rmin_idx"]) * int(range_interp)
    config_rmax_idx = int(config["rmax_idx"]) * int(range_interp)
    rmin_idx = max(config_rmin_idx, int(round(rmin_km / gate_km)))
    rmax_idx = min(config_rmax_idx, int(round(rmax_km / gate_km)))
    if rmax_idx < rmin_idx:
        raise ValueError(f"{station}: empty range index interval for {rmin_km}-{rmax_km} km")
    return rmin_idx, rmax_idx


def command_for_station(station: str, args: argparse.Namespace) -> list[str]:
    rmin_km, rmax_km = STATION_RANGES_KM[station]
    rmin_idx, rmax_idx = range_indices(args.data_dir, station, rmin_km, rmax_km, args.range_resolution_m, args.range_interp)
    out = args.out_dir / f"rdmf_peak_{station}_merged.h5"
    cmd = [
        str(args.python),
        "range_doppler_mf_bolide.py",
        "--data-dir",
        str(args.data_dir),
        "--station",
        station,
        "--iso-hour",
        args.iso_hour,
        "--channels",
        *[str(ch) for ch in args.channels],
        "--t0",
        args.t0,
        "--t1",
        args.t1,
        "--n-coh",
        "1",
        "--codes-per-step",
        "1",
        "--head-incoherent-codes",
        str(args.head_incoherent_codes),
        "--range-interp",
        str(args.range_interp),
        "--doppler-zero-pad-factor",
        str(args.doppler_zero_pad_factor),
        "--range-resolution-m",
        str(args.range_resolution_m),
        "--rmin-idx",
        str(rmin_idx),
        "--rmax-idx",
        str(rmax_idx),
        "--fmin",
        str(args.fmin),
        "--fmax",
        str(args.fmax),
        "--ignore-freq",
        str(args.ignore_freq),
        "--head-min-doppler-hz",
        str(args.head_min_doppler_hz),
        "--head-search-range-half-px",
        str(args.head_search_range_half_px),
        "--head-search-range-half-km",
        str(args.head_search_range_half_km),
        "--head-search-doppler-half-hz",
        str(args.head_search_doppler_half_hz),
        "--no-doppler-whiten",
        "--tx-aod-beamform",
        "--tx-aod-trajectory-json",
        str(args.trajectory_json),
        "--tx-aod-phase-sign",
        str(args.tx_aod_phase_sign),
        "--polarization-combine",
        "--progress-every",
        str(args.progress_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--out",
        str(out),
    ]
    if args.save_rdmf_frames:
        cmd.extend(
            [
                "--save-rdmf-frames",
                "--rdmf-frame-dir",
                str(args.rdmf_frame_dir or (args.out_dir / "frames")),
                "--rdmf-frame-vmin-db",
                str(args.rdmf_frame_vmin_db),
                "--rdmf-frame-vmax-db",
                str(args.rdmf_frame_vmax_db),
                "--rdmf-frame-every",
                str(args.rdmf_frame_every),
                "--rdmf-frame-zoom-range-half-km",
                str(args.rdmf_frame_zoom_range_half_km),
                "--rdmf-frame-zoom-doppler-half-hz",
                str(args.rdmf_frame_zoom_doppler_half_hz),
            ]
        )
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stations", nargs="+", default=list(STATION_RANGES_KM))
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--t0", default="18:06:18")
    p.add_argument("--t1", default="18:06:23")
    p.add_argument("--range-interp", type=int, default=1)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=16)
    p.add_argument("--range-resolution-m", type=float, default=3000.0)
    p.add_argument("--fmin", type=float, default=-4000.0)
    p.add_argument("--fmax", type=float, default=1000.0)
    p.add_argument("--ignore-freq", type=float, default=150.0)
    p.add_argument("--head-min-doppler-hz", type=float, default=250.0)
    p.add_argument("--head-search-range-half-px", type=int, default=20)
    p.add_argument("--head-search-range-half-km", type=float, default=20.0)
    p.add_argument("--head-search-doppler-half-hz", type=float, default=1000.0)
    p.add_argument("--head-incoherent-codes", type=int, default=1)
    p.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--tx-aod-phase-sign", type=float, default=-1.0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--save-rdmf-frames", action="store_true")
    p.add_argument("--rdmf-frame-dir", type=Path, default=None)
    p.add_argument("--rdmf-frame-vmin-db", type=float, default=0.0)
    p.add_argument("--rdmf-frame-vmax-db", type=float, default=23.0)
    p.add_argument("--rdmf-frame-every", type=int, default=1)
    p.add_argument("--rdmf-frame-zoom-range-half-km", type=float, default=12.0)
    p.add_argument("--rdmf-frame-zoom-doppler-half-hz", type=float, default=700.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mpi-dispatch", action="store_true", help="When run under mpi4py, split stations over ranks.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stations = list(args.stations)
    if args.mpi_dispatch:
        try:
            from mpi4py import MPI
        except ModuleNotFoundError as exc:
            raise SystemExit("--mpi-dispatch requires mpi4py") from exc
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        stations = stations[rank::size]
        print(f"MPI rank {rank}/{size} assigned stations: {stations}", flush=True)
    for station in stations:
        cmd = command_for_station(station, args)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
