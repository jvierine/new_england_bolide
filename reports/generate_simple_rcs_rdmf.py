#!/usr/bin/env python3
"""Simple RDMF product for SNR/RCS diagnostics.

This processor intentionally avoids the whitening and 2-D FFT filtering used by
the trajectory products. It uses one-code coherent integrations, sums power
incoherently over 16 adjacent code periods, records the peak power over Doppler,
and separately records the zero-Doppler-bin power for later noise/RCS work.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml  # noqa: E402
from estimate_head_echo_trajectory import predict_ranges, station_geometry  # noqa: E402
from range_doppler_mf_bolide import detrend_window, get_codes, interpolate_rect, read_channel_files, rf_file_for_time  # noqa: E402
from station_config import short_station_name  # noqa: E402

DATA_DIR = ROOT / "2026_05_30_bolide"
PRODUCT_DIR = ROOT / "products" / "rdmf_simple_rcs_1coh_16incoh_180609_180625"
FIG_DIR = ROOT / "reports" / "figures"
TRAJECTORY_JSON = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"

EVENT_T0 = 1780164380.0
REF_START_UNIX = 1780164369.0
REF_END_UNIX = 1780164379.0
RUN_START_UNIX = 1780164369.0
RUN_END_UNIX = 1780164385.0
PLOT_START_UNIX = 1780164378.0
PLOT_END_UNIX = 1780164384.0

STATION_RANGES_KM = [
    ("beverly", 180.0, 275.0),
    ("clinton", 220.0, 280.0),
    ("pomfret", 350.0, 420.0),
    ("siena", 400.0, 500.0),
    ("amherstnh", 200.0, 350.0),
]
HEAD_TRACE_STATIONS = ["beverly", "clinton", "pomfret"]


def unix_to_mpl(t: np.ndarray) -> np.ndarray:
    ns = np.asarray(np.asarray(t, dtype=float) * 1e9, dtype="datetime64[ns]")
    return mdates.date2num(ns.astype("datetime64[us]").astype(object))


def unix_label(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).strftime("%H:%M:%S")


def station_start_second(station: str, channel: int, iso_hour: str, target_unix: float) -> float:
    files = rf_file_for_time(Path(DATA_DIR) / station, channel, iso_hour)
    with h5py.File(files[0], "r") as h:
        first_index = int(h["rf_data_index"][0, 0])
        sr = float(h["rf_data"].attrs["sample_rate_numerator"]) / float(h["rf_data"].attrs["sample_rate_denominator"])
    return float(target_unix - first_index / sr)


def read_station_channels(station: str, channels: list[int], iso_hour: str, start_unix: float, seconds: float, range_interp: int):
    start_second = station_start_second(station, channels[0], iso_hour, start_unix)
    zc = []
    sample_rate = None
    abs_index0 = None
    station_dir = Path(DATA_DIR) / station
    for ch in channels:
        files = rf_file_for_time(station_dir, ch, iso_hour)
        z, sr, idx0 = read_channel_files(files, start_second=start_second, seconds=seconds)
        zc.append(interpolate_rect(z, range_interp))
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
    return zc, float(sample_rate), int(abs_index0), float(start_second)


def estimate_modulo_background(z: np.ndarray, abs_index0: int, code_len_i: int, range_interp: int) -> np.ndarray:
    phases = (int(abs_index0) * int(range_interp) + np.arange(len(z), dtype=np.int64)) % int(code_len_i)
    counts = np.bincount(phases, minlength=code_len_i).astype(np.float64)
    real = np.bincount(phases, weights=z.real, minlength=code_len_i)
    imag = np.bincount(phases, weights=z.imag, minlength=code_len_i)
    bg = (real + 1j * imag) / np.maximum(counts, 1.0)
    return bg.astype(np.complex64)


def subtract_modulo_background(z: np.ndarray, bg: np.ndarray, abs_index0: int, range_interp: int) -> np.ndarray:
    phases = (int(abs_index0) * int(range_interp) + np.arange(len(z), dtype=np.int64)) % len(bg)
    return (z - bg[phases]).astype(np.complex64)


def process_station(
    station: str,
    rmin_km: float,
    rmax_km: float,
    args: argparse.Namespace,
) -> Path:
    site = load_station_yaml(DATA_DIR, station)
    code_len = int(site["code_len"])
    code_len_i = code_len * args.range_interp
    range_gate_km = float(site["range_resolution_m"]) / args.range_interp / 1000.0
    range_idx = np.arange(
        int(np.ceil(rmin_km / range_gate_km)),
        int(np.floor(rmax_km / range_gate_km)) + 1,
        dtype=np.int64,
    )
    rmax_idx = int(range_idx[-1])

    one_code_len = code_len_i
    fft_len = int(args.fft_codes) * code_len_i
    if fft_len < one_code_len:
        raise ValueError("--fft-codes must be at least 1")

    read_extra_s = ((args.n_incoh - 1) * code_len_i + one_code_len + rmax_idx + 1) / (args.sample_rate_hint * args.range_interp)
    read_seconds = (args.run_end_unix - args.run_start_unix) + read_extra_s + 0.05
    zc_raw, sample_rate, abs_index0, start_second = read_station_channels(
        station,
        args.channels,
        args.iso_hour,
        args.run_start_unix,
        read_seconds,
        args.range_interp,
    )
    sample_rate_i = sample_rate * args.range_interp
    read_extra_s = ((args.n_incoh - 1) * code_len_i + one_code_len + rmax_idx + 1) / sample_rate_i

    ref_zc, ref_sample_rate, ref_abs_index0, _ref_start_second = read_station_channels(
        station,
        args.channels,
        args.iso_hour,
        args.ref_start_unix,
        (args.ref_end_unix - args.ref_start_unix) + read_extra_s + 0.05,
        args.range_interp,
    )
    if abs(ref_sample_rate - sample_rate) > 0:
        raise ValueError(f"{station}: reference sample-rate mismatch")

    backgrounds = [
        estimate_modulo_background(z[: int(round((args.ref_end_unix - args.ref_start_unix) * sample_rate_i))], ref_abs_index0, code_len_i, args.range_interp)
        for z in ref_zc
    ]
    zc = [subtract_modulo_background(z, bg, abs_index0, args.range_interp) for z, bg in zip(zc_raw, backgrounds, strict=True)]

    codes = get_codes(site["seeds"], code_len, 1, range_interp=args.range_interp)
    fvec = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / sample_rate_i))
    dc_idx = int(np.argmin(np.abs(fvec)))
    step_len = code_len_i
    n_steps = int(np.floor((args.run_end_unix - args.run_start_unix) * sample_rate_i / step_len))

    out = args.product_dir / f"simple_rcs_rdmf_{station}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with h5py.File(tmp, "w") as h:
        h.attrs["description"] = "Simple 1-code coherent, 16-code incoherent RDMF for SNR/RCS diagnostics"
        h.attrs["station"] = station
        h.attrs["iso_hour"] = args.iso_hour
        h.attrs["start_second"] = start_second
        h.attrs["run_start_unix"] = float(args.run_start_unix)
        h.attrs["run_end_unix"] = float(args.run_end_unix)
        h.attrs["reference_start_unix"] = float(args.ref_start_unix)
        h.attrs["reference_end_unix"] = float(args.ref_end_unix)
        h.attrs["modulo_background"] = True
        h.attrs["n_coh"] = 1
        h.attrs["n_incoh"] = int(args.n_incoh)
        h.attrs["step_codes"] = 1
        h.attrs["fft_codes"] = int(args.fft_codes)
        h.attrs["dc_bin_hz"] = float(fvec[dc_idx])
        h.create_dataset("station", data=np.bytes_(station))
        h.create_dataset("channels", data=np.asarray(args.channels, dtype=np.int32))
        h.create_dataset("seeds", data=np.asarray(site["seeds"], dtype=np.int64))
        h.create_dataset("code_len", data=code_len)
        h.create_dataset("sample_rate", data=sample_rate)
        h.create_dataset("range_interp", data=args.range_interp)
        h.create_dataset("range_resolution_m", data=float(site["range_resolution_m"]))
        h.create_dataset("range_idx", data=range_idx.astype(np.int32))
        h.create_dataset("range_km", data=(range_idx * range_gate_km).astype(np.float32))
        h.create_dataset("fvec", data=fvec.astype(np.float32))
        h.create_dataset("t_sample", (n_steps,), dtype="f8")
        chunks = (min(128, n_steps), min(128, len(range_idx)))
        h.create_dataset("peak_power", (n_steps, len(range_idx)), dtype="f4", chunks=chunks)
        h.create_dataset("peak_doppler_hz", (n_steps, len(range_idx)), dtype="f4", chunks=chunks)
        h.create_dataset("dc_power", (n_steps, len(range_idx)), dtype="f4", chunks=chunks)

        norm = float(len(args.channels) * len(codes) * args.n_incoh)
        for ti in range(n_steps):
            start = ti * step_len
            S = np.zeros((len(range_idx), len(fvec)), dtype=np.float64)
            for incoh_i in range(args.n_incoh):
                offset = start + incoh_i * step_len
                read_len = one_code_len + rmax_idx + 1
                for chi, z in enumerate(zc):
                    z0 = detrend_window(z[offset : offset + read_len])
                    windows = np.lib.stride_tricks.sliding_window_view(z0, one_code_len)[range_idx, :]
                    for code in codes:
                        Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), n=fft_len, axis=1), axes=1)
                        S += np.abs(Z) ** 2
            S /= norm
            imax = np.argmax(S, axis=1)
            h["t_sample"][ti] = abs_index0 + ti * code_len
            h["peak_power"][ti, :] = S[np.arange(len(range_idx)), imax].astype(np.float32)
            h["peak_doppler_hz"][ti, :] = fvec[imax].astype(np.float32)
            h["dc_power"][ti, :] = S[:, dc_idx].astype(np.float32)
            if ti == 0 or ti == n_steps - 1 or (ti + 1) % args.progress_every == 0:
                print(f"{station}: {ti + 1}/{n_steps}", flush=True)

        t_unix = h["t_sample"][()] / sample_rate
        ref_keep = (t_unix >= args.ref_start_unix) & (t_unix < args.ref_end_unix)
        if not np.any(ref_keep):
            raise ValueError(f"{station}: no samples in reference interval")
        noise_ref = np.nanmedian(h["dc_power"][ref_keep, :], axis=0)
        h.create_dataset("noise_ref_power", data=np.asarray(noise_ref, dtype=np.float32))
        snr_db = 10.0 * np.log10(np.maximum(h["peak_power"][()] / np.maximum(noise_ref[None, :], 1e-30), 1e-30))
        h.create_dataset("snr_ref_db", data=np.asarray(snr_db, dtype=np.float32), chunks=chunks)
        h.attrs["complete"] = 1
    tmp.replace(out)
    print(f"Wrote {out}", flush=True)
    return out


def load_simple_station(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        return {
            "station": h["station"][()].decode(),
            "t": h["t_sample"][()] / float(h["sample_rate"][()]),
            "range_km": h["range_km"][()],
            "snr_db": h["snr_ref_db"][()],
            "peak_doppler_hz": h["peak_doppler_hz"][()],
            "noise_ref_power": h["noise_ref_power"][()],
        }


def load_trajectory_ranges(stations: list[str], times: np.ndarray) -> dict[str, np.ndarray]:
    if not TRAJECTORY_JSON.exists():
        return {}
    fit = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8"))
    geom = station_geometry(DATA_DIR, stations)
    tx_site = load_station_yaml(DATA_DIR, fit["used_stations"][0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    out = {}
    for station in stations:
        obs = [{"station": station, "unix_time": float(t)} for t in times]
        out[station] = predict_ranges(params, obs, geom, fit["t_ref_unix"], tx0, rot) / 1000.0
    return out


def plot_range_time(datasets: list[dict], out: Path) -> None:
    all_snr = np.concatenate([d["snr_db"].ravel() for d in datasets])
    vmin = max(-10.0, float(np.nanpercentile(all_snr, 2.0)))
    vmax = min(35.0, max(15.0, float(np.nanpercentile(all_snr, 99.8))))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.2), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, data in zip(axes, datasets, strict=True):
        mesh = ax.pcolormesh(
            unix_to_mpl(data["t"]),
            data["range_km"],
            data["snr_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{short_station_name(data['station'])}\nrange (km)")
        ax.set_ylim(float(data["range_km"][0]), float(data["range_km"][-1]))
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.axvspan(unix_to_mpl(np.array([REF_START_UNIX]))[0], unix_to_mpl(np.array([REF_END_UNIX]))[0], color="white", alpha=0.06, lw=0)
        ax.set_xlim(*unix_to_mpl(np.array([PLOT_START_UNIX, PLOT_END_UNIX])))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.suptitle("Simple RDMF peak-power SNR")
    cbar = fig.colorbar(mesh, ax=axes, label="Peak power / fixed DC-bin reference (dB)", shrink=0.93)
    cbar.ax.tick_params(labelsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_head_traces(datasets: list[dict], out: Path) -> None:
    by_station = {d["station"]: d for d in datasets}
    stations = [s for s in HEAD_TRACE_STATIONS if s in by_station]
    if not stations:
        return
    common_times = by_station[stations[0]]["t"]
    ranges = load_trajectory_ranges(stations, common_times)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    for station in stations:
        data = by_station[station]
        model_range = ranges.get(station)
        if model_range is None:
            continue
        snr = np.full_like(data["t"], np.nan, dtype=float)
        for i, r in enumerate(model_range):
            if r < data["range_km"][0] or r > data["range_km"][-1]:
                continue
            ri = int(np.argmin(np.abs(data["range_km"] - r)))
            snr[i] = data["snr_db"][i, ri]
        keep = (data["t"] >= EVENT_T0 - 1.0) & (data["t"] <= EVENT_T0 + 2.2)
        ax.plot(data["t"][keep] - EVENT_T0, snr[keep], lw=1.7, label=short_station_name(station))
    ax.set_xlabel("Seconds after 18:06:20 UTC")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Simple RDMF SNR along fitted head-echo range")
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.legend(ncol=3, loc="upper right", frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.product_dir.mkdir(parents=True, exist_ok=True)
    products = []
    for station, rmin, rmax in STATION_RANGES_KM:
        if station not in args.stations:
            continue
        path = args.product_dir / f"simple_rcs_rdmf_{station}.h5"
        if args.plot_only:
            products.append(path)
        else:
            products.append(process_station(station, rmin, rmax, args))
    if args.process_only:
        return
    datasets = [load_simple_station(path) for path in products]
    plot_range_time(datasets, args.range_time_fig)
    plot_head_traces(datasets, args.trace_fig)
    print(f"Wrote {args.range_time_fig}", flush=True)
    print(f"Wrote {args.trace_fig}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=[s for s, _r0, _r1 in STATION_RANGES_KM])
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--run-start-unix", type=float, default=RUN_START_UNIX)
    p.add_argument("--run-end-unix", type=float, default=RUN_END_UNIX)
    p.add_argument("--ref-start-unix", type=float, default=REF_START_UNIX)
    p.add_argument("--ref-end-unix", type=float, default=REF_END_UNIX)
    p.add_argument("--range-interp", type=int, default=1)
    p.add_argument("--n-incoh", type=int, default=16)
    p.add_argument("--fft-codes", type=int, default=2)
    p.add_argument("--sample-rate-hint", type=float, default=100000.0)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--product-dir", type=Path, default=PRODUCT_DIR)
    p.add_argument("--range-time-fig", type=Path, default=FIG_DIR / "station_range_time_simple_rcs_snr.png")
    p.add_argument("--trace-fig", type=Path, default=FIG_DIR / "head_echo_simple_rcs_snr_traces.png")
    p.add_argument("--process-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    return p


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
