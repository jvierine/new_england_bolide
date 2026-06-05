#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from range_doppler_mf_bolide import bpsk, read_config, read_channel_files, rf_file_for_time
from station_config import STATION_ORDER, short_station_name

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml
from estimate_head_echo_trajectory import model_state_series, predict_ranges, station_geometry


DEFAULT_DATA_DIR = Path("2026_05_30_bolide")
DEFAULT_PRODUCT_DIR = Path("products/trail_echo_deconv")
DEFAULT_FIG_DIR = Path("reports/figures")
DEFAULT_TRAJECTORY_JSON = Path("products/head_echo_trajectory_roi_fit/head_echo_trajectory.json")
TRAIL_RANGE_WINDOWS_KM = {
    "beverly": (180.0, 275.0),
    "clinton": (220.0, 280.0),
    "pomfret": (350.0, 420.0),
    "siena": (400.0, 500.0),
    "amherstnh": (200.0, 350.0),
}


def periodic_convolution_matrix(code: np.ndarray, rmin: int, rmax: int) -> np.ndarray:
    ridx = np.arange(rmin, rmax + 1, dtype=np.int64)
    rows = np.arange(len(code), dtype=np.int64)[:, None]
    return code[(rows - ridx[None, :]) % len(code)].astype(np.complex64)


def cached_estimation_matrix(
    code: np.ndarray,
    rmin: int,
    rmax: int,
    cache_dir: Path,
    cache_key: str,
    rcond: float,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}_r{rmin:04d}_{rmax:04d}_rcond{rcond:.0e}.h5"
    if path.exists():
        with h5py.File(path, "r") as h:
            return h["B"][()].astype(np.complex64)

    a = periodic_convolution_matrix(code, rmin, rmax)
    b = np.linalg.pinv(a, rcond=rcond).astype(np.complex64)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as h:
        h["B"] = b
        h["code"] = code.astype(np.complex64)
        h.attrs["rmin_idx"] = int(rmin)
        h.attrs["rmax_idx"] = int(rmax)
        h.attrs["rcond"] = float(rcond)
        h.attrs["description"] = "Stationary-target periodic coded-CW convolution pseudoinverse."
    tmp.replace(path)
    return b


def unix_to_mpl(t: np.ndarray) -> np.ndarray:
    ns = np.asarray(np.asarray(t, dtype=float) * 1e9, dtype="datetime64[ns]")
    return mdates.date2num(ns.astype("datetime64[us]").astype(object))


def robust_db(power: np.ndarray) -> np.ndarray:
    floor = np.nanmedian(power, axis=1, keepdims=True)
    return 10.0 * np.log10(np.maximum(power / np.maximum(floor, 1e-20), 1e-12))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def process_station(args: argparse.Namespace, station: str) -> Path:
    if args.code_chunk % args.coherent_codes != 0:
        raise ValueError("--code-chunk must be an integer multiple of --coherent-codes")
    cfg = read_config(args.data_dir, station)
    rmin = cfg["rmin_idx"] if args.rmin_idx is None else args.rmin_idx
    rmax = cfg["rmax_idx"] if args.rmax_idx is None else args.rmax_idx
    if args.max_range_km is not None:
        rmax = min(rmax, int(round(args.max_range_km / (args.range_resolution_m / 1000.0))))
    ranges_km = np.arange(rmin, rmax + 1, dtype=np.float32) * (args.range_resolution_m / 1000.0)

    station_dir = Path(args.data_dir) / station
    channel_data = []
    sample_rate = None
    abs_index0 = None
    for ch in args.channels:
        files = rf_file_for_time(station_dir, ch, args.iso_hour)
        z, sr, idx0 = read_channel_files(files, start_second=args.start_second, seconds=args.seconds)
        n_code_samples = (len(z) // cfg["code_len"]) * cfg["code_len"]
        z = z[:n_code_samples]
        channel_data.append(z.reshape(-1, cfg["code_len"]))
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
        if sample_rate != sr or abs_index0 != idx0:
            raise ValueError(f"{station}: channel {ch} time base mismatch")

    n_codes_total = min(z.shape[0] for z in channel_data)
    if args.max_codes is not None:
        n_codes_total = min(n_codes_total, args.max_codes)
    n_blocks = n_codes_total // args.coherent_codes
    if n_blocks <= 0:
        raise ValueError(f"{station}: not enough data for {args.coherent_codes} code coherent integrations")
    n_codes_used = n_blocks * args.coherent_codes
    channel_data = [z[:n_codes_used] for z in channel_data]

    dt_code = cfg["code_len"] / float(sample_rate)
    t_block = (abs_index0 + (np.arange(n_blocks) * args.coherent_codes + 0.5 * args.coherent_codes) * cfg["code_len"]) / float(sample_rate)
    t_code = (abs_index0 + (np.arange(n_codes_used) + 0.5) * cfg["code_len"]) / float(sample_rate)
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(args.coherent_codes, d=dt_code)).astype(np.float32)
    window = np.hanning(args.coherent_codes).astype(np.float32)
    window_norm = float(np.sum(window**2))

    n_ch = len(args.channels)
    n_tx = len(cfg["seeds"])
    n_range = len(ranges_km)
    spectra_power = np.zeros((n_blocks, n_range, len(doppler_hz)), dtype=np.float32)
    peak_power = np.zeros((n_blocks, n_range), dtype=np.float32)
    peak_doppler = np.zeros((n_blocks, n_range), dtype=np.float32)
    mean_voltage = np.zeros((n_blocks, n_range), dtype=np.complex64)

    out_path = Path(args.product_dir) / f"trail_deconv_{station}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    h5 = h5py.File(tmp_path, "w")
    code_voltage_ds = None
    if args.store_code_voltage:
        code_voltage_ds = h5.create_dataset(
            "code_voltage",
            (n_codes_used, n_ch, n_tx, n_range),
            dtype="c8",
            chunks=(min(512, n_codes_used), 1, 1, n_range),
            compression="gzip",
            compression_opts=1,
        )

    print(f"{station}: deconvolving {n_codes_used} codes, {n_range} ranges, {n_ch} channels, {n_tx} tx codes", flush=True)
    t0 = time.time()
    for txi, seed in enumerate(cfg["seeds"]):
        code = bpsk(int(seed), cfg["code_len"])
        b = cached_estimation_matrix(
            code,
            rmin,
            rmax,
            Path(args.product_dir) / "pinv_cache",
            f"seed{int(seed)}_n{cfg['code_len']}",
            args.pinv_rcond,
        )
        for chi, z_code in enumerate(channel_data):
            for i0 in range(0, n_codes_used, args.code_chunk):
                i1 = min(n_codes_used, i0 + args.code_chunk)
                v = (b @ z_code[i0:i1].T).T.astype(np.complex64)
                if code_voltage_ds is not None:
                    code_voltage_ds[i0:i1, chi, txi, :] = v
                for b0 in range(i0 // args.coherent_codes, (i1 - 1) // args.coherent_codes + 1):
                    c0 = b0 * args.coherent_codes
                    c1 = c0 + args.coherent_codes
                    if c0 < i0 or c1 > i1:
                        continue
                    block = v[c0 - i0 : c1 - i0, :]
                    spec = np.fft.fftshift(np.fft.fft(block * window[:, None], axis=0), axes=0)
                    spectra_power[b0] += (np.abs(spec.T) ** 2 / (window_norm * n_ch * n_tx)).astype(np.float32)
                    mean_voltage[b0] += np.mean(block, axis=0).astype(np.complex64) / float(n_ch * n_tx)
        print(f"{station}: tx seed {seed} done ({time.time() - t0:.1f}s)", flush=True)

    if args.doppler_abs_max_hz is not None:
        keep_f = np.abs(doppler_hz) <= args.doppler_abs_max_hz
    else:
        keep_f = np.ones_like(doppler_hz, dtype=bool)
    search = np.where(keep_f[None, None, :], spectra_power, -np.inf)
    imax = np.argmax(search, axis=2)
    rr = np.arange(n_range)
    for bi in range(n_blocks):
        peak_power[bi, :] = spectra_power[bi, rr, imax[bi]]
        peak_doppler[bi, :] = doppler_hz[imax[bi]]

    h5["spectra_power"] = spectra_power
    h5["peak_power"] = peak_power
    h5["peak_doppler_hz"] = peak_doppler
    h5["mean_voltage"] = mean_voltage
    h5["time_unix"] = t_block.astype(np.float64)
    h5["code_time_unix"] = t_code.astype(np.float64)
    h5["range_km"] = ranges_km.astype(np.float32)
    h5["range_idx"] = np.arange(rmin, rmax + 1, dtype=np.int32)
    h5["doppler_hz"] = doppler_hz
    h5["channels"] = np.asarray(args.channels, dtype=np.int32)
    h5["seeds"] = np.asarray(cfg["seeds"], dtype=np.int64)
    h5["sample_rate"] = float(sample_rate)
    h5["t0_sample"] = int(abs_index0)
    h5["code_len"] = int(cfg["code_len"])
    h5["coherent_codes"] = int(args.coherent_codes)
    h5["range_resolution_m"] = float(args.range_resolution_m)
    h5["station"] = np.bytes_(station)
    h5.attrs["method"] = "Stationary-target per-transmit-code periodic convolution pseudoinverse, followed by 100-code windowed Doppler FFT."
    h5.attrs["voltage_average_note"] = "Diagnostic powers average over receive channels and transmit seeds; optional code_voltage retains per-channel/per-seed complex range voltages."
    h5.close()
    tmp_path.replace(out_path)
    print(f"{station}: wrote {out_path}", flush=True)
    return out_path


def load_product(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        peak_power = h["peak_power"][()]
        snr_db = robust_db(peak_power)
        return {
            "station": h["station"][()].decode(),
            "time_unix": h["time_unix"][()],
            "range_km": h["range_km"][()],
            "snr_db": snr_db,
            "peak_doppler_hz": h["peak_doppler_hz"][()],
        }


def trajectory_range_limits(
    datasets: list[dict],
    data_dir: Path,
    trajectory_json: Path,
    altitude_min_km: float,
    altitude_max_km: float,
    range_pad_km: float,
) -> dict[str, tuple[float, float]]:
    if not trajectory_json.exists():
        return {}
    fit = json.loads(trajectory_json.read_text(encoding="utf-8"))
    stations = [d["station"] for d in datasets]
    geom = station_geometry(data_dir, stations)
    tx_site = load_station_yaml(data_dir, fit["used_stations"][0])
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]

    limits = {}
    for data in datasets:
        t = np.asarray(data["time_unix"], dtype=float)
        if len(t) < 2:
            continue
        t_grid = np.linspace(float(t[0]), float(t[-1]), max(400, 4 * len(t)))
        state = model_state_series(params, t_grid, fit["t_ref_unix"], tx0, rot)
        alt_km = state["lat_lon_alt"][:, 2] / 1000.0
        keep_alt = (alt_km >= altitude_min_km) & (alt_km <= altitude_max_km)
        if not np.any(keep_alt):
            continue
        model_obs = [{"station": data["station"], "unix_time": float(tt)} for tt in t_grid]
        range_km = predict_ranges(params, model_obs, geom, fit["t_ref_unix"], tx0, rot) / 1000.0
        r0 = float(np.nanmin(range_km[keep_alt]) - range_pad_km)
        r1 = float(np.nanmax(range_km[keep_alt]) + range_pad_km)
        r_data0 = float(np.nanmin(data["range_km"]))
        r_data1 = float(np.nanmax(data["range_km"]))
        limits[data["station"]] = (max(r_data0, r0), min(r_data1, r1))
    return limits


def crop_for_station(data: dict, limits: dict[str, tuple[float, float]]) -> tuple[np.ndarray, tuple[float, float]]:
    if data["station"] in limits:
        r0, r1 = limits[data["station"]]
    else:
        r0, r1 = float(data["range_km"][0]), float(data["range_km"][-1])
    keep = (data["range_km"] >= min(r0, r1)) & (data["range_km"] <= max(r0, r1))
    return keep, (min(r0, r1), max(r0, r1))


def plot_rti(
    products: list[Path],
    fig_dir: Path,
    data_dir: Path,
    trajectory_json: Path,
    altitude_min_km: float,
    altitude_max_km: float,
    range_pad_km: float,
) -> Path:
    data = [load_product(p) for p in products]
    limits = trajectory_range_limits(data, data_dir, trajectory_json, altitude_min_km, altitude_max_km, range_pad_km)
    vals = []
    for d in data:
        keep_r, _ = crop_for_station(d, limits)
        vals.append(d["snr_db"][:, keep_r].ravel())
    vals = np.concatenate(vals)
    vmin = max(-8.0, float(np.nanpercentile(vals, 2.0)))
    vmax = min(30.0, float(np.nanpercentile(vals, 99.5)))
    fig, axes = plt.subplots(len(data), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    if len(data) == 1:
        axes = [axes]
    mesh = None
    for ax, d in zip(axes, data, strict=True):
        keep_r, ylim = crop_for_station(d, limits)
        mesh = ax.pcolormesh(
            unix_to_mpl(d["time_unix"]),
            d["range_km"][keep_r],
            d["snr_db"][:, keep_r].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
        ax.set_ylim(ylim)
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes, label="Peak Doppler power / median range power (dB)", shrink=0.92)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "trail_deconv_range_time_intensity.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trail_doppler_rti(
    products: list[Path],
    fig_dir: Path,
    snr_threshold_db: float,
    data_dir: Path,
    trajectory_json: Path,
    altitude_min_km: float,
    altitude_max_km: float,
    range_pad_km: float,
    doppler_color_abs_hz: float,
) -> Path:
    data = [load_product(p) for p in products]
    limits = trajectory_range_limits(data, data_dir, trajectory_json, altitude_min_km, altitude_max_km, range_pad_km)
    fig, axes = plt.subplots(len(data), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    if len(data) == 1:
        axes = [axes]
    mesh = None
    for ax, d in zip(axes, data, strict=True):
        keep_r, ylim = crop_for_station(d, limits)
        doppler = d["peak_doppler_hz"][:, keep_r].astype(float)
        doppler = np.where(d["snr_db"][:, keep_r] >= snr_threshold_db, doppler, np.nan)
        mesh = ax.pcolormesh(
            unix_to_mpl(d["time_unix"]),
            d["range_km"][keep_r],
            doppler.T,
            shading="auto",
            cmap="coolwarm",
            vmin=-doppler_color_abs_hz,
            vmax=doppler_color_abs_hz,
        )
        ax.set_ylabel(f"{short_station_name(d['station'])}\nrange (km)")
        ax.set_ylim(ylim)
        ax.grid(True, color="0.9", linewidth=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes, label="Peak Doppler shift (Hz)", shrink=0.92)
    out = fig_dir / "trail_deconv_peak_doppler_time.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stationary-target trail echo deconvolution and 100-code Doppler products.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--stations", nargs="+", default=STATION_ORDER)
    p.add_argument("--iso-hour", default="2026-05-30T18-00-00")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    p.add_argument("--start-second", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--coherent-codes", type=int, default=100)
    p.add_argument("--code-chunk", type=int, default=1000)
    p.add_argument("--range-resolution-m", type=float, default=3000.0)
    p.add_argument("--rmin-idx", type=int, default=None)
    p.add_argument("--rmax-idx", type=int, default=None)
    p.add_argument("--max-range-km", type=float, default=None)
    p.add_argument("--pinv-rcond", type=float, default=1e-4)
    p.add_argument("--doppler-abs-max-hz", type=float, default=50.0)
    p.add_argument("--doppler-snr-threshold-db", type=float, default=6.0)
    p.add_argument("--doppler-color-abs-hz", type=float, default=12.0)
    p.add_argument("--trajectory-json", default=str(DEFAULT_TRAJECTORY_JSON))
    p.add_argument("--altitude-min-km", type=float, default=10.0)
    p.add_argument("--altitude-max-km", type=float, default=150.0)
    p.add_argument("--trajectory-range-pad-km", type=float, default=9.0)
    p.add_argument("--store-code-voltage", action="store_true")
    p.add_argument("--product-dir", default=str(DEFAULT_PRODUCT_DIR))
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    product_dir = Path(args.product_dir)
    if args.plot_only:
        products = [product_dir / f"trail_deconv_{station}.h5" for station in args.stations]
    else:
        products = [process_station(args, station) for station in args.stations]
    fig_dir = Path(args.fig_dir)
    data_dir = resolve_path(args.data_dir)
    trajectory_json = resolve_path(args.trajectory_json)
    print(
        f"wrote {plot_rti(products, fig_dir, data_dir, trajectory_json, args.altitude_min_km, args.altitude_max_km, args.trajectory_range_pad_km)}"
    )
    print(
        "wrote "
        f"{plot_trail_doppler_rti(products, fig_dir, args.doppler_snr_threshold_db, data_dir, trajectory_json, args.altitude_min_km, args.altitude_max_km, args.trajectory_range_pad_km, args.doppler_color_abs_hz)}"
    )


if __name__ == "__main__":
    main()
