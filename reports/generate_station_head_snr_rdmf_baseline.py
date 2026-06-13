#!/usr/bin/env python3
"""Generate the baseline station range-time SNR from stored RDMF products.

This is intentionally conservative: no polarization matching.  It plots the
stored incoherent RDMF baseline and the single next increment, a trajectory-
steered coherent TX-code sum with receive polarizations still incoherent.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from angle_departure_estimator import C, enu_to_ecef_matrix, geodetic_to_ecef, load_station_yaml  # noqa: E402
from reports.generate_station_head_rcs_figure import (  # noqa: E402
    HEAD_PLOT_END_UNIX,
    HEAD_PLOT_START_UNIX,
    STATION_LABELS,
    bin_edges,
    unix_to_mpl,
)

RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
DATA_DIR = ROOT / "2026_05_30_bolide"
DEFAULT_TRAJECTORY = (
    ROOT
    / "products"
    / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5"
    / "ballistic_head_trail_correlated_fit.json"
)
DEFAULT_EXTENT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_from_1s_noise_reference.h5"
OUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_rdmf_baseline.h5"
OUT_FIG = ROOT / "reports" / "figures" / "station_range_time_head_snr_rdmf_baseline.png"
OUT_TX_FIG = ROOT / "reports" / "figures" / "station_range_time_head_snr_rdmf_txsteer_incohpol.png"
OUT_TX_POL_FIG = ROOT / "reports" / "figures" / "station_range_time_head_snr_rdmf_txsteer_polmatch.png"
OUT_TX_POL_DOPPLER_FIG = ROOT / "reports" / "figures" / "station_head_doppler_rdmf_txsteer_polmatch.png"
DEFAULT_GAUSSIAN_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_v0p2p7_1coh_2incoh_zp16_sparse_dcmean.csv"
STATIONS = ("beverly", "clinton", "pomfret")


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload["model"]
    return {key: np.asarray(model[key], dtype=np.float64) for key in ("time_unix", "lat_deg", "lon_deg", "alt_m")}


def target_ecef(traj: dict[str, np.ndarray], time_unix: np.ndarray) -> np.ndarray:
    lat = np.interp(time_unix, traj["time_unix"], traj["lat_deg"])
    lon = np.interp(time_unix, traj["time_unix"], traj["lon_deg"])
    alt = np.interp(time_unix, traj["time_unix"], traj["alt_m"])
    return np.stack([geodetic_to_ecef(a, b, c) for a, b, c in zip(lat, lon, alt, strict=True)], axis=0)


def model_bistatic_range_km(h: h5py.File, traj: dict[str, np.ndarray], time_unix: np.ndarray) -> np.ndarray:
    target = target_ecef(traj, time_unix)
    tx_ecef = geodetic_to_ecef(*np.asarray(h["gps_tx"][()], dtype=float))
    rx_ecef = geodetic_to_ecef(*np.asarray(h["gps_rx"][()], dtype=float))
    return (np.linalg.norm(target - tx_ecef[None, :], axis=1) + np.linalg.norm(target - rx_ecef[None, :], axis=1)) / 1000.0


def steering_vectors(site: dict, traj: dict[str, np.ndarray], time_unix: np.ndarray, phase_sign: float) -> np.ndarray:
    tx_ecef = geodetic_to_ecef(*site["gps_tx"])
    rot = enu_to_ecef_matrix(site["gps_tx"][0], site["gps_tx"][1])
    vec_enu = (target_ecef(traj, time_unix) - tx_ecef[None, :]) @ rot
    unit = vec_enu / np.maximum(np.linalg.norm(vec_enu, axis=1)[:, None], 1e-30)
    wavelength = C / float(site["frequency_hz"])
    phase = phase_sign * (2.0 * np.pi / wavelength) * (unit @ np.asarray(site["antenna_pos_tx"], dtype=np.float64).T)
    return np.exp(1.0j * phase).astype(np.complex64)


def load_gaussian_summary(path: Path) -> dict[str, dict[str, np.ndarray]]:
    fields = (
        "time_unix",
        "fit_range_m",
        "fit_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "range_doppler_correlation",
    )
    rows: dict[str, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("success", "")).lower() != "true":
                continue
            station = str(row["station"])
            out = rows.setdefault(station, {key: [] for key in fields})
            try:
                vals = {key: float(row[key]) for key in fields}
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in vals.values()):
                continue
            for key, val in vals.items():
                out[key].append(val)
    return {
        station: {key: np.asarray(vals, dtype=np.float64) for key, vals in data.items()}
        for station, data in rows.items()
        if len(data["time_unix"]) > 0
    }


def match_gaussian_rows(
    station: str,
    time_unix: np.ndarray,
    gaussian: dict[str, dict[str, np.ndarray]],
    max_dt_s: float,
) -> dict[str, np.ndarray]:
    if station not in gaussian:
        raise ValueError(f"No Gaussian fit summary rows found for {station}")
    src = gaussian[station]
    order = np.argsort(src["time_unix"])
    src = {key: val[order] for key, val in src.items()}
    idx = np.searchsorted(src["time_unix"], time_unix)
    idx0 = np.clip(idx - 1, 0, len(src["time_unix"]) - 1)
    idx1 = np.clip(idx, 0, len(src["time_unix"]) - 1)
    use1 = np.abs(src["time_unix"][idx1] - time_unix) < np.abs(src["time_unix"][idx0] - time_unix)
    nearest = np.where(use1, idx1, idx0)
    dt = np.abs(src["time_unix"][nearest] - time_unix)
    out = {key: val[nearest] for key, val in src.items()}
    out["gaussian_valid"] = dt <= max_dt_s
    return out


def process_station(
    station: str,
    args: argparse.Namespace,
    traj: dict[str, np.ndarray],
    gaussian_summary: dict[str, dict[str, np.ndarray]],
) -> dict:
    path = args.rdmf_dir / f"rdmf_peak_{station}_merged.h5"
    site = load_station_yaml(DATA_DIR, station)
    with h5py.File(path, "r") as h:
        p_head = h["P_head"][()].astype(np.float64) if "P_head" in h else h["P"][()].astype(np.float64)
        noise = h["N"][()].astype(np.float64)
        d_head = h["D_head"][()].astype(np.float64) if "D_head" in h else h["D"][()].astype(np.float64)
        voltage = h["voltage_peak"][()].astype(np.complex64)
        time_unix = h["t_sample"][()].astype(np.float64) / float(h["sample_rate"][()])
        range_km = h["range_km"][()].astype(np.float64)
        model_range_km = model_bistatic_range_km(h, traj, time_unix)
    snr_db = 10.0 * np.log10(np.maximum(p_head / np.maximum(noise, 1e-30), 1e-30))
    voltage_power = np.sum(np.abs(voltage) ** 2, axis=(1, 3), dtype=np.float64) / float(voltage.shape[1] * voltage.shape[3])
    rdmf_power_scale = p_head / np.maximum(voltage_power, 1e-30)
    steering = steering_vectors(site, traj, time_unix, float(args.phase_sign))
    if voltage.shape[-1] != steering.shape[-1]:
        raise ValueError(f"{station}: voltage has {voltage.shape[-1]} TX channels but steering has {steering.shape[-1]}")
    tx_beam = np.einsum("tcrx,tx->tcr", voltage, np.conj(steering), optimize=True)
    txsteer_raw_power = np.sum(np.abs(tx_beam) ** 2, axis=1, dtype=np.float64) / float(voltage.shape[1] * voltage.shape[3])
    txsteer_power = txsteer_raw_power * rdmf_power_scale
    txsteer_snr_db = 10.0 * np.log10(np.maximum(txsteer_power / np.maximum(noise, 1e-30), 1e-30))
    fit = match_gaussian_rows(station, time_unix, gaussian_summary, float(args.gaussian_max_dt_s))
    range_m = range_km * 1000.0
    path_range_idx = np.argmin(np.abs(range_km[None, :] - model_range_km[:, None]), axis=1)
    txsteer_pol_raw_power = np.empty_like(txsteer_raw_power)
    peak_range_idx = np.empty(len(time_unix), dtype=np.int32)
    pol_gamma = np.empty((len(time_unix), voltage.shape[1]), dtype=np.complex64)
    for ti in range(len(time_unix)):
        if bool(fit["gaussian_valid"][ti]):
            sig_r = max(float(fit["sigma_range_m"][ti]), 1e-12)
            sig_d = max(float(fit["sigma_doppler_hz"][ti]), 1e-12)
            rho = float(np.clip(fit["range_doppler_correlation"][ti], -0.95, 0.95))
            rr = (range_m - float(fit["fit_range_m"][ti])) / sig_r
            dd = (d_head[ti] - float(fit["fit_doppler_hz"][ti])) / sig_d
            q = (rr * rr - 2.0 * rho * rr * dd + dd * dd) / max(1.0 - rho * rho, 1e-12)
            mask = np.isfinite(q) & (q <= 1.0) & np.isfinite(txsteer_raw_power[ti])
        else:
            mask = np.zeros_like(range_m, dtype=bool)
            mask[path_range_idx[ti]] = True
        if not np.any(mask):
            mask = np.isfinite(txsteer_raw_power[ti])
        ri = int(np.nanargmax(np.where(mask, txsteer_raw_power[ti], -np.inf)))
        peak_range_idx[ti] = ri
        peak = tx_beam[ti, :, ri].astype(np.complex128)
        gamma = peak / max(np.linalg.norm(peak), 1e-30)
        pol_gamma[ti] = gamma.astype(np.complex64)
        combined = np.einsum("c,cr->r", np.conj(gamma), tx_beam[ti], optimize=True)
        txsteer_pol_raw_power[ti] = np.abs(combined) ** 2 / float(voltage.shape[3])
    txsteer_pol_power = txsteer_pol_raw_power * rdmf_power_scale
    txsteer_pol_snr_db = 10.0 * np.log10(np.maximum(txsteer_pol_power / np.maximum(noise, 1e-30), 1e-30))
    row_idx = np.arange(len(time_unix), dtype=np.int64)
    selected_doppler_hz = d_head[row_idx, peak_range_idx]
    selected_txsteer_pol_snr_db = txsteer_pol_snr_db[row_idx, peak_range_idx]
    finite_ranges = range_km[np.any(np.isfinite(snr_db), axis=0)]
    rmin = float(np.nanmin(finite_ranges)) if finite_ranges.size else float(range_km[0])
    rmax = float(np.nanmax(finite_ranges)) if finite_ranges.size else float(range_km[-1])
    pad = max(1.0, 0.025 * (rmax - rmin))
    return {
        "station": station,
        "time_unix": time_unix,
        "range_km": range_km,
        "snr_db": snr_db,
        "txsteer_snr_db": txsteer_snr_db,
        "txsteer_pol_snr_db": txsteer_pol_snr_db,
        "model_range_km": model_range_km,
        "peak_range_idx": peak_range_idx,
        "selected_doppler_hz": selected_doppler_hz,
        "selected_txsteer_pol_snr_db": selected_txsteer_pol_snr_db,
        "fit_doppler_hz": fit["fit_doppler_hz"],
        "pol_gamma": pol_gamma,
        "gaussian_valid": fit["gaussian_valid"].astype(np.uint8),
        "rmin_km": rmin - pad,
        "rmax_km": rmax + pad,
    }


def save_results(datasets: list[dict], args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h:
        h.attrs["description"] = "RDMF head-echo SNR baseline, trajectory-steered TX sum, and TX-selected polarization match"
        h.attrs["rdmf_dir"] = str(args.rdmf_dir)
        h.attrs["trajectory_json"] = str(args.trajectory_json)
        h.attrs["gaussian_summary_csv"] = str(args.gaussian_summary_csv)
        h.attrs["snr_definition"] = "baseline: stored P_head/N; txsteer: trajectory-steered coherent TX sum in stored RDMF power units divided by stored N, receive polarizations still incoherent; txsteer_pol: pixel selected from txsteer inside Gaussian range-Doppler area, then receive polarization matched at that pixel"
        h.attrs["phase_sign"] = float(args.phase_sign)
        h.create_dataset("stations", data=np.asarray([d["station"].encode("utf-8") for d in datasets]))
        for data in datasets:
            g = h.create_group(data["station"])
            for key in (
                "time_unix",
                "range_km",
                "snr_db",
                "txsteer_snr_db",
                "txsteer_pol_snr_db",
                "model_range_km",
                "peak_range_idx",
                "selected_doppler_hz",
                "selected_txsteer_pol_snr_db",
                "fit_doppler_hz",
                "pol_gamma",
                "gaussian_valid",
            ):
                g.create_dataset(key, data=data[key])
            g.attrs["rmin_km"] = data["rmin_km"]
            g.attrs["rmax_km"] = data["rmax_km"]


def load_results(path: Path) -> list[dict]:
    datasets = []
    with h5py.File(path, "r") as h:
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            datasets.append(
                {
                    "station": station,
                    "time_unix": g["time_unix"][()],
                    "range_km": g["range_km"][()],
                    "snr_db": g["snr_db"][()],
                    "txsteer_snr_db": g["txsteer_snr_db"][()],
                    "txsteer_pol_snr_db": g["txsteer_pol_snr_db"][()],
                    "model_range_km": g["model_range_km"][()],
                    "selected_doppler_hz": g["selected_doppler_hz"][()],
                    "selected_txsteer_pol_snr_db": g["selected_txsteer_pol_snr_db"][()],
                    "fit_doppler_hz": g["fit_doppler_hz"][()],
                    "gaussian_valid": g["gaussian_valid"][()],
                    "rmin_km": float(g.attrs["rmin_km"]),
                    "rmax_km": float(g.attrs["rmax_km"]),
                }
            )
    return datasets


def load_extent_reference(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    out = {}
    with h5py.File(path, "r") as h:
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            out[station] = (float(g.attrs["rmin_km"]), float(g.attrs["rmax_km"]))
    return out


def plot_results(datasets: list[dict], args: argparse.Namespace, key: str, out_fig: Path, label: str, title: str) -> None:
    extents = load_extent_reference(args.extent_reference_h5)
    vmin = float(args.vmin_db)
    vmax = float(args.vmax_db)
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.90")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, data in zip(axes, datasets, strict=True):
        x = unix_to_mpl(data["time_unix"])
        station = str(data["station"])
        rmin_km, rmax_km = extents.get(station, (data["rmin_km"], data["rmax_km"]))
        keep = (data["range_km"] >= rmin_km) & (data["range_km"] <= rmax_km)
        ranges = data["range_km"][keep]
        image = data[key][:, keep]
        mesh = ax.pcolormesh(bin_edges(x), bin_edges(ranges), image.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(x, data["model_range_km"], color="white", linestyle="--", linewidth=1.0, alpha=0.20)
        ax.set_ylabel(f"{STATION_LABELS.get(station, station)}\nrange (km)")
        ax.set_ylim(rmin_km, rmax_km)
        ax.grid(True, color="0.9", linewidth=0.4)
    for ax in axes:
        ax.set_xlim(unix_to_mpl(np.array([HEAD_PLOT_START_UNIX, HEAD_PLOT_END_UNIX])))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].text(
        0.012,
        0.94,
        title,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
        alpha=0.65,
    )
    axes[-1].set_xlabel("UTC on 2026-05-30")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label=label, shrink=0.93)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_selected_doppler(datasets: list[dict], out_fig: Path, vmin_db: float, vmax_db: float) -> None:
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.94")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 6.8), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    ylim = (-4.0, 1.0)
    doppler_edges = np.linspace(ylim[0], ylim[1], 251)
    doppler_centers = 0.5 * (doppler_edges[:-1] + doppler_edges[1:])
    mesh = None

    for ax, data in zip(axes, datasets, strict=True):
        x = unix_to_mpl(data["time_unix"])
        station = str(data["station"])
        valid = np.asarray(data["gaussian_valid"], dtype=bool)
        doppler = np.where(valid, data["fit_doppler_hz"], np.nan) / 1000.0
        snr = np.asarray(data["selected_txsteer_pol_snr_db"], dtype=np.float64)
        image = np.full((len(x), len(doppler_centers)), np.nan, dtype=np.float64)
        good = valid & np.isfinite(doppler) & np.isfinite(snr) & (doppler >= ylim[0]) & (doppler <= ylim[1])
        row_idx = np.flatnonzero(good)
        col_idx = np.searchsorted(doppler_edges, doppler[good], side="right") - 1
        col_idx = np.clip(col_idx, 0, len(doppler_centers) - 1)
        image[row_idx, col_idx] = snr[good]
        mesh = ax.pcolormesh(
            bin_edges(x),
            doppler_edges,
            image.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
        )
        ax.set_ylabel(f"{STATION_LABELS.get(station, station)}\nDoppler (kHz)")
        ax.grid(True, color="0.88", linewidth=0.5)
        ax.set_ylim(*ylim)
    for ax in axes:
        ax.set_xlim(unix_to_mpl(np.array([HEAD_PLOT_START_UNIX, HEAD_PLOT_END_UNIX])))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].text(
        0.012,
        0.94,
        "Gaussian-refined Doppler; color is TX-steered matched-pol SNR",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
        alpha=0.65,
    )
    axes[-1].set_xlabel("UTC on 2026-05-30")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="Selected matched-pol SNR (dB)", shrink=0.93)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--rdmf-dir", type=Path, default=RDMF_DIR)
    p.add_argument("--trajectory-json", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--output-h5", type=Path, default=OUT_H5)
    p.add_argument("--output-fig", type=Path, default=OUT_FIG)
    p.add_argument("--output-tx-fig", type=Path, default=OUT_TX_FIG)
    p.add_argument("--output-tx-pol-fig", type=Path, default=OUT_TX_POL_FIG)
    p.add_argument("--output-tx-pol-doppler-fig", type=Path, default=OUT_TX_POL_DOPPLER_FIG)
    p.add_argument("--extent-reference-h5", type=Path, default=DEFAULT_EXTENT_H5)
    p.add_argument("--gaussian-summary-csv", type=Path, default=DEFAULT_GAUSSIAN_SUMMARY)
    p.add_argument("--gaussian-max-dt-s", type=float, default=0.03)
    p.add_argument("--phase-sign", type=float, default=-1.0)
    p.add_argument("--vmin-db", type=float, default=2.0)
    p.add_argument("--vmax-db", type=float, default=15.0)
    args = p.parse_args()
    if args.mode in {"compute", "both"}:
        traj = load_trajectory(args.trajectory_json)
        gaussian_summary = load_gaussian_summary(args.gaussian_summary_csv)
        datasets = [process_station(station, args, traj, gaussian_summary) for station in args.stations]
        save_results(datasets, args)
        print(args.output_h5, flush=True)
    if args.mode in {"plot", "both"}:
        datasets = load_results(args.output_h5)
        plot_results(
            datasets,
            args,
            "snr_db",
            args.output_fig,
            "Baseline RDMF SNR (dB)",
            "Stored RDMF P_head/N; incoherent RX polarization and TX-code average",
        )
        print(args.output_fig, flush=True)
        plot_results(
            datasets,
            args,
            "txsteer_snr_db",
            args.output_tx_fig,
            "Trajectory-steered TX SNR (dB)",
            "Trajectory-steered coherent TX-code sum; receive polarizations incoherent",
        )
        print(args.output_tx_fig, flush=True)
        plot_results(
            datasets,
            args,
            "txsteer_pol_snr_db",
            args.output_tx_pol_fig,
            "Trajectory-steered TX + matched-pol SNR (dB)",
            "TX-steered peak-pixel selection; receive polarization matched after selection",
        )
        print(args.output_tx_pol_fig, flush=True)
        plot_selected_doppler(datasets, args.output_tx_pol_doppler_fig, float(args.vmin_db), float(args.vmax_db))
        print(args.output_tx_pol_doppler_fig, flush=True)


if __name__ == "__main__":
    main()
