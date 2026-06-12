#!/usr/bin/env python3
"""Generate station range-time head-echo SNR from the 100 kHz noise reference."""

from __future__ import annotations

import argparse
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

from reports.generate_station_head_rcs_figure import (  # noqa: E402
    HEAD_PLOT_END_UNIX,
    HEAD_PLOT_START_UNIX,
    STATION_LABELS,
    bin_edges,
    unix_to_mpl,
)

DEFAULT_PATCH_H5 = (
    ROOT
    / "products"
    / "head_echo_gaussian_patches"
    / "head_echo_gaussian_patches_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_retained_ext0p5.h5"
)
DEFAULT_NOISE_H5 = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_noise_power_1s.h5"
OUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_from_1s_noise_reference.h5"
OUT_FIG = ROOT / "reports" / "figures" / "station_range_time_head_snr_noise_reference.png"


def load_noise_reference(path: Path) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    with h5py.File(path, "r") as h:
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            group = h[station]
            out[station] = {
                "time_unix": group["time_unix"][()],
                "dc_subtracted_power": group["dc_subtracted_power"][()],
            }
    return out


def nearest_noise_power(noise: dict[str, np.ndarray], time_unix: float) -> float:
    idx = int(np.argmin(np.abs(noise["time_unix"] - float(time_unix))))
    return float(noise["dc_subtracted_power"][idx])


def patch_noise_gain(patch: h5py.Group, fallback_n_channels: int) -> tuple[float, bool]:
    """Noise gain relative to the channel-averaged 100 kHz noise reference."""
    if "polarization_gamma" not in patch:
        return float(fallback_n_channels), False
    gamma = np.asarray(patch["polarization_gamma"][()], dtype=np.complex128)
    equalizer = np.asarray(patch.get("channel_noise_equalizer", []), dtype=float)
    powers = np.asarray(patch.get("channel_noise_powers", []), dtype=float)
    if (
        gamma.size == 0
        or equalizer.size != gamma.size
        or powers.size != gamma.size
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(equalizer))
        or not np.all(np.isfinite(powers))
        or np.nanmean(powers) <= 0.0
    ):
        return float(fallback_n_channels), False
    gain = float(np.sum(np.abs(gamma) ** 2 * equalizer**2 * powers) / np.nanmean(powers))
    return max(gain, 1e-12), True


def process_station(
    patch_h5: h5py.File,
    station: str,
    noise: dict[str, np.ndarray],
    *,
    n_codes: int,
    code_len: int,
    n_channels: int,
) -> tuple[dict, dict]:
    group = patch_h5[station]
    patch_names = [name for name in sorted(group) if name.startswith("patch_") and bool(group[name].attrs.get("valid", False))]
    if not patch_names:
        raise ValueError(f"{station}: no valid patch groups in {patch_h5.filename}")

    ranges_m = np.unique(
        np.concatenate([np.rint(np.asarray(group[name]["range_m"][()], dtype=float)).astype(np.int64) for name in patch_names])
    )
    times = np.array([float(group[name].attrs["time_unix"]) for name in patch_names], dtype=float)
    snr_db = np.full((len(patch_names), len(ranges_m)), np.nan, dtype=float)
    noise_100khz = np.full(len(patch_names), np.nan, dtype=float)
    noise_100hz = np.full(len(patch_names), np.nan, dtype=float)
    noise_rdmf = np.full(len(patch_names), np.nan, dtype=float)
    noise_gain = np.full(len(patch_names), np.nan, dtype=float)
    polarization_matched = np.full(len(patch_names), False, dtype=bool)

    for ti, name in enumerate(patch_names):
        patch = group[name]
        patch_ranges = np.rint(np.asarray(patch["range_m"][()], dtype=float)).astype(np.int64)
        raw_power = 10.0 ** (np.asarray(patch["raw_db"][()], dtype=float) / 10.0)
        peak_power = np.nanmax(raw_power, axis=1)
        n_incoh = int(patch.attrs["n_incoh"])
        noise_100khz[ti] = nearest_noise_power(noise, times[ti])
        noise_100hz[ti] = noise_100khz[ti] / 1000.0
        # The 1 s reference is channel-averaged 100 kHz voltage power. For
        # uncombined patches the RDMF image sums over channels, so the noise
        # gain is n_channels. For polarization-combined patches the RDMF image
        # has one combined stream; its effective noise gain is determined from
        # gamma, channel equalization, and the stored channel-noise powers.
        noise_gain[ti], polarization_matched[ti] = patch_noise_gain(patch, n_channels)
        noise_rdmf[ti] = noise_100khz[ti] * code_len * noise_gain[ti] * n_codes * n_incoh
        cols = np.searchsorted(ranges_m, patch_ranges)
        snr_db[ti, cols] = 10.0 * np.log10(np.maximum(peak_power / max(noise_rdmf[ti], 1e-300), 1e-300))

    ranges_km = ranges_m.astype(float) / 1000.0
    finite_ranges = ranges_km[np.any(np.isfinite(snr_db), axis=0)]
    if finite_ranges.size:
        rmin_km = float(np.nanmin(finite_ranges))
        rmax_km = float(np.nanmax(finite_ranges))
    else:
        rmin_km, rmax_km = float(ranges_km[0]), float(ranges_km[-1])
    pad_km = max(1.0, 0.025 * (rmax_km - rmin_km))
    data = {
        "station": station,
        "time_unix": times,
        "range_km": ranges_km,
        "snr_db": snr_db,
        "noise_100khz_power": noise_100khz,
        "noise_100hz_power": noise_100hz,
        "noise_rdmf_power": noise_rdmf,
        "noise_gain": noise_gain,
        "polarization_matched": polarization_matched,
        "rmin_km": rmin_km - pad_km,
        "rmax_km": rmax_km + pad_km,
    }
    meta = {
        "station": station,
        "n_patches": len(patch_names),
        "n_ranges": len(ranges_km),
        "code_len": int(code_len),
        "n_codes": int(n_codes),
        "n_channels": int(n_channels),
        "n_incoh": int(group[patch_names[0]].attrs["n_incoh"]),
        "polarization_matched": bool(np.any(polarization_matched)),
        "median_noise_gain": float(np.nanmedian(noise_gain)),
        "snr_definition": "per-range Doppler peak raw RDMF power divided by 100 kHz sparse-residual noise converted to 100 Hz and scaled to cached unnormalized RDMF power",
    }
    return data, meta


def save_results(datasets: list[dict], metas: list[dict], args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_h5.with_suffix(args.output_h5.suffix + ".tmp")
    metadata = {
        "source": "reports/generate_station_head_snr_from_noise_reference.py",
        "patch_h5": str(args.patch_h5),
        "noise_h5": str(args.noise_h5),
        "noise_bandwidth_conversion": "100 kHz residual voltage power divided by 1000 gives 100 Hz one-code RDMF-bin noise power",
        "snr_definition": "per-range Doppler peak raw RDMF power / (noise_100khz * code_len * noise_gain * n_codes * n_incoh)",
        "stations": metas,
    }
    with h5py.File(tmp, "w") as h:
        h.attrs["metadata_json"] = json.dumps(metadata, indent=2)
        h.attrs["snr_definition"] = metadata["snr_definition"]
        h.attrs["noise_bandwidth_conversion"] = metadata["noise_bandwidth_conversion"]
        h.create_dataset("stations", data=np.asarray([data["station"] for data in datasets], dtype="S"))
        for data in datasets:
            g = h.create_group(data["station"])
            for key in (
                "time_unix",
                "range_km",
                "snr_db",
                "noise_100khz_power",
                "noise_100hz_power",
                "noise_rdmf_power",
                "noise_gain",
                "polarization_matched",
            ):
                g.create_dataset(key, data=data[key])
            g.attrs["rmin_km"] = float(data["rmin_km"])
            g.attrs["rmax_km"] = float(data["rmax_km"])
    tmp.replace(args.output_h5)


def load_results(path: Path) -> tuple[list[dict], dict]:
    with h5py.File(path, "r") as h:
        metadata = json.loads(h.attrs["metadata_json"])
        datasets = []
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            datasets.append(
                {
                    "station": station,
                    "time_unix": g["time_unix"][()],
                    "range_km": g["range_km"][()],
                    "snr_db": g["snr_db"][()],
                    "rmin_km": float(g.attrs["rmin_km"]),
                    "rmax_km": float(g.attrs["rmax_km"]),
                }
            )
    return datasets, metadata


def plot_results(datasets: list[dict], metadata: dict, out_fig: Path, *, vmin_db: float | None = None, vmax_db: float | None = None) -> None:
    vals = np.concatenate([data["snr_db"][np.isfinite(data["snr_db"])] for data in datasets])
    vmin = float(vmin_db) if vmin_db is not None else (max(-10.0, float(np.nanpercentile(vals, 2.0))) if vals.size else -10.0)
    vmax = float(vmax_db) if vmax_db is not None else (min(35.0, max(vmin + 10.0, float(np.nanpercentile(vals, 99.5)))) if vals.size else 25.0)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.90")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, data in zip(axes, datasets, strict=True):
        mesh = ax.pcolormesh(
            bin_edges(unix_to_mpl(data["time_unix"])),
            bin_edges(data["range_km"]),
            data["snr_db"].T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{STATION_LABELS.get(data['station'], data['station'])}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
    for ax in axes:
        ax.set_xlim(unix_to_mpl(np.array([HEAD_PLOT_START_UNIX, HEAD_PLOT_END_UNIX])))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].text(
        0.012,
        0.94,
        "100 kHz sparse-residual noise reference; RDMF noise bandwidth 100 Hz",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
        alpha=0.65,
    )
    axes[-1].set_xlabel("UTC on 2026-05-30")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="SNR from 1 s sparse-residual noise reference (dB)", shrink=0.93)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=240, bbox_inches="tight")
    plt.close(fig)


def compute(args: argparse.Namespace) -> None:
    noise = load_noise_reference(args.noise_h5)
    datasets = []
    metas = []
    with h5py.File(args.patch_h5, "r") as h:
        stations = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]
        for station in stations:
            data, meta = process_station(
                h,
                station,
                noise[station],
                n_codes=args.n_codes,
                code_len=args.code_len,
                n_channels=args.n_channels,
            )
            datasets.append(data)
            metas.append(meta)
    save_results(datasets, metas, args)
    print(args.output_h5, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--patch-h5", type=Path, default=DEFAULT_PATCH_H5)
    parser.add_argument("--noise-h5", type=Path, default=DEFAULT_NOISE_H5)
    parser.add_argument("--output-h5", type=Path, default=OUT_H5)
    parser.add_argument("--output-fig", type=Path, default=OUT_FIG)
    parser.add_argument("--n-codes", type=int, default=5)
    parser.add_argument("--code-len", type=int, default=1000)
    parser.add_argument("--n-channels", type=int, default=2)
    parser.add_argument("--vmin-db", type=float, default=None)
    parser.add_argument("--vmax-db", type=float, default=None)
    args = parser.parse_args()
    if args.mode in {"compute", "both"}:
        compute(args)
    if args.mode in {"plot", "both"}:
        datasets, metadata = load_results(args.output_h5)
        plot_results(datasets, metadata, args.output_fig, vmin_db=args.vmin_db, vmax_db=args.vmax_db)
        print(args.output_fig, flush=True)


if __name__ == "__main__":
    main()
