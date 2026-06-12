#!/usr/bin/env python3
"""Estimate head-echo SNR using the sparse-residual 1 s noise reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler import STATION_LABELS  # noqa: E402

EVENT_T0 = 1780164380.0
STATIONS = ("beverly", "clinton", "pomfret")
DEFAULT_PATCH_H5 = (
    ROOT
    / "products"
    / "head_echo_gaussian_patches"
    / "head_echo_gaussian_patches_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_retained_ext0p5.h5"
)
DEFAULT_FIT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_v0p2p7_1coh_2incoh_zp16_sparse_dcmean.csv"
DEFAULT_NOISE_H5 = ROOT / "products" / "sparse_residual_noise_power" / "sparse_residual_noise_power_1s.h5"
OUT_H5 = ROOT / "products" / "sparse_residual_noise_power" / "head_echo_snr_from_1s_noise_reference.h5"
OUT_FIG = ROOT / "reports" / "figures" / "head_echo_snr_from_1s_noise_reference.png"


def load_fit_rows(path: Path, stations: tuple[str, ...]) -> dict[str, dict[int, dict]]:
    out = {station: {} for station in stations}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            station = row["station"]
            if station not in out or row.get("success", "").lower() != "true":
                continue
            out[station][int(row["idx"])] = row
    return out


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


def process_station(
    patch_h5: h5py.File,
    station: str,
    fit_rows: dict[int, dict],
    noise: dict[str, np.ndarray],
    *,
    n_codes: int,
    code_len: int,
    rel_half_width_hz: float,
) -> tuple[dict, list[dict]]:
    group = patch_h5[station]
    spectra = []
    rows = []
    rel_ref = None
    for key in sorted(k for k in group.keys() if k.startswith("patch_")):
        patch = group[key]
        if not bool(patch.attrs.get("valid", False)):
            continue
        idx = int(patch.attrs["idx"])
        fit = fit_rows.get(idx)
        if fit is None:
            continue
        range_m = patch["range_m"][()].astype(float)
        rel_freqs = patch["rel_freqs"][()].astype(float)
        raw_power = 10.0 ** (patch["raw_db"][()].astype(float) / 10.0)
        fit_range_m = float(fit["fit_range_m"])
        fit_rel_hz = float(fit["fit_rel_doppler_hz"])
        row_i = int(np.argmin(np.abs(range_m - fit_range_m)))
        n_incoh = int(patch.attrs["n_incoh"])
        channels = json.loads(patch_h5.attrs.get("channels_json", "null")) if "channels_json" in patch_h5.attrs else None
        n_channels = len(channels) if channels is not None else 2
        noise_100khz = nearest_noise_power(noise, float(fit["time_unix"]))
        # The 1 s noise product is the 100 kHz residual voltage power. A
        # one-code coherent RDMF bin has 100 Hz noise bandwidth, so the
        # normalized RDMF-bin noise is noise_100khz / 1000. The cached patch
        # stores unnormalized summed FFT powers, hence the equivalent
        # denominator below includes the coherent code length and summed
        # channel/code/incoherent accumulations.
        noise_rdmf_power = noise_100khz * code_len * n_channels * n_codes * n_incoh
        snr_db = 10.0 * np.log10(np.maximum(raw_power[row_i, :] / max(noise_rdmf_power, 1e-300), 1e-300))
        shifted_rel = rel_freqs - fit_rel_hz
        if rel_ref is None:
            df = float(np.nanmedian(np.diff(rel_freqs)))
            rel_ref = np.arange(-rel_half_width_hz, rel_half_width_hz + 0.5 * df, df)
        spectra.append(np.interp(rel_ref, shifted_rel, snr_db, left=np.nan, right=np.nan))
        rows.append(
            {
                "idx": idx,
                "time_unix": float(fit["time_unix"]),
                "time_rel_s": float(fit["time_rel_s"]),
                "fit_range_m": fit_range_m,
                "fit_doppler_hz": float(fit["fit_doppler_hz"]),
                "fit_rel_doppler_hz": fit_rel_hz,
                "noise_100khz_power": noise_100khz,
                "noise_100hz_power": noise_100khz / 1000.0,
                "noise_rdmf_power": noise_rdmf_power,
                "n_channels": n_channels,
                "n_codes": n_codes,
                "n_incoh": n_incoh,
                "code_len": code_len,
                "peak_snr_db": float(np.nanmax(snr_db)),
            }
        )
    if not spectra:
        raise ValueError(f"{station}: no SNR rows generated")
    order = np.argsort([row["time_rel_s"] for row in rows])
    spectra = np.asarray(spectra, dtype=float)[order]
    rows = [rows[int(i)] for i in order]
    result = {
        "station": station,
        "time_rel": np.asarray([row["time_rel_s"] for row in rows], dtype=float),
        "time_unix": np.asarray([row["time_unix"] for row in rows], dtype=float),
        "ranges_km": np.asarray([row["fit_range_m"] / 1000.0 for row in rows], dtype=float),
        "fit_dopplers_hz": np.asarray([row["fit_doppler_hz"] for row in rows], dtype=float),
        "rel_freqs": rel_ref,
        "snr_db": spectra,
        "peak_snr_db": np.asarray([row["peak_snr_db"] for row in rows], dtype=float),
        "noise_100khz_power": np.asarray([row["noise_100khz_power"] for row in rows], dtype=float),
        "noise_100hz_power": np.asarray([row["noise_100hz_power"] for row in rows], dtype=float),
        "noise_rdmf_power": np.asarray([row["noise_rdmf_power"] for row in rows], dtype=float),
    }
    return result, rows


def save_results(results: list[dict], rows_by_station: dict[str, list[dict]], args: argparse.Namespace) -> None:
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_h5.with_suffix(args.output_h5.suffix + ".tmp")
    metadata = {
        "source": "reports/generate_head_echo_snr_from_noise_reference.py",
        "patch_h5": str(args.patch_h5),
        "fit_summary": str(args.fit_summary),
        "noise_h5": str(args.noise_h5),
        "stations": [res["station"] for res in results],
        "noise_bandwidth_conversion": "100 kHz residual voltage power divided by 1000 gives 100 Hz one-code RDMF-bin noise power",
        "snr_definition": "cached unnormalized RDMF row power / (noise_100khz * code_len * n_channels * n_codes * n_incoh)",
        "n_codes": int(args.n_codes),
        "code_len": int(args.code_len),
    }
    with h5py.File(tmp, "w") as h:
        h.attrs["metadata_json"] = json.dumps(metadata, indent=2)
        h.attrs["snr_definition"] = metadata["snr_definition"]
        h.attrs["noise_bandwidth_conversion"] = metadata["noise_bandwidth_conversion"]
        h.create_dataset("stations", data=np.asarray([res["station"] for res in results], dtype="S"))
        for res in results:
            g = h.create_group(res["station"])
            for key in (
                "time_rel",
                "time_unix",
                "ranges_km",
                "fit_dopplers_hz",
                "rel_freqs",
                "snr_db",
                "peak_snr_db",
                "noise_100khz_power",
                "noise_100hz_power",
                "noise_rdmf_power",
            ):
                g.create_dataset(key, data=res[key])
            rows_json = json.dumps(rows_by_station[res["station"]], indent=2)
            g.attrs["rows_json"] = rows_json
    tmp.replace(args.output_h5)


def load_results(path: Path) -> tuple[list[dict], dict]:
    with h5py.File(path, "r") as h:
        metadata = json.loads(h.attrs["metadata_json"])
        results = []
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            results.append(
                {
                    "station": station,
                    "time_rel": g["time_rel"][()],
                    "ranges_km": g["ranges_km"][()],
                    "rel_freqs": g["rel_freqs"][()],
                    "snr_db": g["snr_db"][()],
                    "peak_snr_db": g["peak_snr_db"][()],
                }
            )
    return results, metadata


def plot_results(results: list[dict], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    finite = np.concatenate([res["snr_db"][np.isfinite(res["snr_db"])] for res in results])
    vmin = max(-10.0, float(np.nanpercentile(finite, 2.0)))
    vmax = max(10.0, float(np.nanpercentile(finite, 99.5)))
    mappable = None
    for ax, res in zip(axes, results, strict=True):
        ax.set_facecolor("black")
        mappable = ax.pcolormesh(
            res["time_rel"],
            res["rel_freqs"],
            res["snr_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.set_ylabel("$f-f_{D,fit}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo SNR")
        ax.text(
            0.01,
            0.93,
            f"range {np.nanmedian(res['ranges_km']):.0f} km, peak median {np.nanmedian(res['peak_snr_db']):.1f} dB",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, pad=0.012)
        cb.set_label("SNR from 1 s sparse-residual noise reference (dB)")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute(args: argparse.Namespace) -> None:
    fit_rows = load_fit_rows(args.fit_summary, tuple(args.stations))
    noise = load_noise_reference(args.noise_h5)
    results = []
    rows_by_station = {}
    with h5py.File(args.patch_h5, "r") as h:
        for station in args.stations:
            res, rows = process_station(
                h,
                station,
                fit_rows[station],
                noise[station],
                n_codes=args.n_codes,
                code_len=args.code_len,
                rel_half_width_hz=args.rel_half_width_hz,
            )
            results.append(res)
            rows_by_station[station] = rows
    save_results(results, rows_by_station, args)
    print(args.output_h5, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compute", "plot", "both"), default="both")
    parser.add_argument("--stations", nargs="+", default=list(STATIONS))
    parser.add_argument("--patch-h5", type=Path, default=DEFAULT_PATCH_H5)
    parser.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    parser.add_argument("--noise-h5", type=Path, default=DEFAULT_NOISE_H5)
    parser.add_argument("--output-h5", type=Path, default=OUT_H5)
    parser.add_argument("--output-fig", type=Path, default=OUT_FIG)
    parser.add_argument("--n-codes", type=int, default=5)
    parser.add_argument("--code-len", type=int, default=1000)
    parser.add_argument("--rel-half-width-hz", type=float, default=900.0)
    args = parser.parse_args()
    if args.mode in {"compute", "both"}:
        compute(args)
    if args.mode in {"plot", "both"}:
        results, _metadata = load_results(args.output_h5)
        plot_results(results, args.output_fig)
        print(args.output_fig, flush=True)


if __name__ == "__main__":
    main()
