#!/usr/bin/env python3
"""Batch 2-D Gaussian fits for all retained head-echo range-Doppler patches."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import (  # noqa: E402
    correlated_ellipse_points,
    fit_gaussian_patch,
    gaussian_power_snr,
    gaussian2d,
    smooth_power_db,
)
from reports.generate_head_microdoppler import STATION_LABELS, STATIONS, load_station_observations  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    DEFAULT_SPARSE_PRODUCT_DIR,
    EVENT_T0,
    compute_filtered_patch,
    init_station,
)

OUT_DIR = ROOT / "reports" / "figures" / "head_echo_gaussian_fits"
OUT_CSV = ROOT / "reports" / "head_echo_gaussian_fit_summary.csv"


def safe_name(station: str, idx: int, time_rel: float) -> str:
    return f"{station}_{idx:03d}_t{time_rel:+07.3f}s.png".replace("+", "p").replace("-", "m")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def polarization_summary_values(patch: dict | None) -> dict:
    if patch is None:
        return {}
    gamma = np.asarray(patch.get("polarization_gamma", []), dtype=np.complex64)
    if gamma.size < 2 or not np.all(np.isfinite(gamma[:2])):
        return {}
    channel_noise_equalizer = np.asarray(patch.get("channel_noise_equalizer", []), dtype=float)
    channel_noise_powers = np.asarray(patch.get("channel_noise_powers", []), dtype=float)
    phase_deg = float((np.rad2deg(np.angle(gamma[1]) - np.angle(gamma[0])) + 180.0) % 360.0 - 180.0)
    return {
        "polarization_gamma0_real": float(np.real(gamma[0])),
        "polarization_gamma0_imag": float(np.imag(gamma[0])),
        "polarization_gamma1_real": float(np.real(gamma[1])),
        "polarization_gamma1_imag": float(np.imag(gamma[1])),
        "polarization_gamma0_abs": float(np.abs(gamma[0])),
        "polarization_gamma1_abs": float(np.abs(gamma[1])),
        "polarization_relative_phase_deg": phase_deg,
        "polarization_metric_kind": patch.get("polarization_metric_kind", ""),
        "polarization_metric_mask_pixels": patch.get("polarization_metric_mask_pixels", ""),
        "polarization_metric": patch.get("polarization_metric", ""),
        "polarization_signal_power": patch.get("polarization_signal_power", ""),
        "polarization_noise_power": patch.get("polarization_noise_power", ""),
        "channel_noise_equalizer0": channel_noise_equalizer[0] if channel_noise_equalizer.size > 0 else "",
        "channel_noise_equalizer1": channel_noise_equalizer[1] if channel_noise_equalizer.size > 1 else "",
        "channel_noise_power0": channel_noise_powers[0] if channel_noise_powers.size > 0 else "",
        "channel_noise_power1": channel_noise_powers[1] if channel_noise_powers.size > 1 else "",
    }


def fit_patch(patch: dict, obs: dict, args: argparse.Namespace) -> dict:
    seed_range_km = float(obs["range_m"]) / 1000.0
    seed_rel_hz = float(obs["doppler_hz"]) - float(patch["model_doppler_hz"])
    fit = fit_gaussian_patch(
        patch["range_m"] / 1000.0,
        patch["rel_freqs"],
        patch["filtered_db"],
        seed_range_km,
        seed_rel_hz,
        args.search_range_half_width_km,
        args.search_doppler_half_width_hz,
        args.fit_range_half_width_km,
        args.fit_doppler_half_width_hz,
    )
    fit["absolute_doppler_hz"] = float(patch["model_doppler_hz"]) + fit["rel_doppler_hz"]
    fit["range_m"] = fit["range_km"] * 1000.0
    fit["range_correction_m"] = fit["range_m"] - float(obs["range_m"])
    fit["doppler_correction_hz"] = fit["absolute_doppler_hz"] - float(obs["doppler_hz"])
    snr = gaussian_power_snr(
        patch["range_m"] / 1000.0,
        patch["rel_freqs"],
        fit,
        seed_range_km,
        seed_rel_hz,
    )
    fit.update(snr)
    return fit


def process_item(item: tuple[int, dict]) -> tuple[dict | None, dict | None, str | None]:
    idx, obs = item
    try:
        patch = compute_filtered_patch((idx, obs))
        if patch is None:
            return None, None, "outside RF span"
        patch["filtered_unsmoothed_db"] = patch["filtered_db"]
        patch["filtered_db"] = smooth_power_db(
            patch["filtered_unsmoothed_db"],
            G_ARGS.smooth_range_pixels,
            G_ARGS.smooth_frequency_pixels,
        )
        patch["smooth_range_pixels"] = float(G_ARGS.smooth_range_pixels)
        patch["smooth_frequency_pixels"] = float(G_ARGS.smooth_frequency_pixels)
        fit = fit_patch(patch, obs, G_ARGS)
        return patch, fit, None
    except Exception as exc:
        return None, None, str(exc)


def plot_patch(out: Path, station: str, obs: dict, patch: dict | None, fit: dict | None, error: str | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if patch is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.03,
            0.95,
            f"{STATION_LABELS.get(station, station)} t={float(obs['unix_time']) - EVENT_T0:.3f} s\n{error}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    range_km = patch["range_m"] / 1000.0
    rel_hz = patch["rel_freqs"]
    raw_rel_hz = patch.get("raw_rel_freqs", rel_hz)
    seed_range_km = float(obs["range_m"]) / 1000.0
    seed_rel_hz = float(obs["doppler_hz"]) - float(patch["model_doppler_hz"])
    raw = patch["raw_db"]
    filtered = patch["filtered_db"]

    fit_model = None
    if fit is not None:
        rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
        fit_model = gaussian2d(
            np.array(
                [
                    float(fit["offset_db"]),
                    float(fit["amplitude_db"]),
                    float(fit["range_km"]),
                    float(fit["rel_doppler_hz"]),
                    float(fit["sigma_range_km"]),
                    float(fit["sigma_doppler_hz"]),
                    float(fit.get("range_doppler_correlation", 0.0)),
                ]
            ),
            rr,
            ff,
        )

    x_pad = 650.0
    y_pad = 14.0
    if fit is None:
        x_min, x_max = seed_rel_hz - x_pad, seed_rel_hz + x_pad
        y_min, y_max = seed_range_km - y_pad, seed_range_km + y_pad
    else:
        fit_range_km = float(fit["range_km"])
        fit_rel_hz = float(fit["rel_doppler_hz"])
        x_min = min(seed_rel_hz, fit_rel_hz) - x_pad
        x_max = max(seed_rel_hz, fit_rel_hz) + x_pad
        y_min = min(seed_range_km, fit_range_km) - y_pad
        y_max = max(seed_range_km, fit_range_km) + y_pad

    plt.rcParams.update({"font.size": 9})
    filtered_unsmoothed = patch.get("filtered_unsmoothed_db", filtered)
    images = [
        ("Raw matched filter", raw, raw_rel_hz),
        ("2D FFT notch filtered", filtered_unsmoothed, rel_hz),
        (
            f"2D FFT filtered + smoothed ({patch.get('smooth_range_pixels', 4):g}, {patch.get('smooth_frequency_pixels', 16):g}) px",
            filtered,
            rel_hz,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, img, img_rel_hz) in zip(axes, images, strict=True):
        show = (
            (range_km[:, None] >= y_min)
            & (range_km[:, None] <= y_max)
            & (img_rel_hz[None, :] >= x_min)
            & (img_rel_hz[None, :] <= x_max)
        )
        finite = img[show & np.isfinite(img)]
        if finite.size:
            vmin = float(np.nanpercentile(finite, 3.0))
            vmax = float(np.nanpercentile(finite, 99.6))
        else:
            vmin, vmax = float(np.nanpercentile(img, 3.0)), float(np.nanpercentile(img, 99.6))
        mappable = ax.pcolormesh(img_rel_hz, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=9, mew=1.4, color="#9be7ff", linestyle="none")
        if fit is not None:
            fit_range_km = float(fit["range_km"])
            fit_rel_hz = float(fit["rel_doppler_hz"])
            sig_r = float(fit["sigma_range_km"])
            sig_f = float(fit["sigma_doppler_hz"])
            rho = float(fit.get("range_doppler_correlation", 0.0))
            ax.plot([seed_rel_hz, fit_rel_hz], [seed_range_km, fit_range_km], color="white", lw=0.9, alpha=0.9)
            ax.plot(fit_rel_hz, fit_range_km, marker="x", ms=8, mew=1.4, color="#42ff9f", linestyle="none")
            ell_x, ell_y = correlated_ellipse_points(fit_range_km, fit_rel_hz, sig_r, sig_f, rho)
            ax.plot(ell_x, ell_y, color="#42ff9f", lw=1.2)
            if fit_model is not None:
                ax.contour(
                    rel_hz,
                    range_km,
                    fit_model,
                    levels=[
                        float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-2.0),
                        float(fit["offset_db"]) + float(fit["amplitude_db"]) * np.exp(-0.5),
                    ],
                    colors=["#42ff9f", "#d8ffe8"],
                    linewidths=[0.8, 1.0],
                )
        ax.set_title(title)
        ax.set_xlabel("$f-f_{D,model}$ (Hz)")
        ax.grid(color="white", alpha=0.12, lw=0.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        cb = fig.colorbar(mappable, ax=ax, pad=0.012)
        cb.set_label("dB")
    axes[0].set_ylabel("Bistatic range (km)")

    if fit is None:
        title = f"{STATION_LABELS.get(station, station)} t={float(obs['unix_time']) - EVENT_T0:.3f} s: fit failed ({error})"
    else:
        title = (
            f"{STATION_LABELS.get(station, station)} #{int(patch['idx']):03d} "
            f"t={float(obs['unix_time']) - EVENT_T0:.3f} s: "
            f"correction {fit['range_correction_m']:+.0f} m, {fit['doppler_correction_hz']:+.1f} Hz"
        )
    fig.suptitle(title, y=1.03, fontsize=9)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "station",
        "idx",
        "time_unix",
        "time_rel_s",
        "seed_range_m",
        "fit_range_m",
        "range_correction_m",
        "seed_doppler_hz",
        "fit_doppler_hz",
        "doppler_correction_hz",
        "model_doppler_hz",
        "fit_rel_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "range_doppler_correlation",
        "amplitude_db",
        "signal_power_gaussian",
        "noise_power_floor",
        "snr_linear",
        "snr_db",
        "noise_bandwidth_hz",
        "noise_range_width_km",
        "noise_n_range_bins",
        "noise_n_doppler_bins",
        "cost",
        "success",
        "error",
        "n_coh",
        "n_incoh",
        "step_codes",
        "doppler_zero_pad_factor",
        "notch_half_width",
        "fft_filter_domain",
        "inpaint_notched_columns",
        "smooth_range_pixels",
        "smooth_frequency_pixels",
        "coherent_window_s",
        "total_window_s",
        "polarization_gamma0_real",
        "polarization_gamma0_imag",
        "polarization_gamma1_real",
        "polarization_gamma1_imag",
        "polarization_gamma0_abs",
        "polarization_gamma1_abs",
        "polarization_relative_phase_deg",
        "polarization_metric_kind",
        "polarization_metric_mask_pixels",
        "polarization_metric",
        "polarization_signal_power",
        "polarization_noise_power",
        "channel_noise_equalizer0",
        "channel_noise_equalizer1",
        "channel_noise_power0",
        "channel_noise_power1",
        "sparse_voltage_subtract",
        "sparse_voltage_mode",
        "sparse_voltage_gain",
        "figure",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_station(station: str, observations: list[dict], payload: dict, args: argparse.Namespace) -> list[dict]:
    print(f"{station}: loading raw data", flush=True)
    init_station(payload, station, args)
    items = list(enumerate(observations))
    if args.limit:
        items = items[: args.limit]
    n_workers = min(max(1, args.workers), len(items))
    print(f"{station}: fitting {len(items)} head-echo patches with {n_workers} worker(s)", flush=True)

    if n_workers == 1:
        results = [process_item(item) for item in items]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_worker_set_args, initargs=(args,)) as pool:
            results = list(pool.imap(process_item, items, chunksize=1))

    rows = []
    for (idx, obs), (patch, fit, error) in zip(items, results, strict=True):
        time_rel = float(obs["unix_time"]) - EVENT_T0
        fig_path = args.out_dir / station / safe_name(station, idx, time_rel)
        plot_patch(fig_path, station, obs, patch, fit, error)
        if patch is None or fit is None:
            rows.append(
                {
                    "station": station,
                    "idx": idx,
                    "time_unix": float(obs["unix_time"]),
                    "time_rel_s": time_rel,
                    "seed_range_m": float(obs["range_m"]),
                    "seed_doppler_hz": float(obs["doppler_hz"]),
                    "success": False,
                    "error": error,
                    "n_coh": args.n_coh,
                    "n_incoh": args.n_incoh,
                    "step_codes": args.step_codes,
                    "doppler_zero_pad_factor": args.doppler_zero_pad_factor,
                    "notch_half_width": args.notch_half_width,
                    "fft_filter_domain": args.fft_filter_domain,
                    "inpaint_notched_columns": bool(args.inpaint_notched_columns),
                    "smooth_range_pixels": args.smooth_range_pixels,
                    "smooth_frequency_pixels": args.smooth_frequency_pixels,
                    "coherent_window_s": "",
                    "total_window_s": "",
                    "sparse_voltage_subtract": bool(args.sparse_voltage_subtract),
                    "sparse_voltage_mode": args.sparse_voltage_mode,
                    "sparse_voltage_gain": args.sparse_voltage_gain,
                    "figure": display_path(fig_path),
                }
                | polarization_summary_values(patch)
            )
            continue
        rows.append(
            {
                "station": station,
                "idx": idx,
                "time_unix": float(obs["unix_time"]),
                "time_rel_s": time_rel,
                "seed_range_m": float(obs["range_m"]),
                "fit_range_m": fit["range_m"],
                "range_correction_m": fit["range_correction_m"],
                "seed_doppler_hz": float(obs["doppler_hz"]),
                "fit_doppler_hz": fit["absolute_doppler_hz"],
                "doppler_correction_hz": fit["doppler_correction_hz"],
                "model_doppler_hz": float(patch["model_doppler_hz"]),
                "fit_rel_doppler_hz": fit["rel_doppler_hz"],
                "sigma_range_m": 1000.0 * fit["sigma_range_km"],
                "sigma_doppler_hz": fit["sigma_doppler_hz"],
                "range_doppler_correlation": fit["range_doppler_correlation"],
                "amplitude_db": fit["amplitude_db"],
                "signal_power_gaussian": fit["signal_power_gaussian"],
                "noise_power_floor": fit["noise_power_floor"],
                "snr_linear": fit["snr_linear"],
                "snr_db": fit["snr_db"],
                "noise_bandwidth_hz": fit["noise_bandwidth_hz"],
                "noise_range_width_km": fit["noise_range_width_km"],
                "noise_n_range_bins": fit["noise_n_range_bins"],
                "noise_n_doppler_bins": fit["noise_n_doppler_bins"],
                "cost": fit["cost"],
                "success": fit["success"],
                "error": error or "",
                "n_coh": args.n_coh,
                "n_incoh": args.n_incoh,
                "step_codes": args.step_codes,
                "doppler_zero_pad_factor": args.doppler_zero_pad_factor,
                "notch_half_width": args.notch_half_width,
                "fft_filter_domain": args.fft_filter_domain,
                "inpaint_notched_columns": bool(args.inpaint_notched_columns),
                "smooth_range_pixels": args.smooth_range_pixels,
                "smooth_frequency_pixels": args.smooth_frequency_pixels,
                "coherent_window_s": patch.get("coherent_window_s", ""),
                "total_window_s": patch.get("total_window_s", ""),
                "sparse_voltage_subtract": bool(args.sparse_voltage_subtract),
                "sparse_voltage_mode": args.sparse_voltage_mode,
                "sparse_voltage_gain": args.sparse_voltage_gain,
                "figure": display_path(fig_path),
            }
            | polarization_summary_values(patch)
        )
    return rows


G_ARGS = None


def _worker_set_args(args: argparse.Namespace) -> None:
    global G_ARGS
    G_ARGS = args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--fft-filter-domain", choices=("db", "linear"), default="linear")
    p.add_argument("--inpaint-notched-columns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument(
        "--smooth-frequency-pixels",
        type=float,
        default=None,
        help="Frequency-axis Gaussian smoothing in pixels; defaults to 2*n_coh.",
    )
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--sparse-voltage-subtract", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sparse-product-dir", type=Path, default=DEFAULT_SPARSE_PRODUCT_DIR)
    p.add_argument("--sparse-voltage-mode", choices=("full", "dc"), default="full")
    p.add_argument("--sparse-voltage-gain", type=float, default=1.0)
    p.add_argument("--polarization-combine", action="store_true")
    p.add_argument("--polarization-method", choices=("phase", "grid", "grid_image"), default="phase")
    p.add_argument("--polarization-equalize-noise", action="store_true")
    p.add_argument("--polarization-equalize-block-samples", type=int, default=100000)
    p.add_argument("--polarization-grid-theta", type=int, default=91)
    p.add_argument("--polarization-grid-phase", type=int, default=361)
    p.add_argument("--polarization-phase-chunk", type=int, default=32)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--summary", type=Path, default=OUT_CSV)
    args = p.parse_args()
    if args.smooth_frequency_pixels is None:
        args.smooth_frequency_pixels = 2.0 * float(args.n_coh)

    global G_ARGS
    G_ARGS = args

    payload, observations = load_station_observations()
    all_rows = []
    for station in args.stations:
        if station not in observations:
            raise ValueError(f"Unknown station: {station}")
        all_rows.extend(run_station(station, observations[station], payload, args))
    write_summary(args.summary, all_rows)
    print(f"Wrote {args.summary}", flush=True)
    print(f"Wrote {len(all_rows)} diagnostic figures under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
