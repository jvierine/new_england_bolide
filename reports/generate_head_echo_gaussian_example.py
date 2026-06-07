#!/usr/bin/env python3
"""Plot one 2-D matched-filter patch before/after filtering with a Gaussian fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler import STATION_LABELS, load_station_observations  # noqa: E402
import reports.generate_head_microdoppler_2dfft_notched as md2d  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    DATA_DIR,
    EVENT_T0,
    DEFAULT_SPARSE_PRODUCT_DIR,
    compute_filtered_patch,
    init_station,
)

OUT_FIG = ROOT / "reports" / "figures" / "head_echo_gaussian_2dfft_example.png"
OUT_CACHE = ROOT / "reports" / "head_echo_gaussian_2dfft_example.h5"


def closest_observation(observations: list[dict], time_rel: float) -> tuple[int, dict]:
    target = EVENT_T0 + time_rel
    idx = int(np.argmin([abs(float(obs["unix_time"]) - target) for obs in observations]))
    return idx, observations[idx]


def gaussian2d(params: np.ndarray, range_km: np.ndarray, rel_hz: np.ndarray) -> np.ndarray:
    offset, amp, r0, f0, sig_r, sig_f, rho = params
    rr = (range_km - r0) / sig_r
    ff = (rel_hz - f0) / sig_f
    one_minus_rho2 = np.maximum(1.0 - rho * rho, 1e-6)
    exponent = -0.5 * (rr * rr - 2.0 * rho * rr * ff + ff * ff) / one_minus_rho2
    return offset + amp * np.exp(exponent)


def smooth_power_db(image_db: np.ndarray, range_sigma_px: float, frequency_sigma_px: float) -> np.ndarray:
    if range_sigma_px <= 0.0 and frequency_sigma_px <= 0.0:
        return np.asarray(image_db, dtype=float)
    power = 10.0 ** (np.asarray(image_db, dtype=float) / 10.0)
    smoothed = gaussian_filter(power, sigma=(range_sigma_px, frequency_sigma_px), mode="nearest")
    return 10.0 * np.log10(np.maximum(smoothed, 1e-30))


def correlated_ellipse_points(
    range_km: float,
    rel_hz: float,
    sigma_range_km: float,
    sigma_doppler_hz: float,
    rho: float,
    n: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    cov = np.array(
        [
            [sigma_range_km * sigma_range_km, rho * sigma_range_km * sigma_doppler_hz],
            [rho * sigma_range_km * sigma_doppler_hz, sigma_doppler_hz * sigma_doppler_hz],
        ],
        dtype=float,
    )
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    pts = np.array([range_km, rel_hz])[:, None] + (vecs @ (np.sqrt(vals)[:, None] * unit))
    return pts[1], pts[0]


def gaussian_plot_window(
    seed_range_km: float,
    seed_rel_hz: float,
    fit_range_km: float,
    fit_rel_hz: float,
    x_pad_hz: float = 650.0,
    y_pad_km: float = 14.0,
) -> tuple[float, float, float, float]:
    x_min = min(seed_rel_hz, fit_rel_hz) - x_pad_hz
    x_max = max(seed_rel_hz, fit_rel_hz) + x_pad_hz
    y_min = min(seed_range_km, fit_range_km) - y_pad_km
    y_max = max(seed_range_km, fit_range_km) + y_pad_km
    return x_min, x_max, y_min, y_max


def gaussian_power_snr(
    range_km: np.ndarray,
    rel_hz: np.ndarray,
    fit: dict,
    seed_range_km: float,
    seed_rel_hz: float,
    x_pad_hz: float = 650.0,
    y_pad_km: float = 14.0,
) -> dict:
    """Estimate SNR from the fitted Gaussian blob over the plotted RD window.

    The fit model is in dB.  We convert the fitted surface and fitted constant
    floor to linear power, integrate the excess above the floor as signal, and
    integrate the floor over the same range-Doppler plot area as noise.
    """
    fit_range_km = float(fit["range_km"])
    fit_rel_hz = float(fit["rel_doppler_hz"])
    x_min, x_max, y_min, y_max = gaussian_plot_window(
        seed_range_km,
        seed_rel_hz,
        fit_range_km,
        fit_rel_hz,
        x_pad_hz=x_pad_hz,
        y_pad_km=y_pad_km,
    )
    keep_r = (range_km >= y_min) & (range_km <= y_max)
    keep_f = (rel_hz >= x_min) & (rel_hz <= x_max)
    if not np.any(keep_r) or not np.any(keep_f):
        raise ValueError("Gaussian SNR plot window is empty")
    rr, ff = np.meshgrid(range_km[keep_r], rel_hz[keep_f], indexing="ij")
    model_db = gaussian2d(
        np.array(
            [
                float(fit["offset_db"]),
                float(fit["amplitude_db"]),
                fit_range_km,
                fit_rel_hz,
                float(fit["sigma_range_km"]),
                float(fit["sigma_doppler_hz"]),
                float(fit.get("range_doppler_correlation", 0.0)),
            ],
            dtype=float,
        ),
        rr,
        ff,
    )
    floor_power = 10.0 ** (float(fit["offset_db"]) / 10.0)
    model_power = 10.0 ** (model_db / 10.0)
    signal_density = np.maximum(model_power - floor_power, 0.0)
    signal_power = float(np.nansum(signal_density))
    # For the SNR used for RCS estimation, integrate the constant floor over
    # the Doppler bandwidth shown in the RD view.  The Gaussian fit itself
    # supplies the range weighting of the signal blob.
    noise_power = float(floor_power * np.sum(keep_f))
    snr_linear = signal_power / max(noise_power, 1e-300)
    return {
        "signal_power_gaussian": signal_power,
        "noise_power_floor": noise_power,
        "snr_linear": snr_linear,
        "snr_db": float(10.0 * np.log10(max(snr_linear, 1e-300))),
        "noise_bandwidth_hz": float(x_max - x_min),
        "noise_range_width_km": float(y_max - y_min),
        "noise_n_range_bins": int(np.sum(keep_r)),
        "noise_n_doppler_bins": int(np.sum(keep_f)),
        "signal_n_range_bins": int(np.sum(keep_r)),
        "signal_n_doppler_bins": int(np.sum(keep_f)),
        "plot_doppler_min_hz": float(x_min),
        "plot_doppler_max_hz": float(x_max),
        "plot_range_min_km": float(y_min),
        "plot_range_max_km": float(y_max),
    }


def gaussian_one_sigma_mask(range_km: np.ndarray, rel_hz: np.ndarray, fit: dict) -> np.ndarray:
    rr = (range_km[:, None] - float(fit["range_km"])) / max(float(fit["sigma_range_km"]), 1e-12)
    ff = (rel_hz[None, :] - float(fit["rel_doppler_hz"])) / max(float(fit["sigma_doppler_hz"]), 1e-12)
    rho = float(fit.get("range_doppler_correlation", 0.0))
    one_minus_rho2 = max(1.0 - rho * rho, 1e-6)
    q = (rr * rr - 2.0 * rho * rr * ff + ff * ff) / one_minus_rho2
    return q <= 1.0


def fit_gaussian_patch(
    range_km: np.ndarray,
    rel_hz: np.ndarray,
    img_db: np.ndarray,
    seed_range_km: float,
    seed_rel_hz: float,
    search_range_half_width_km: float,
    search_doppler_half_width_hz: float,
    fit_range_half_width_km: float,
    fit_doppler_half_width_hz: float,
) -> dict:
    search = (
        (np.abs(range_km[:, None] - seed_range_km) <= search_range_half_width_km)
        & (np.abs(rel_hz[None, :] - seed_rel_hz) <= search_doppler_half_width_hz)
        & np.isfinite(img_db)
    )
    if not np.any(search):
        raise ValueError("Gaussian search window is empty")

    masked = np.where(search, img_db, -np.inf)
    peak_ri, peak_fi = np.unravel_index(int(np.nanargmax(masked)), img_db.shape)
    peak_range_km = float(range_km[peak_ri])
    peak_rel_hz = float(rel_hz[peak_fi])

    fit_mask = (
        (np.abs(range_km[:, None] - peak_range_km) <= fit_range_half_width_km)
        & (np.abs(rel_hz[None, :] - peak_rel_hz) <= fit_doppler_half_width_hz)
        & np.isfinite(img_db)
    )
    rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
    x_r = rr[fit_mask]
    x_f = ff[fit_mask]
    y = img_db[fit_mask]
    if y.size < 20:
        raise ValueError("Gaussian fit window has too few pixels")

    floor = float(np.nanpercentile(y, 20.0))
    peak = float(np.nanmax(y))
    amp0 = max(peak - floor, 1.0)
    min_sig_r = max(float(np.nanmedian(np.diff(range_km))), 0.05)
    min_sig_f = max(float(np.nanmedian(np.diff(rel_hz))), 1.0)
    lower = np.array(
        [
            floor - 30.0,
            0.0,
            peak_range_km - fit_range_half_width_km,
            peak_rel_hz - fit_doppler_half_width_hz,
            min_sig_r,
            min_sig_f,
            -0.95,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            peak + 10.0,
            80.0,
            peak_range_km + fit_range_half_width_km,
            peak_rel_hz + fit_doppler_half_width_hz,
            fit_range_half_width_km,
            fit_doppler_half_width_hz,
            0.95,
        ],
        dtype=float,
    )
    p0 = np.array(
        [
            floor,
            amp0,
            peak_range_km,
            peak_rel_hz,
            min(max(1.2, min_sig_r), fit_range_half_width_km),
            min(max(80.0, min_sig_f), fit_doppler_half_width_hz),
            0.0,
        ],
        dtype=float,
    )

    scale = max(float(np.nanstd(y)), 1.0)

    def residual(params: np.ndarray) -> np.ndarray:
        return (gaussian2d(params, x_r, x_f) - y) / scale

    result = least_squares(residual, p0, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=3000)
    offset, amp, r0, f0, sig_r, sig_f, rho = result.x
    return {
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "offset_db": float(offset),
        "amplitude_db": float(amp),
        "range_km": float(r0),
        "rel_doppler_hz": float(f0),
        "sigma_range_km": float(sig_r),
        "sigma_doppler_hz": float(sig_f),
        "range_doppler_correlation": float(rho),
        "peak_range_km": peak_range_km,
        "peak_rel_doppler_hz": peak_rel_hz,
    }


def save_example(path: Path, patch: dict, fit: dict, station: str, obs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Single head-echo 2D FFT notch Gaussian-fit diagnostic"
        h.attrs["station"] = station
        h.attrs["unix_time"] = float(obs["unix_time"])
        h.attrs["time_rel"] = float(obs["unix_time"]) - EVENT_T0
        h.attrs["seed_range_m"] = float(obs["range_m"])
        h.attrs["seed_doppler_hz"] = float(obs["doppler_hz"])
        h.attrs["model_doppler_hz"] = float(patch["model_doppler_hz"])
        h.attrs["coherent_window_s"] = float(patch.get("coherent_window_s", np.nan))
        h.attrs["total_window_s"] = float(patch.get("total_window_s", np.nan))
        h.attrs["n_coh"] = int(patch.get("n_coh", -1))
        h.attrs["n_incoh"] = int(patch.get("n_incoh", -1))
        h.attrs["step_codes"] = int(patch.get("step_codes", -1))
        for key in ("polarization_metric", "polarization_signal_power", "polarization_noise_power"):
            if key in patch and np.isfinite(patch[key]):
                h.attrs[key] = float(patch[key])
        if "channel_noise_equalizer" in patch and len(patch["channel_noise_equalizer"]):
            h["channel_noise_equalizer"] = np.asarray(patch["channel_noise_equalizer"], dtype=np.float32)
        if "channel_noise_powers" in patch and len(patch["channel_noise_powers"]):
            h["channel_noise_powers"] = np.asarray(patch["channel_noise_powers"], dtype=np.float64)
        if patch.get("polarization_metric_kind"):
            h.attrs["polarization_metric_kind"] = str(patch["polarization_metric_kind"])
        if int(patch.get("polarization_metric_mask_pixels", 0)) > 0:
            h.attrs["polarization_metric_mask_pixels"] = int(patch["polarization_metric_mask_pixels"])
        if "polarization_gamma" in patch and len(patch["polarization_gamma"]):
            h["polarization_gamma"] = np.asarray(patch["polarization_gamma"], dtype=np.complex64)
        if "polarization_metric_mask" in patch:
            h["polarization_metric_mask"] = np.asarray(patch["polarization_metric_mask"], dtype=np.uint8)
        if "polarization_reference_fit" in patch:
            rg = h.create_group("polarization_reference_fit")
            for key, val in patch["polarization_reference_fit"].items():
                if isinstance(val, str):
                    rg.attrs[key] = val
                else:
                    rg.attrs[key] = val
        for key in ("range_m", "rel_freqs", "raw_db", "filtered_db", "filtered_unsmoothed_db"):
            if key not in patch:
                continue
            h[key] = patch[key]
        g = h.create_group("gaussian_fit")
        for key, val in fit.items():
            if isinstance(val, str):
                g.attrs[key] = val
            else:
                g.attrs[key] = val
        g.attrs["absolute_doppler_hz"] = float(patch["model_doppler_hz"]) + fit["rel_doppler_hz"]
        g.attrs["range_m"] = fit["range_km"] * 1000.0


def load_example(path: Path) -> tuple[dict, dict, str, dict]:
    with h5py.File(path, "r") as h:
        patch = {
            "range_m": h["range_m"][()],
            "rel_freqs": h["rel_freqs"][()],
            "raw_db": h["raw_db"][()],
            "filtered_db": h["filtered_db"][()],
            "model_doppler_hz": float(h.attrs["model_doppler_hz"]),
        }
        if "filtered_unsmoothed_db" in h:
            patch["filtered_unsmoothed_db"] = h["filtered_unsmoothed_db"][()]
        if "polarization_gamma" in h:
            patch["polarization_gamma"] = h["polarization_gamma"][()]
        if "channel_noise_equalizer" in h:
            patch["channel_noise_equalizer"] = h["channel_noise_equalizer"][()]
        if "channel_noise_powers" in h:
            patch["channel_noise_powers"] = h["channel_noise_powers"][()]
        for key in ("polarization_metric", "polarization_signal_power", "polarization_noise_power"):
            if key in h.attrs:
                patch[key] = float(h.attrs[key])
        if "polarization_metric_kind" in h.attrs:
            patch["polarization_metric_kind"] = str(h.attrs["polarization_metric_kind"])
        if "polarization_metric_mask_pixels" in h.attrs:
            patch["polarization_metric_mask_pixels"] = int(h.attrs["polarization_metric_mask_pixels"])
        if "polarization_metric_mask" in h:
            patch["polarization_metric_mask"] = h["polarization_metric_mask"][()].astype(bool)
        if "polarization_reference_fit" in h:
            patch["polarization_reference_fit"] = dict(h["polarization_reference_fit"].attrs.items())
        fit = dict(h["gaussian_fit"].attrs.items())
        station = str(h.attrs["station"])
        obs = {
            "unix_time": float(h.attrs["unix_time"]),
            "range_m": float(h.attrs["seed_range_m"]),
            "doppler_hz": float(h.attrs["seed_doppler_hz"]),
        }
    return patch, fit, station, obs


def plot_example(path: Path, out: Path) -> None:
    patch, fit, station, obs = load_example(path)
    range_km = patch["range_m"] / 1000.0
    rel_hz = patch["rel_freqs"]
    seed_range_km = obs["range_m"] / 1000.0
    seed_rel_hz = obs["doppler_hz"] - patch["model_doppler_hz"]
    fit_range_km = float(fit["range_km"])
    fit_rel_hz = float(fit["rel_doppler_hz"])
    sig_r = float(fit["sigma_range_km"])
    sig_f = float(fit["sigma_doppler_hz"])
    rho = float(fit.get("range_doppler_correlation", 0.0))

    raw = patch["raw_db"]
    filtered = patch["filtered_db"]
    filtered_unsmoothed = patch.get("filtered_unsmoothed_db", filtered)
    rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
    fit_model = gaussian2d(
        np.array(
            [
                float(fit["offset_db"]),
                float(fit["amplitude_db"]),
                fit_range_km,
                fit_rel_hz,
                sig_r,
                sig_f,
                rho,
            ]
        ),
        rr,
        ff,
    )
    x_min, x_max, y_min, y_max = gaussian_plot_window(seed_range_km, seed_rel_hz, fit_range_km, fit_rel_hz)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharex=True, sharey=True, constrained_layout=True)
    images = [
        ("Raw matched filter", raw),
        ("2D FFT notch filtered", filtered_unsmoothed),
        ("2D FFT filtered + smoothed", filtered),
    ]
    ell_x, ell_y = correlated_ellipse_points(fit_range_km, fit_rel_hz, sig_r, sig_f, rho)
    for ax, (title, img) in zip(axes, images, strict=True):
        show = (
            (range_km[:, None] >= y_min)
            & (range_km[:, None] <= y_max)
            & (rel_hz[None, :] >= x_min)
            & (rel_hz[None, :] <= x_max)
        )
        finite = img[show & np.isfinite(img)]
        vmin = float(np.nanpercentile(finite, 3.0))
        vmax = float(np.nanpercentile(finite, 99.6))
        mappable = ax.pcolormesh(rel_hz, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot([seed_rel_hz, fit_rel_hz], [seed_range_km, fit_range_km], color="white", lw=0.9, alpha=0.9)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=9, mew=1.4, color="#9be7ff", linestyle="none")
        ax.plot(fit_rel_hz, fit_range_km, marker="x", ms=8, mew=1.4, color="#42ff9f", linestyle="none")
        ax.plot(ell_x, ell_y, color="#42ff9f", lw=1.2)
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

    dr_m = (fit_range_km - seed_range_km) * 1000.0
    df_hz = fit_rel_hz - seed_rel_hz
    abs_doppler = patch["model_doppler_hz"] + fit_rel_hz
    title = (
        f"{STATION_LABELS.get(station, station)} at t={obs['unix_time'] - EVENT_T0:.3f} s: "
        f"Gaussian correction {dr_m:+.0f} m, {df_hz:+.1f} Hz; "
        f"refined Doppler {abs_doppler:.1f} Hz"
    )
    if "polarization_gamma" in patch and len(patch["polarization_gamma"]):
        gamma = np.asarray(patch["polarization_gamma"], dtype=np.complex64)
        title += f"; gamma=[{gamma[0]:.2f}, {gamma[1]:.2f}]"
    fig.suptitle(title, y=1.03, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default="clinton")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--time-rel", type=float, default=0.4)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=1)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--fft-filter-domain", choices=("db", "linear"), default="linear")
    p.add_argument("--inpaint-notched-columns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--polarization-combine", action="store_true")
    p.add_argument("--polarization-method", choices=("phase", "grid", "grid_image"), default="phase")
    p.add_argument("--polarization-image-metric", choices=("peak", "gaussian_1sigma"), default="peak")
    p.add_argument("--polarization-equalize-noise", action="store_true")
    p.add_argument("--polarization-equalize-block-samples", type=int, default=100000)
    p.add_argument("--polarization-grid-theta", type=int, default=91)
    p.add_argument("--polarization-grid-phase", type=int, default=361)
    p.add_argument("--polarization-phase-chunk", type=int, default=32)
    p.add_argument("--dc-cancel", action="store_true")
    p.add_argument("--sparse-voltage-subtract", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sparse-product-dir", type=Path, default=DEFAULT_SPARSE_PRODUCT_DIR)
    p.add_argument("--sparse-voltage-mode", choices=("full", "dc"), default="full")
    p.add_argument("--sparse-voltage-gain", type=float, default=1.0)
    p.add_argument("--search-range-half-width-km", type=float, default=10.0)
    p.add_argument("--search-doppler-half-width-hz", type=float, default=700.0)
    p.add_argument("--fit-range-half-width-km", type=float, default=5.0)
    p.add_argument("--fit-doppler-half-width-hz", type=float, default=300.0)
    p.add_argument("--smooth-range-pixels", type=float, default=4.0)
    p.add_argument("--smooth-frequency-pixels", type=float, default=16.0)
    p.add_argument("--cache", type=Path, default=OUT_CACHE)
    p.add_argument("--out", type=Path, default=OUT_FIG)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    md2d.DATA_DIR = args.data_dir

    if not args.plot_only:
        payload, observations = load_station_observations()
        if args.station not in observations:
            raise ValueError(f"Unknown station {args.station!r}")
        idx, obs = closest_observation(observations[args.station], args.time_rel)
        init_station(payload, args.station, args)
        md2d.G.pop("polarization_metric_mask", None)
        polarization_reference_fit = None
        polarization_metric_mask = None
        if (
            args.polarization_combine
            and args.polarization_method == "grid_image"
            and args.polarization_image_metric == "gaussian_1sigma"
        ):
            args.polarization_combine = False
            reference_patch = compute_filtered_patch((idx, obs))
            args.polarization_combine = True
            if reference_patch is None:
                raise RuntimeError("Selected observation is outside the available RF span")
            reference_filtered = smooth_power_db(
                reference_patch["filtered_db"],
                args.smooth_range_pixels,
                args.smooth_frequency_pixels,
            )
            seed_range_km_ref = float(obs["range_m"]) / 1000.0
            seed_rel_hz_ref = float(obs["doppler_hz"]) - float(reference_patch["model_doppler_hz"])
            polarization_reference_fit = fit_gaussian_patch(
                reference_patch["range_m"] / 1000.0,
                reference_patch["rel_freqs"],
                reference_filtered,
                seed_range_km_ref,
                seed_rel_hz_ref,
                args.search_range_half_width_km,
                args.search_doppler_half_width_hz,
                args.fit_range_half_width_km,
                args.fit_doppler_half_width_hz,
            )
            polarization_metric_mask = gaussian_one_sigma_mask(
                reference_patch["range_m"] / 1000.0,
                reference_patch["rel_freqs"],
                polarization_reference_fit,
            )
            md2d.G["polarization_metric_mask"] = polarization_metric_mask
        patch = compute_filtered_patch((idx, obs))
        if patch is None:
            raise RuntimeError("Selected observation is outside the available RF span")
        if polarization_metric_mask is not None:
            patch["polarization_metric_mask"] = polarization_metric_mask
            patch["polarization_reference_fit"] = polarization_reference_fit
        patch["filtered_unsmoothed_db"] = patch["filtered_db"]
        patch["filtered_db"] = smooth_power_db(
            patch["filtered_unsmoothed_db"],
            args.smooth_range_pixels,
            args.smooth_frequency_pixels,
        )
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
        fit.update(
            gaussian_power_snr(
                patch["range_m"] / 1000.0,
                patch["rel_freqs"],
                fit,
                seed_range_km,
                seed_rel_hz,
            )
        )
        save_example(args.cache, patch, fit, args.station, obs)
        print(args.cache, flush=True)
        print(
            f"{args.station} t={float(obs['unix_time']) - EVENT_T0:.3f}s "
            f"seed range={seed_range_km:.3f} km seed residual={seed_rel_hz:.1f} Hz "
            f"fit range={fit['range_km']:.3f} km fit residual={fit['rel_doppler_hz']:.1f} Hz",
            flush=True,
        )
    plot_example(args.cache, args.out)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
