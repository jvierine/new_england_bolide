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
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    EVENT_T0,
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
    p0 = np.array([floor, amp0, peak_range_km, peak_rel_hz, max(1.2, min_sig_r), max(80.0, min_sig_f), 0.0], dtype=float)
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
    x_pad = 650.0
    y_pad = 14.0
    x_min = min(seed_rel_hz, fit_rel_hz) - x_pad
    x_max = max(seed_rel_hz, fit_rel_hz) + x_pad
    y_min = min(seed_range_km, fit_range_km) - y_pad
    y_max = max(seed_range_km, fit_range_km) + y_pad

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
    fig.suptitle(title, y=1.03, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default="clinton")
    p.add_argument("--time-rel", type=float, default=0.4)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=16)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--dc-cancel", action="store_true")
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

    if not args.plot_only:
        payload, observations = load_station_observations()
        if args.station not in observations:
            raise ValueError(f"Unknown station {args.station!r}")
        idx, obs = closest_observation(observations[args.station], args.time_rel)
        init_station(payload, args.station, args)
        patch = compute_filtered_patch((idx, obs))
        if patch is None:
            raise RuntimeError("Selected observation is outside the available RF span")
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
