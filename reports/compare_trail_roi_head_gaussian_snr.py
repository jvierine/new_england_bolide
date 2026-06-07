#!/usr/bin/env python3
"""Compare trail-ROI SNR and head-echo Gaussian matched-filter SNR.

The trail ROI figure uses a picked Doppler-bin matched-filter power divided by
the per-range noise estimate. This script reads the underlying trail RDMF HDF5
product to determine that Doppler-bin bandwidth, then applies the same noise
bandwidth to the fitted 16-code head-echo Gaussian blob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


EVENT_T0 = 1780164380.0


def gaussian2d(params: np.ndarray, range_km: np.ndarray, rel_hz: np.ndarray) -> np.ndarray:
    offset, amp, r0, f0, sig_r, sig_f, rho = params
    rr = (range_km - r0) / sig_r
    ff = (rel_hz - f0) / sig_f
    one_minus_rho2 = max(1.0 - rho * rho, 1e-6)
    exponent = -0.5 * (rr * rr - 2.0 * rho * rr * ff + ff * ff) / one_minus_rho2
    return offset + amp * np.exp(exponent)


def ellipse_points(
    range_km: float,
    rel_hz: float,
    sigma_range_km: float,
    sigma_doppler_hz: float,
    rho: float,
    n: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    cov = np.array(
        [
            [sigma_range_km**2, rho * sigma_range_km * sigma_doppler_hz],
            [rho * sigma_range_km * sigma_doppler_hz, sigma_doppler_hz**2],
        ],
        dtype=float,
    )
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.vstack([np.cos(theta), np.sin(theta)])
    pts = np.array([range_km, rel_hz])[:, None] + vecs @ (np.sqrt(vals)[:, None] * unit)
    return pts[1], pts[0]


def db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def read_trail_product(path: Path, event_t0: float, example_time_rel_s: float) -> dict:
    with h5py.File(path, "r") as h:
        sample_rate = float(h["sample_rate"][()])
        fvec = h["fvec"][()]
        df_hz = float(np.median(np.diff(fvec)))
        t_rel = h["t_sample"][()] / sample_rate - event_t0
        range_km = h["range_km"][()]
        power = h["P_rti"][()] if "P_rti" in h else h["P"][()]
        noise = h["N"][()]
        doppler = h["D_rti"][()] if "D_rti" in h else h["D"][()]
        n_coh = int(h["n_coh"][()])
        code_len = int(h["code_len"][()])

    snr_db = db(power / np.maximum(noise, 1e-300))
    ti = int(np.argmin(np.abs(t_rel - example_time_rel_s)))
    ri = int(np.nanargmax(snr_db[ti, :]))
    near_zero = np.abs(fvec) < 100.0
    return {
        "t_rel": t_rel,
        "range_km": range_km,
        "snr_db": snr_db,
        "doppler_hz": doppler,
        "doppler_bin_hz": df_hz,
        "coherent_window_s": n_coh * code_len / sample_rate,
        "near_zero_gate_hz": 200.0,
        "near_zero_accepted_bins": int(np.sum(near_zero)),
        "near_zero_accepted_bandwidth_hz": float(np.sum(near_zero) * df_hz),
        "near_zero_min_hz": float(np.min(fvec[near_zero])),
        "near_zero_max_hz": float(np.max(fvec[near_zero])),
        "example_time_rel_s": float(t_rel[ti]),
        "example_snr_db": float(snr_db[ti, ri]),
        "example_range_km": float(range_km[ri]),
        "example_doppler_hz": float(doppler[ti, ri]),
    }


def read_head_gaussian(path: Path, noise_bandwidth_hz: float) -> dict:
    with h5py.File(path, "r") as h:
        range_km = h["range_m"][()] / 1000.0
        rel_hz = h["rel_freqs"][()]
        image_db = h["filtered_db"][()]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
        fit = {k: (v.item() if hasattr(v, "item") else v) for k, v in h["gaussian_fit"].attrs.items()}

    rr, ff = np.meshgrid(range_km, rel_hz, indexing="ij")
    params = np.array(
        [
            fit["offset_db"],
            fit["amplitude_db"],
            fit["range_km"],
            fit["rel_doppler_hz"],
            fit["sigma_range_km"],
            fit["sigma_doppler_hz"],
            fit["range_doppler_correlation"],
        ],
        dtype=float,
    )
    model_db = gaussian2d(params, rr, ff)
    floor_power = 10.0 ** (float(fit["offset_db"]) / 10.0)
    signal_power = float(np.sum(np.maximum(10.0 ** (model_db / 10.0) - floor_power, 0.0)))
    df_hz = float(np.median(np.diff(rel_hz)))
    noise_power = float(floor_power * noise_bandwidth_hz / df_hz)
    snr_linear = signal_power / max(noise_power, 1e-300)
    return {
        "range_km": range_km,
        "rel_hz": rel_hz,
        "image_db": image_db,
        "fit": fit,
        "attrs": attrs,
        "doppler_bin_hz": df_hz,
        "noise_bandwidth_hz": float(noise_bandwidth_hz),
        "noise_equivalent_bins": float(noise_bandwidth_hz / df_hz),
        "signal_power_gaussian": signal_power,
        "floor_power_per_bin": floor_power,
        "noise_power": noise_power,
        "snr_db": float(10.0 * np.log10(max(snr_linear, 1e-300))),
    }


def draw(out_png: Path, trail: dict, head: dict, time_window: tuple[float, float], range_window: tuple[float, float]) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    t = trail["t_rel"]
    r = trail["range_km"]
    keep_t = (t >= time_window[0]) & (t <= time_window[1])
    keep_r = (r >= range_window[0]) & (r <= range_window[1])
    ax = axes[0]
    im = ax.pcolormesh(t[keep_t], r[keep_r], trail["snr_db"][np.ix_(keep_t, keep_r)].T, shading="auto", cmap="inferno")
    ax.axvline(trail["example_time_rel_s"], color="cyan", lw=1.0)
    ax.set_title("Trail ROI product")
    ax.set_xlabel("seconds after 18:06:20 UTC")
    ax.set_ylabel("range (km)")
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                "SNR = picked-bin P/N",
                f"noise BW = {trail['doppler_bin_hz']:.1f} Hz",
                "ROI gate: |D| < 100 Hz",
                f"gate bins: {trail['near_zero_min_hz']:.1f}..{trail['near_zero_max_hz']:.1f} Hz",
                f"example: {trail['example_snr_db']:.1f} dB",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
    )
    fig.colorbar(im, ax=ax, label="SNR (dB)")

    ax = axes[1]
    fit = head["fit"]
    x0 = float(fit["rel_doppler_hz"])
    y0 = float(fit["range_km"])
    keep_f = (head["rel_hz"] >= x0 - 750.0) & (head["rel_hz"] <= x0 + 750.0)
    keep_r = (head["range_km"] >= y0 - 16.0) & (head["range_km"] <= y0 + 16.0)
    view = head["image_db"][np.ix_(keep_r, keep_f)]
    im = ax.pcolormesh(head["rel_hz"][keep_f], head["range_km"][keep_r], view, shading="auto", cmap="viridis")
    ex, ey = ellipse_points(
        y0,
        x0,
        float(fit["sigma_range_km"]),
        float(fit["sigma_doppler_hz"]),
        float(fit["range_doppler_correlation"]),
    )
    ax.plot(ex, ey, color="lime", lw=1.4)
    ax.plot([x0], [y0], marker="+", color="lime", ms=12, mew=2.0)
    ax.set_title("Head 16-code Gaussian fit")
    ax.set_xlabel("Doppler residual (Hz)")
    ax.set_ylabel("range (km)")
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"coherent window = {head['attrs'].get('coherent_window_s', float('nan')):.2f} s",
                f"head df = {head['doppler_bin_hz']:.2f} Hz",
                f"noise BW = {head['noise_bandwidth_hz']:.1f} Hz",
                f"SNR = {head['snr_db']:.1f} dB",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
    )
    fig.colorbar(im, ax=ax, label="filtered matched-filter power (dB)")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-rdmf", type=Path, required=True, help="Trail ROI/RDMF HDF5 product for one station.")
    parser.add_argument("--head-gaussian-h5", type=Path, required=True, help="Head Gaussian example HDF5 product.")
    parser.add_argument("--event-t0", type=float, default=EVENT_T0)
    parser.add_argument("--example-time-rel-s", type=float, default=0.4)
    parser.add_argument("--time-window", type=float, nargs=2, default=(-0.05, 0.85))
    parser.add_argument("--range-window-km", type=float, nargs=2, default=(220.0, 285.0))
    parser.add_argument("--out-json", type=Path, default=Path("reports/trail_roi_head_gaussian_snr_comparison.json"))
    parser.add_argument("--out-png", type=Path, default=Path("reports/figures/trail_roi_head_gaussian_snr_comparison.png"))
    args = parser.parse_args()

    trail = read_trail_product(args.trail_rdmf, args.event_t0, args.example_time_rel_s)
    head = read_head_gaussian(args.head_gaussian_h5, noise_bandwidth_hz=trail["doppler_bin_hz"])
    draw(args.out_png, trail, head, tuple(args.time_window), tuple(args.range_window_km))

    payload = {
        "trail_roi_figure": {
            "snr_definition": "picked Doppler-bin matched-filter power divided by per-range noise",
            "noise_bandwidth_hz": trail["doppler_bin_hz"],
            "coherent_window_s": trail["coherent_window_s"],
            "near_zero_gate_hz": trail["near_zero_gate_hz"],
            "near_zero_accepted_bandwidth_hz": trail["near_zero_accepted_bandwidth_hz"],
            "example_time_rel_s": trail["example_time_rel_s"],
            "example_snr_db": trail["example_snr_db"],
            "example_range_km": trail["example_range_km"],
            "example_doppler_hz": trail["example_doppler_hz"],
        },
        "head_gaussian_example": {
            "snr_definition": "fitted Gaussian excess power divided by fitted flat noise floor scaled to the trail figure noise bandwidth",
            "noise_bandwidth_hz": head["noise_bandwidth_hz"],
            "head_doppler_bin_hz": head["doppler_bin_hz"],
            "noise_equivalent_head_bins": head["noise_equivalent_bins"],
            "signal_power_gaussian": head["signal_power_gaussian"],
            "noise_power": head["noise_power"],
            "snr_db": head["snr_db"],
            "coherent_window_s": head["attrs"].get("coherent_window_s"),
            "fit_range_km": head["fit"]["range_km"],
            "fit_rel_doppler_hz": head["fit"]["rel_doppler_hz"],
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
