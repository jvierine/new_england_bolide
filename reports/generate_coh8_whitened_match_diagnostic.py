#!/usr/bin/env python3
"""Try a whitened/generalized matched-filter patch for the eight-code head echo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_echo_gaussian_example import closest_observation  # noqa: E402
from reports.generate_head_microdoppler import detrend_window, load_station_observations  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import (  # noqa: E402
    EVENT_T0,
    G,
    best_fit_doppler_hz,
    init_station,
)

OUT_CACHE = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "cache"
    / "coh8_whitened_match_clinton_t040.h5"
)
OUT_FIG = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "figures"
    / "coh8_whitened_match_clinton_t040.png"
)


def whiten_matrix(samples: np.ndarray, shrinkage: float, eig_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and inverse-square-root covariance for rows of complex samples."""
    mu = np.mean(samples, axis=0)
    xc = samples - mu[None, :]
    cov = (xc.conj().T @ xc) / max(samples.shape[0] - 1, 1)
    diag_mean = float(np.real(np.trace(cov)) / cov.shape[0])
    cov = (1.0 - shrinkage) * cov + shrinkage * diag_mean * np.eye(cov.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    floor = max(eig_floor * diag_mean, 1e-18)
    vals = np.maximum(vals, floor)
    inv_sqrt = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.conj().T
    return mu, inv_sqrt, vals


def compute_complex_patch(idx: int, obs: dict, args: argparse.Namespace) -> dict:
    station = G["station"]
    bulk = best_fit_doppler_hz(G["doppler_model"], station, float(obs["unix_time"]))
    fvec_full = G["fvec_full"]
    fidx = np.where(np.abs(fvec_full - bulk) <= args.doppler_half_width_hz)[0]
    rel_f = fvec_full[fidx] - bulk

    gate_m = G["gate_m"]
    seed_range_m = float(obs["range_m"])
    ridx = np.arange(
        max(0, int(np.floor((seed_range_m - args.range_half_width_km * 1000.0) / gate_m))),
        int(np.ceil((seed_range_m + args.range_half_width_km * 1000.0) / gate_m)) + 1,
        dtype=np.int64,
    )

    raw_sample = int(round(float(obs["unix_time"]) * G["sample_rate"] - G["t0_sample"]))
    start_i = raw_sample * G["range_interp"]
    read_len = G["win_len"] + int(ridx[-1]) + 1
    incoh_stop_i = start_i + (args.n_incoh - 1) * G["step_len"] + read_len
    if start_i < 0 or incoh_stop_i >= min(map(len, G["zc"])):
        raise RuntimeError("Selected observation is outside the available RF span")

    chunks = []
    raw_power = np.zeros((len(ridx), len(fidx)), dtype=np.float64)
    for incoh_i in range(args.n_incoh):
        incoh_start_i = start_i + incoh_i * G["step_len"]
        for z in G["zc"]:
            z0 = detrend_window(z[incoh_start_i : incoh_start_i + read_len])
            windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
            for code in G["codes"]:
                zspec = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
                chunks.append(zspec.astype(np.complex64, copy=False))
                raw_power += np.abs(zspec) ** 2

    cube = np.asarray(chunks)  # component, range, frequency
    range_m = ridx.astype(np.float64) * gate_m
    bg_ranges = np.abs(range_m - seed_range_m) >= args.exclude_head_km * 1000.0
    if np.count_nonzero(bg_ranges) < 8:
        raise RuntimeError("Too few off-head ranges available for covariance estimate")

    samples = cube[:, bg_ranges, :].reshape(-1, cube.shape[-1]).astype(np.complex128, copy=False)
    mu, inv_sqrt, eigvals = whiten_matrix(samples, args.shrinkage, args.eig_floor)
    whitened_power = np.zeros_like(raw_power)
    for zspec in cube:
        centered = zspec.astype(np.complex128, copy=False) - mu[None, :]
        zw = centered @ inv_sqrt
        whitened_power += np.abs(zw) ** 2

    return {
        "idx": idx,
        "station": station,
        "time_rel": float(obs["unix_time"]) - EVENT_T0,
        "range_m": range_m,
        "rel_freqs": rel_f,
        "raw_power": raw_power,
        "whitened_power": whitened_power,
        "seed_range_m": seed_range_m,
        "seed_doppler_hz": float(obs["doppler_hz"]),
        "model_doppler_hz": bulk,
        "bg_range_mask": bg_ranges,
        "cov_eigvals": eigvals,
        "n_components": cube.shape[0],
    }


def save_patch(path: Path, patch: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Eight-code head-echo generalized matched-filter Doppler covariance whitening diagnostic"
        h.attrs["station"] = patch["station"]
        h.attrs["time_rel"] = patch["time_rel"]
        h.attrs["seed_range_m"] = patch["seed_range_m"]
        h.attrs["seed_doppler_hz"] = patch["seed_doppler_hz"]
        h.attrs["model_doppler_hz"] = patch["model_doppler_hz"]
        h.attrs["n_components"] = patch["n_components"]
        h.attrs["n_coh"] = args.n_coh
        h.attrs["n_incoh"] = args.n_incoh
        h.attrs["exclude_head_km"] = args.exclude_head_km
        h.attrs["shrinkage"] = args.shrinkage
        h.attrs["eig_floor"] = args.eig_floor
        for key in ("range_m", "rel_freqs", "raw_power", "whitened_power", "bg_range_mask", "cov_eigvals"):
            h[key] = patch[key]


def load_patch(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        return {
            "station": str(h.attrs["station"]),
            "time_rel": float(h.attrs["time_rel"]),
            "seed_range_m": float(h.attrs["seed_range_m"]),
            "seed_doppler_hz": float(h.attrs["seed_doppler_hz"]),
            "model_doppler_hz": float(h.attrs["model_doppler_hz"]),
            "range_m": h["range_m"][()],
            "rel_freqs": h["rel_freqs"][()],
            "raw_power": h["raw_power"][()],
            "whitened_power": h["whitened_power"][()],
        }


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-30))


def plot_patch(path: Path, out: Path, range_pad_km: float, doppler_pad_hz: float) -> None:
    patch = load_patch(path)
    range_km = patch["range_m"] / 1000.0
    rel_hz = patch["rel_freqs"]
    seed_range_km = patch["seed_range_m"] / 1000.0
    seed_rel_hz = patch["seed_doppler_hz"] - patch["model_doppler_hz"]
    images = [
        ("Raw full-waveform power", to_db(patch["raw_power"]), "dB"),
        ("Whitened generalized MF power", to_db(patch["whitened_power"]), "dB"),
    ]

    keep = (
        (range_km[:, None] >= seed_range_km - range_pad_km)
        & (range_km[:, None] <= seed_range_km + range_pad_km)
        & (rel_hz[None, :] >= seed_rel_hz - doppler_pad_hz)
        & (rel_hz[None, :] <= seed_rel_hz + doppler_pad_hz)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, img, label) in zip(axes, images, strict=True):
        finite = img[keep & np.isfinite(img)]
        vmin = float(np.nanpercentile(finite, 2.0))
        vmax = float(np.nanpercentile(finite, 99.5))
        im = ax.pcolormesh(rel_hz, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=10, mew=1.5, color="#8ee7ff", linestyle="none")
        ax.set_title(title)
        ax.set_xlabel(r"$f - f_{D,\mathrm{model}}$ (Hz)")
        ax.set_xlim(seed_rel_hz - doppler_pad_hz, seed_rel_hz + doppler_pad_hz)
        ax.set_ylim(seed_range_km - range_pad_km, seed_range_km + range_pad_km)
        ax.grid(color="white", alpha=0.12, lw=0.5)
        cb = fig.colorbar(im, ax=ax, pad=0.012)
        cb.set_label(label)
    axes[0].set_ylabel("Bistatic range (km)")
    fig.suptitle(f"{patch['station']} t={patch['time_rel']:.3f} s, eight-code Doppler-covariance whitening", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default="clinton")
    p.add_argument("--time-rel", type=float, default=0.4)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=8)
    p.add_argument("--n-incoh", type=int, default=8)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--exclude-head-km", type=float, default=8.0)
    p.add_argument("--shrinkage", type=float, default=0.08)
    p.add_argument("--eig-floor", type=float, default=1e-4)
    p.add_argument("--range-pad-km", type=float, default=22.0)
    p.add_argument("--doppler-pad-hz", type=float, default=900.0)
    p.add_argument("--cache", type=Path, default=OUT_CACHE)
    p.add_argument("--out", type=Path, default=OUT_FIG)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    if not args.plot_only:
        payload, observations = load_station_observations()
        idx, obs = closest_observation(observations[args.station], args.time_rel)
        init_station(payload, args.station, args)
        patch = compute_complex_patch(idx, obs, args)
        save_patch(args.cache, patch, args)
        print(args.cache)
    plot_patch(args.cache, args.out, args.range_pad_km, args.doppler_pad_hz)


if __name__ == "__main__":
    main()
