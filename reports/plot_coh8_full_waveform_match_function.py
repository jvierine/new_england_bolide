#!/usr/bin/env python3
"""Plot the raw eight-code full-waveform head-echo match function."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "cache"
    / "head_echo_gaussian_2dfft_example_clinton_t040_ncoh8.h5"
)
DEFAULT_OUT = (
    ROOT
    / "reports"
    / "coh8_head_echo_test"
    / "figures"
    / "coh8_full_waveform_raw_match_clinton_t040.png"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--range-pad-km", type=float, default=22.0)
    p.add_argument("--doppler-pad-hz", type=float, default=900.0)
    args = p.parse_args()

    with h5py.File(args.cache, "r") as h:
        range_km = h["range_m"][()] / 1000.0
        rel_hz = h["rel_freqs"][()]
        raw_db = h["raw_db"][()]
        station = h.attrs["station"]
        time_rel = float(h.attrs["time_rel"])
        seed_range_km = float(h.attrs["seed_range_m"]) / 1000.0
        seed_rel_hz = float(h.attrs["seed_doppler_hz"]) - float(h.attrs["model_doppler_hz"])

    keep = (
        (range_km[:, None] >= seed_range_km - args.range_pad_km)
        & (range_km[:, None] <= seed_range_km + args.range_pad_km)
        & (rel_hz[None, :] >= seed_rel_hz - args.doppler_pad_hz)
        & (rel_hz[None, :] <= seed_rel_hz + args.doppler_pad_hz)
    )
    finite = raw_db[keep & np.isfinite(raw_db)]
    if finite.size == 0:
        raise ValueError("Requested plotting window contains no finite samples")

    vmin = float(np.nanpercentile(finite, 2.0))
    vmax = float(np.nanpercentile(finite, 99.7))

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    im = ax.pcolormesh(rel_hz, range_km, raw_db, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.plot(seed_rel_hz, seed_range_km, marker="+", ms=10, mew=1.6, color="#8ee7ff", linestyle="none")
    ax.set_xlim(seed_rel_hz - args.doppler_pad_hz, seed_rel_hz + args.doppler_pad_hz)
    ax.set_ylim(seed_range_km - args.range_pad_km, seed_range_km + args.range_pad_km)
    ax.set_xlabel(r"$f - f_{D,\mathrm{model}}$ (Hz)")
    ax.set_ylabel("Bistatic range (km)")
    ax.set_title(
        f"{station} t={time_rel:.3f} s: raw eight-code full-waveform matched-filter power"
    )
    ax.grid(color="white", alpha=0.12, lw=0.5)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("Raw matched-filter power (dB)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
