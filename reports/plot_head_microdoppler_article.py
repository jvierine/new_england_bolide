#!/usr/bin/env python3
"""Render the article micro-Doppler figure from the archived spectral cache."""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        default=Path("reports/head_echo_microdoppler_2dfft_notched.h5"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/figures/head_echo_microdoppler.png"))
    args = parser.parse_args()
    labels = {"beverly": "Beverly, MA", "clinton": "Clinton, MA", "pomfret": "Pomfret, VT"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True,
                             sharey=True, constrained_layout=True)
    with h5py.File(args.cache, "r") as cache:
        for ax, (station, label) in zip(axes, labels.items()):
            group = cache[station]
            ax.set_facecolor("black")
            mesh = ax.pcolormesh(group["time_rel"][:], group["rel_freqs"][:],
                                 group["spec_db"][:].T, shading="auto",
                                 cmap="inferno", vmin=0, vmax=8)
            ax.axhline(0, color="white", alpha=0.5, linestyle="--", linewidth=0.8)
            ax.set_ylim(-500, 500)
            ax.set_ylabel(r"$\Delta f_d$ (Hz)")
            ax.set_title(f"{label} head echo")
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    fig.colorbar(mesh, ax=axes, pad=0.012).set_label("SNR (dB)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
