#!/usr/bin/env python3
"""Plot atmospheric mean free path for the Cape Cod Bay bolide."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymsis import msis


OUT = Path(__file__).resolve().parent / "figures" / "mean_free_path_profile.png"
OUT_PDF = OUT.with_suffix(".pdf")
EFFECTIVE_COLLISION_DIAMETER_M = 3.7e-10


def total_number_density_m3(time_dt64, lat_deg, lon_deg, alt_km):
    alt_km = np.asarray(alt_km, dtype=float)
    times = np.full(alt_km.shape, np.datetime64(time_dt64, "s"), dtype="datetime64[s]")
    lats = np.full(alt_km.shape, float(lat_deg), dtype=float)
    lons = np.full(alt_km.shape, float(lon_deg), dtype=float)
    data = np.asarray(msis.run(times, lats, lons, alt_km, geomagnetic_activity=-1), dtype=float)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data[None, :]
    return np.nansum(data[..., 1:-1], axis=-1)


def mean_free_path_m(number_density_m3):
    sigma_m2 = np.pi * EFFECTIVE_COLLISION_DIAMETER_M**2
    return 1.0 / (np.sqrt(2.0) * sigma_m2 * np.maximum(number_density_m3, 1e-30))


def main() -> None:
    alt_km = np.linspace(40.0, 100.0, 601)
    number_density = total_number_density_m3(
        time_dt64=np.datetime64("2026-05-30T18:06:20"),
        lat_deg=41.98,
        lon_deg=-70.66,
        alt_km=alt_km,
    )
    mfp_m = mean_free_path_m(number_density)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.semilogx(mfp_m, alt_km, color="black", lw=2.0, label=r"$\lambda_{\rm mfp}$")
    ax.semilogx(100.0 * mfp_m, alt_km, color="#1f77b4", lw=1.8, label=r"$100\lambda_{\rm mfp}$")
    ax.set_xlabel("Atmospheric mean free path (m)")
    ax.set_ylabel("Altitude (km)")
    ax.set_ylim(40, 100)
    ax.set_xlim(1e-5, 1e2)
    ax.grid(True, which="both", color="0.86", lw=0.7)

    for y, label in [(72.5, "72.5 km"), (90.0, "90 km")]:
        ax.axhline(y, color="0.45", lw=1.0, ls="-.", alpha=0.75)
        ax.text(
            0.98,
            y + 0.8,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="0.25",
            fontsize=10,
        )

    for x, label in [(1e-3, "1 mm"), (1e-2, "1 cm"), (1e-1, "10 cm"), (1.0, "1 m")]:
        ax.axvline(x, color="0.35", lw=0.9, ls="--", alpha=0.5)
        ax.text(x, 41.0, label, rotation=90, ha="right", va="bottom", color="0.25", fontsize=10)

    ax.text(
        0.03,
        0.97,
        "NRLMSIS 2.0\n2026-05-30 18:06 UTC\n41.98 N, 70.66 W",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 4},
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
