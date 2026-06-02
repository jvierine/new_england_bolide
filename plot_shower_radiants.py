#!/usr/bin/env python3
"""Plot nearby established shower radiants against the corrected bolide radiant."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from meteor_source_search import DEFAULT_CATALOG, DEFAULT_ORBIT, DEFAULT_REFERENCE_FILTERS, read_corrected_radiant, rank_streams
from nea_jopek_search import read_bolide_orbit


DEFAULT_OUT = Path("reports/figures/shower_radiant_comparison.png")


def wrap_delta_ra_deg(ra_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return (ra_deg - center_deg + 180.0) % 360.0 - 180.0


def radiant_samples(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    if radiant.ndim != 2 or radiant.shape[0] != 2:
        raise ValueError(f"Unexpected radiant_orbit_GCRS shape in {path}: {radiant.shape}")
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    return radiant[:, finite]


def covariance_ellipse(samples: np.ndarray, ra0: float, dec0: float) -> Ellipse | None:
    if samples.shape[1] < 3:
        return None
    x = wrap_delta_ra_deg(samples[0], ra0) * math.cos(math.radians(dec0))
    y = samples[1] - dec0
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    chi2_95 = 5.991464547
    width, height = 2.0 * np.sqrt(vals * chi2_95)
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse((ra0, dec0), width / math.cos(math.radians(dec0)), height, angle=angle)


def select_nearby(rows: list[dict], top_orbit: int, top_radiant: int) -> list[dict]:
    by_orbit = rows[:top_orbit]
    by_radiant = sorted(rows, key=lambda row: row["radiant_sep_deg"])[:top_radiant]
    selected = {}
    for row in by_orbit + by_radiant:
        selected[(row["code"], row["name"], row["reference"])] = row
    return sorted(selected.values(), key=lambda row: row["radiant_sep_deg"])


def marker_sizes(rows: list[dict]) -> np.ndarray:
    n = np.array([max(float(row.get("n", 0) or 0), 1.0) for row in rows], dtype=float)
    return 35.0 + 22.0 * np.sqrt(np.clip(n, 1.0, 2000.0))


def plot(args: argparse.Namespace) -> Path:
    bolide = read_bolide_orbit(args.orbit_h5)
    ra0, dec0 = read_corrected_radiant(args.orbit_h5)
    samples = radiant_samples(args.orbit_h5)
    rows = rank_streams(
        args.catalog,
        bolide,
        ra0,
        dec0,
        reference_filters=tuple(args.reference_filter),
    )
    rows = select_nearby(rows, args.top_orbit, args.top_radiant)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)

    ra = np.array([row["ra_deg"] for row in rows], dtype=float)
    dec = np.array([row["dec_deg"] for row in rows], dtype=float)
    dra = wrap_delta_ra_deg(ra, ra0)
    plot_ra = ra0 + dra
    dh = np.array([row["D_H"] for row in rows], dtype=float)

    sc = ax.scatter(
        plot_ra,
        dec,
        s=marker_sizes(rows),
        c=dh,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.86,
        label="Established showers",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label(r"Jopek $D_H$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    for row, x, y in zip(rows, plot_ra, dec, strict=True):
        label = row["code"] or row["name"][:5]
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)

    if samples.shape[1] > 1:
        sx = ra0 + wrap_delta_ra_deg(samples[0], ra0)
        ax.scatter(sx, samples[1], s=10, color="0.2", alpha=0.16, linewidths=0, label="Orbit samples")
    ellipse = covariance_ellipse(samples, ra0, dec0)
    if ellipse is not None:
        ellipse.set_facecolor("tab:red")
        ellipse.set_alpha(0.18)
        ellipse.set_edgecolor("tab:red")
        ellipse.set_linewidth(1.8)
        ax.add_patch(ellipse)
    ax.scatter([ra0], [dec0], marker="*", s=260, color="tab:red", edgecolor="black", linewidth=0.8, zorder=5)
    ax.annotate("Bolide corrected radiant", (ra0, dec0), xytext=(8, -14), textcoords="offset points", fontsize=11)

    max_abs_dra = max(args.min_half_width_deg, float(np.nanmax(np.abs(dra))) + 8.0)
    dec_values = np.r_[dec, dec0, samples[1] if samples.size else []]
    ypad = 8.0
    ax.set_xlim(ra0 + max_abs_dra, ra0 - max_abs_dra)
    ax.set_ylim(max(-90.0, np.nanmin(dec_values) - ypad), min(90.0, np.nanmax(dec_values) + ypad))
    ax.set_xlabel(r"Right ascension, J2000 (deg)", fontsize=14)
    ax.set_ylabel(r"Declination, J2000 (deg)", fontsize=14)
    ax.set_title("Established Shower Radiants Near the Corrected Bolide Radiant", fontsize=15)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=10, frameon=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return args.out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-orbit", type=int, default=10)
    parser.add_argument("--top-radiant", type=int, default=22)
    parser.add_argument("--min-half-width-deg", type=float, default=35.0)
    parser.add_argument(
        "--reference-filter",
        action="append",
        default=list(DEFAULT_REFERENCE_FILTERS),
        help="Keep only catalogue rows whose References field contains this token. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    out = plot(parse_args())
    print(out)


if __name__ == "__main__":
    main()
