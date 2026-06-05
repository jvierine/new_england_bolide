#!/usr/bin/env python3
"""Plot bolide orbital elements against local NEA and comet catalog densities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from comet_jopek_search import read_catalog as read_comet_catalog
from nea_jopek_search import DEFAULT_ORBIT, Orbit, parse_mpcorb_line, read_bolide_orbit


DEFAULT_NEA = Path("data/NEA.txt")
DEFAULT_COMETS = Path("data/CometEls.txt")
DEFAULT_OUT = Path("reports/figures/orbit_catalog_density.png")
JUPITER_A_AU = 5.20336301


def tisserand_jupiter(a_au: np.ndarray, e: np.ndarray, inc_deg: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        term = (a_au / JUPITER_A_AU) * (1.0 - e * e)
        return JUPITER_A_AU / a_au + 2.0 * np.cos(np.deg2rad(inc_deg)) * np.sqrt(term)


def orbit_to_record(orbit: Orbit) -> dict[str, float]:
    a_au = float(orbit.a_au)
    e = float(orbit.e)
    inc = float(orbit.i_deg)
    return {
        "a_au": a_au,
        "q_au": float(orbit.q_au),
        "e": e,
        "i_deg": inc,
        "node_deg": float(orbit.node_deg % 360.0),
        "t_j": float(tisserand_jupiter(np.array([a_au]), np.array([e]), np.array([inc]))[0]),
    }


def read_nea_records(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = parse_mpcorb_line(line)
        if rec is None:
            continue
        rows.append(rec)
    return rows


def numeric_catalog(rows: list[dict]) -> dict[str, np.ndarray]:
    a = np.array([float(row.get("a_au", np.nan)) for row in rows], dtype=float)
    e = np.array([float(row.get("e", np.nan)) for row in rows], dtype=float)
    inc = np.array([float(row.get("i_deg", np.nan)) for row in rows], dtype=float)
    node = np.array([float(row.get("node_deg", np.nan)) % 360.0 for row in rows], dtype=float)
    q = np.array([float(row.get("q_au", np.nan)) for row in rows], dtype=float)
    tj = tisserand_jupiter(a, e, inc)
    return {"a_au": a, "q_au": q, "e": e, "i_deg": inc, "node_deg": node, "t_j": tj}


def finite_pair(data: dict[str, np.ndarray], x_key: str, y_key: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x = data[x_key]
    y = data[y_key]
    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xlim[0])
        & (x <= xlim[1])
        & (y >= ylim[0])
        & (y <= ylim[1])
    )
    return x[good], y[good]


def smoothed_density(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bins: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    hist = gaussian_filter(hist.T, sigma=sigma, mode="nearest")
    if np.nanmax(hist) > 0:
        hist = hist / np.nanmax(hist)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, hist


def draw_density(
    ax,
    data: dict[str, np.ndarray],
    x_key: str,
    y_key: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: str,
    label: str,
    alpha: float,
    bins: int,
    sigma: float,
    zorder: int,
) -> None:
    x, y = finite_pair(data, x_key, y_key, xlim, ylim)
    if x.size < 20:
        return
    xx, yy, zz = smoothed_density(x, y, xlim, ylim, bins, sigma)
    levels = np.array([0.08, 0.18, 0.34, 0.54, 0.78])
    ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=alpha, antialiased=True, zorder=zorder)
    ax.contour(xx, yy, zz, levels=levels[1:], colors=[color], linewidths=0.8, alpha=min(0.95, alpha + 0.25), zorder=zorder + 0.1)
    ax.plot([], [], color=color, lw=5, alpha=alpha + 0.1, label=label)


def plot(args: argparse.Namespace) -> None:
    nea = numeric_catalog(read_nea_records(args.nea))
    comets = numeric_catalog(read_comet_catalog(args.comets))
    bolide = orbit_to_record(read_bolide_orbit(args.orbit_h5))

    panels = [
        ("a_au", "e", r"$a$ (AU)", r"$e$", (0.5, 5.2), (0.0, 1.0)),
        ("a_au", "i_deg", r"$a$ (AU)", r"$i$ (deg)", (0.5, 5.2), (0.0, 60.0)),
        ("t_j", "node_deg", r"$T_J$", r"$\Omega$ (deg)", (1.5, 5.0), (0.0, 360.0)),
        ("q_au", "e", r"$q$ (AU)", r"$e$", (0.0, 1.4), (0.0, 1.0)),
    ]

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    for ax, (x_key, y_key, x_label, y_label, xlim, ylim) in zip(axs.ravel(), panels, strict=True):
        draw_density(ax, nea, x_key, y_key, xlim, ylim, "#2f6fb0", "NEA catalogue", 0.34, args.bins, args.sigma, 1)
        draw_density(ax, comets, x_key, y_key, xlim, ylim, "#d95f02", "Comet catalogue", 0.32, args.bins, args.sigma, 2)
        ax.scatter(
            bolide[x_key],
            bolide[y_key],
            marker="*",
            s=260,
            color="black",
            edgecolor="white",
            linewidth=1.1,
            zorder=10,
            label="Cape Cod Bay bolide",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, color="0.90", lw=0.8, zorder=0)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Cape Cod Bay bolide orbital elements relative to local NEA and comet catalogues", y=1.075)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    if args.pdf:
        print(f"Wrote {args.out.with_suffix('.pdf')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2D catalogue-density marginals with the bolide orbit overlaid.")
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--nea", type=Path, default=DEFAULT_NEA)
    parser.add_argument("--comets", type=Path, default=DEFAULT_COMETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=220)
    parser.add_argument("--sigma", type=float, default=2.8)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
