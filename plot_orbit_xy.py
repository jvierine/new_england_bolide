#!/usr/bin/env python3
"""Plot the New England bolide orbit in the heliocentric ecliptic x-y plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyorb
from astropy.time import Time


AU = 1.495978707e11
DEFAULT_ORBIT = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_SAMPLES = Path("products/orbit_determination/orbit_mc_40.h5")
DEFAULT_OUT = Path("plots/orbit_xy_bolide.png")

J2000 = Time("2000-01-01T12:00:00", scale="tdb")

# NASA/JPL approximate planetary mean orbits, J2000, with centennial rates.
# a is AU, angles are degrees, angular rates are arcseconds/Julian century.
PLANETS = {
    "Mercury": {"a": 0.38709893, "e": 0.20563069, "i": 7.00487, "Omega": 48.33167, "varpi": 77.45645, "L": 252.25084, "da": 0.00000066, "de": 0.00002527, "di": -23.51, "dOmega": -446.30, "dvarpi": 573.57, "dL": 538101628.29},
    "Venus": {"a": 0.72333199, "e": 0.00677323, "i": 3.39471, "Omega": 76.68069, "varpi": 131.53298, "L": 181.97973, "da": 0.00000092, "de": -0.00004938, "di": -2.86, "dOmega": -996.89, "dvarpi": -108.80, "dL": 210664136.06},
    "Earth": {"a": 1.00000011, "e": 0.01671022, "i": 0.00005, "Omega": -11.26064, "varpi": 102.94719, "L": 100.46435, "da": -0.00000005, "de": -0.00003804, "di": -46.94, "dOmega": -18228.25, "dvarpi": 1198.28, "dL": 129597740.63},
    "Mars": {"a": 1.52366231, "e": 0.09341233, "i": 1.85061, "Omega": 49.57854, "varpi": 336.04084, "L": 355.45332, "da": -0.00007221, "de": 0.00011902, "di": -25.47, "dOmega": -1020.19, "dvarpi": 1560.78, "dL": 68905103.78},
    "Jupiter": {"a": 5.20336301, "e": 0.04839266, "i": 1.30530, "Omega": 100.55615, "varpi": 14.75385, "L": 34.40438, "da": 0.00060737, "de": -0.00012880, "di": -4.15, "dOmega": 1217.17, "dvarpi": 839.93, "dL": 10925078.35},
    "Saturn": {"a": 9.53707032, "e": 0.05415060, "i": 2.48446, "Omega": 113.71504, "varpi": 92.43194, "L": 49.94432, "da": -0.00301530, "de": -0.00036762, "di": 6.11, "dOmega": -1591.05, "dvarpi": -1948.89, "dL": 4401052.95},
    "Uranus": {"a": 19.19126393, "e": 0.04716771, "i": 0.76986, "Omega": 74.22988, "varpi": 170.96424, "L": 313.23218, "da": 0.00152025, "de": -0.00019150, "di": -2.09, "dOmega": -1681.40, "dvarpi": 1312.56, "dL": 1542547.79},
    "Neptune": {"a": 30.06896348, "e": 0.00858587, "i": 1.76917, "Omega": 131.72169, "varpi": 44.97135, "L": 304.88003, "da": -0.00125196, "de": 0.00002510, "di": -3.64, "dOmega": -151.25, "dvarpi": -844.43, "dL": 786449.21},
    "Pluto": {"a": 39.48168677, "e": 0.24880766, "i": 17.14175, "Omega": 110.30347, "varpi": 224.06676, "L": 238.92881, "da": -0.00076912, "de": 0.00006465, "di": 11.07, "dOmega": -37.33, "dvarpi": -132.25, "dL": 522747.90},
}

PLANET_COLORS = {
    "Mercury": "0.50",
    "Venus": "#a57c48",
    "Earth": "#2f70b7",
    "Mars": "#b5533a",
    "Jupiter": "#8b6f4e",
    "Saturn": "#b59b64",
    "Uranus": "#54a6b5",
    "Neptune": "#3d5fa8",
    "Pluto": "0.45",
}


def kepler_to_xy(kepler: np.ndarray, n_points: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y orbit points in AU for one pyorb Kepler vector."""
    kep = np.array(kepler, dtype=float).copy()
    kep[5] = 0.0
    anomalies = np.linspace(0.0, 360.0, n_points, endpoint=True)
    orb = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=n_points)
    values = np.tile(kep[:, None], (1, n_points))
    values[5, :] = anomalies
    orb.kepler = values
    cart = orb.cartesian
    return cart[0] / AU, cart[1] / AU


def true_anomaly_from_mean(mean_anomaly_deg: float, eccentricity: float) -> float:
    mean_anomaly = np.deg2rad(mean_anomaly_deg % 360.0)
    ecc_anomaly = mean_anomaly
    for _ in range(12):
        ecc_anomaly -= (ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * np.cos(ecc_anomaly)
        )
    true_anomaly = np.arctan2(
        np.sqrt(1.0 - eccentricity * eccentricity) * np.sin(ecc_anomaly),
        np.cos(ecc_anomaly) - eccentricity,
    )
    return float(np.rad2deg(true_anomaly) % 360.0)


def planet_kepler(name: str, epoch: Time) -> np.ndarray:
    elements = PLANETS[name]
    centuries = (epoch.tdb.jd - J2000.jd) / 36525.0
    a = elements["a"] + elements["da"] * centuries
    e = elements["e"] + elements["de"] * centuries
    inc = elements["i"] + elements["di"] * centuries / 3600.0
    raan = elements["Omega"] + elements["dOmega"] * centuries / 3600.0
    varpi = elements["varpi"] + elements["dvarpi"] * centuries / 3600.0
    mean_longitude = elements["L"] + elements["dL"] * centuries / 3600.0
    argp = (varpi - raan) % 360.0
    mean_anomaly = (mean_longitude - varpi) % 360.0
    true_anomaly = true_anomaly_from_mean(mean_anomaly, e)
    return np.array([a * AU, e, inc, argp, raan % 360.0, true_anomaly], dtype=float)


def read_kepler(path: Path) -> tuple[np.ndarray, str, float]:
    with h5py.File(path, "r") as h:
        kepler = h["kepler_HeliocentricMeanEcliptic_final"][()]
        epoch_utc = h.attrs["t_ref_utc"]
        t_ref_unix = float(h.attrs["t_ref_unix"])
    return kepler, epoch_utc, t_ref_unix


def draw_orbits(
    ax,
    best: np.ndarray,
    sample_kepler: np.ndarray,
    epoch: Time,
    limits: tuple[float, float],
    label_names: set[str],
) -> int:
    for name in PLANETS:
        color = PLANET_COLORS[name]
        kep = planet_kepler(name, epoch)
        x, y = kepler_to_xy(kep, n_points=900)
        ax.plot(x, y, color=color, lw=0.9, alpha=0.72, zorder=2)
        orb = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
        orb.kepler = kep[:, None]
        px, py = orb.cartesian[:2, 0] / AU
        size = 60 if name in {"Jupiter", "Saturn"} else 42
        if name in {"Uranus", "Neptune", "Pluto"}:
            size = 34
        ax.scatter(px, py, s=size, color=color, edgecolor="white", linewidth=0.5, zorder=6)
        if name in label_names:
            dx = 0.18 if px < limits[1] - 4 else -2.8
            dy = 0.18 if py < limits[1] - 4 else -1.2
            x_text = np.clip(px + dx, limits[0] + 0.4, limits[1] - 3.0)
            y_text = np.clip(py + dy, limits[0] + 0.8, limits[1] - 0.8)
            ax.text(x_text, y_text, name, fontsize=8, color="0.2", zorder=7, clip_on=True)

    finite_samples = 0
    cloud_points = []
    for i in range(sample_kepler.shape[1]):
        kep = sample_kepler[:, i]
        if not np.all(np.isfinite(kep)) or kep[0] <= 0 or kep[1] >= 1:
            continue
        x, y = kepler_to_xy(kep, n_points=500)
        cloud_points.append(np.column_stack((x, y)))
        finite_samples += 1

    if cloud_points:
        cloud = np.vstack(cloud_points)
        lo, hi = limits
        hist, xedges, yedges = np.histogram2d(cloud[:, 0], cloud[:, 1], bins=360, range=[[lo, hi], [lo, hi]])
        xx = 0.5 * (xedges[:-1] + xedges[1:])
        yy = 0.5 * (yedges[:-1] + yedges[1:])
        masked = np.ma.masked_where(hist.T <= 0, hist.T)
        ax.contourf(xx, yy, masked, levels=7, cmap="Greys", alpha=0.34, zorder=1)
        for pts in cloud_points:
            ax.plot(pts[:, 0], pts[:, 1], color="0.35", alpha=0.07, lw=0.8, zorder=3)

    bx, by = kepler_to_xy(best[:, 0], n_points=1000)
    ax.plot(bx, by, color="#d62728", lw=2.5, label="Bolide best-fit orbit", zorder=8)

    orb_now = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
    orb_now.kepler = best[:, 0:1]
    now = orb_now.cartesian[:2, 0] / AU
    ax.scatter(now[0], now[1], marker="*", s=145, color="#d62728", edgecolor="white", linewidth=0.8, zorder=9, label="Bolide at orbit epoch")
    ax.scatter(0.0, 0.0, s=140, color="gold", edgecolor="0.25", linewidth=0.8, zorder=10, label="Sun")
    return finite_samples


def style_axis(ax, limits: tuple[float, float], title: str) -> None:
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.axvline(0, color="0.9", lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Heliocentric ecliptic x (AU)")
    ax.set_ylabel("Heliocentric ecliptic y (AU)")
    ax.set_title(title)
    ax.grid(True, color="0.92", lw=0.8)


def plot(args: argparse.Namespace) -> None:
    best, epoch_utc, t_ref_unix = read_kepler(args.orbit_h5)
    sample_kepler, _, _ = read_kepler(args.samples_h5)
    epoch = Time(t_ref_unix, scale="utc", format="unix")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_inner, ax_full) = plt.subplots(1, 2, figsize=(15.5, 7.8))
    finite_samples = draw_orbits(ax_inner, best, sample_kepler, epoch, (-5.8, 5.8), {"Mercury", "Venus", "Earth", "Mars", "Jupiter"})
    draw_orbits(ax_full, best, sample_kepler, epoch, (-42.0, 42.0), {"Earth", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"})
    style_axis(ax_inner, (-5.8, 5.8), "Inner solar system")
    style_axis(ax_full, (-42.0, 42.0), "Full planetary context")
    fig.suptitle(f"New England bolide orbit in the J2000 heliocentric ecliptic x-y plane\nNASA mean planet elements advanced to {epoch_utc}", fontsize=14)
    ax_inner.fill_between([], [], [], color="0.55", alpha=0.34, label=f"MC uncertainty area (n={finite_samples})")
    ax_inner.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86, wspace=0.22)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bolide orbit and planet positions in heliocentric ecliptic x-y.")
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--samples-h5", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
