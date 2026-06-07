#!/usr/bin/env python3
"""Plot the Cape Cod Bay bolide orbit in the heliocentric ecliptic x-y plane."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

AU = 1.495978707e11
DEFAULT_ORBIT = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_SAMPLES = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_OUT = Path("reports/figures/orbit_xy_bolide.png")
DEFAULT_MACROS = Path("reports/orbit_analysis_macros.tex")
DEFAULT_NEA_CATALOG = Path("data/NEA.txt")
DEFAULT_NEA_NAME = "2013 XG17"

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


def kepler_position_xy(kepler: np.ndarray, true_anomaly_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return heliocentric ecliptic x/y points in AU for true anomaly values."""
    kep = np.asarray(kepler, dtype=float)
    a_m, ecc, inc_deg, argp_deg, raan_deg, _nu_deg = kep
    nu = np.deg2rad(true_anomaly_deg)
    inc = math.radians(float(inc_deg))
    argp = math.radians(float(argp_deg))
    raan = math.radians(float(raan_deg))
    radius_m = a_m * (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    u = argp + nu
    x_m = radius_m * (np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * np.cos(inc))
    y_m = radius_m * (np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * np.cos(inc))
    return x_m / AU, y_m / AU


def kepler_to_xy(kepler: np.ndarray, n_points: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y orbit points in AU for one Kepler vector."""
    anomalies = np.linspace(0.0, 360.0, n_points, endpoint=True)
    return kepler_position_xy(kepler, anomalies)


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


def circular_mean_deg(values_deg: np.ndarray) -> float:
    values = np.deg2rad(np.asarray(values_deg, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))) % 360.0)


def angular_std_deg(values_deg: np.ndarray, mean_deg: float) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    diff = (values - mean_deg + 180.0) % 360.0 - 180.0
    return float(np.nanstd(diff, ddof=1))


def decimals_for_sigma(sigma: float, min_decimals: int = 0) -> int:
    if not np.isfinite(sigma) or sigma <= 0.0:
        return min_decimals
    exponent = int(math.floor(math.log10(abs(sigma))))
    first_digit = abs(sigma) / (10.0**exponent)
    sig_digits = 2 if first_digit < 3.0 else 1
    return max(min_decimals, sig_digits - 1 - exponent)


def format_pm(mean: float, sigma: float, min_decimals: int = 0) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sigma):
        return f"{mean:.{min_decimals}f}"
    digits = decimals_for_sigma(sigma, min_decimals=min_decimals)
    return f"{mean:.{digits}f} +/- {sigma:.{digits}f}"


def read_kepler(path: Path) -> tuple[np.ndarray, np.ndarray, str, float]:
    with h5py.File(path, "r") as h:
        kepler = h["kepler_HeliocentricMeanEcliptic_final"][()]
        if "kepler_HeliocentricMeanEcliptic_final_mean" in h:
            mean = h["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        else:
            mean = np.nanmean(kepler, axis=1)
        epoch_utc = h.attrs["t_ref_utc"]
        t_ref_unix = float(h.attrs["t_ref_unix"])
    return kepler, mean, epoch_utc, t_ref_unix


def read_orbit_metadata(path: Path) -> dict[str, str]:
    try:
        with h5py.File(path, "r") as h:
            attrs = {
                "trajectory_fit_version": str(h.attrs.get("trajectory_fit_version", "unspecified")),
                "n_samples": str(int(h.attrs.get("n_samples", 0))),
                "seed": str(int(h.attrs.get("seed", 0))),
            }
    except Exception:
        return {}
    return attrs


def add_version_stamp(ax, label: str | None) -> None:
    if not label:
        return
    ax.text(
        0.985,
        0.015,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.35",
        alpha=0.70,
        zorder=30,
    )


def metadata_stamp_label(metadata: dict[str, str]) -> str | None:
    version = metadata.get("trajectory_fit_version")
    if not version or version == "unspecified":
        return None
    n_samples = metadata.get("n_samples", "?")
    seed = metadata.get("seed", "?")
    return f"fit v{version}; n={n_samples}; seed={seed}"


def validate_macro_metadata(macros: dict[str, str], metadata: dict[str, str]) -> None:
    macro_version = macros.get("OrbitTrajectoryFitVersion")
    h5_version = metadata.get("trajectory_fit_version")
    if macro_version and h5_version and macro_version != h5_version:
        raise RuntimeError(f"orbit macro version {macro_version} does not match HDF5 version {h5_version}")
    macro_samples = macros.get("OrbitSampleCount")
    h5_samples = metadata.get("n_samples")
    if macro_samples and h5_samples and macro_samples != h5_samples:
        raise RuntimeError(f"orbit macro sample count {macro_samples} does not match HDF5 sample count {h5_samples}")


def kepler_annotation(sample_kepler: np.ndarray) -> str:
    finite = np.all(np.isfinite(sample_kepler), axis=0)
    kep = sample_kepler[:, finite]
    a_au = kep[0] / AU
    e = kep[1]
    q_au = a_au * (1.0 - e)
    jupiter_a_au = PLANETS["Jupiter"]["a"]
    tj = jupiter_a_au / a_au + 2.0 * np.cos(np.deg2rad(kep[2])) * np.sqrt((a_au / jupiter_a_au) * (1.0 - e * e))

    rows = [
        (r"$a$", "AU", a_au, False, 2),
        (r"$q$", "AU", q_au, False, 3),
        (r"$e$", "", e, False, 3),
        (r"$i$", "deg", kep[2], True, 1),
        (r"$T_J$", "", tj, False, 2),
        (r"$\omega$", "deg", kep[3], True, 1),
        (r"$\Omega$", "deg", kep[4], True, 2),
        (r"$\nu$", "deg", kep[5], True, 1),
    ]
    lines = ["Keplerian elements"]
    for symbol, unit, values, is_angle, min_decimals in rows:
        if is_angle:
            mean = circular_mean_deg(values)
            sigma = angular_std_deg(values, mean)
        else:
            mean = float(np.nanmean(values))
            sigma = float(np.nanstd(values, ddof=1))
        suffix = f" {unit}" if unit else ""
        lines.append(f"{symbol} = {format_pm(mean, sigma, min_decimals)}{suffix}")
    return "\n".join(lines)


def read_macro_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z0-9]+)\}\{(.+)\}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_mpcorb_line(line: str) -> dict | None:
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        return {
            "number": parts[0],
            "h": float(parts[1]),
            "epoch": parts[3],
            "mean_anomaly_deg": float(parts[4]),
            "omega_deg": float(parts[5]),
            "node_deg": float(parts[6]),
            "i_deg": float(parts[7]),
            "e": float(parts[8]),
            "n_deg_day": float(parts[9]),
            "a_au": float(parts[10]),
            "name": line[166:194].strip() if len(line) >= 194 else " ".join(parts[20:]),
        }
    except ValueError:
        return None


def find_nea_orbit(catalog: Path, name: str) -> tuple[str, np.ndarray] | None:
    if not catalog.exists():
        return None
    wanted = name.strip().lower()
    for line in catalog.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = parse_mpcorb_line(line)
        if rec is None:
            continue
        rec_name = str(rec["name"]).strip()
        if rec_name.lower() != wanted and wanted not in rec_name.lower():
            continue
        true_anomaly = true_anomaly_from_mean(float(rec["mean_anomaly_deg"]), float(rec["e"]))
        kep = np.array(
            [
                float(rec["a_au"]) * AU,
                float(rec["e"]),
                float(rec["i_deg"]),
                float(rec["omega_deg"]),
                float(rec["node_deg"]),
                true_anomaly,
            ],
            dtype=float,
        )
        return rec_name, kep
    return None


def kepler_annotation_from_macros(macros: dict[str, str]) -> str | None:
    required = [
        "OrbitSemiMajorMeanAU",
        "OrbitSemiMajorSigmaAU",
        "OrbitPerihelionMeanAU",
        "OrbitPerihelionSigmaAU",
        "OrbitAphelionMeanAU",
        "OrbitAphelionSigmaAU",
        "OrbitEccentricityMean",
        "OrbitEccentricitySigma",
        "OrbitInclinationMeanDeg",
        "OrbitInclinationSigmaDeg",
        "OrbitTisserandMean",
        "OrbitTisserandSigma",
        "OrbitArgPerihelionMeanDeg",
        "OrbitArgPerihelionSigmaDeg",
        "OrbitNodeMeanDeg",
        "OrbitNodeSigmaDeg",
        "OrbitTrueAnomalyMeanDeg",
        "OrbitTrueAnomalySigmaDeg",
    ]
    if any(name not in macros for name in required):
        return None
    rows = [
        (r"$a$", macros["OrbitSemiMajorMeanAU"], macros["OrbitSemiMajorSigmaAU"], "AU"),
        (r"$q$", macros["OrbitPerihelionMeanAU"], macros["OrbitPerihelionSigmaAU"], "AU"),
        (r"$Q$", macros["OrbitAphelionMeanAU"], macros["OrbitAphelionSigmaAU"], "AU"),
        (r"$e$", macros["OrbitEccentricityMean"], macros["OrbitEccentricitySigma"], ""),
        (r"$i$", macros["OrbitInclinationMeanDeg"], macros["OrbitInclinationSigmaDeg"], "deg"),
        (r"$T_J$", macros["OrbitTisserandMean"], macros["OrbitTisserandSigma"], ""),
        (r"$\omega$", macros["OrbitArgPerihelionMeanDeg"], macros["OrbitArgPerihelionSigmaDeg"], "deg"),
        (r"$\Omega$", macros["OrbitNodeMeanDeg"], macros["OrbitNodeSigmaDeg"], "deg"),
        (r"$\nu$", macros["OrbitTrueAnomalyMeanDeg"], macros["OrbitTrueAnomalySigmaDeg"], "deg"),
    ]
    lines = ["Keplerian elements"]
    for symbol, mean, sigma, unit in rows:
        suffix = f" {unit}" if unit else ""
        lines.append(f"{symbol} = {mean} +/- {sigma}{suffix}")
    return "\n".join(lines)


def draw_orbits(
    ax,
    best: np.ndarray | None,
    sample_kepler: np.ndarray,
    epoch: Time,
    limits: tuple[float, float],
    label_names: set[str],
    nea_orbit: tuple[str, np.ndarray] | None = None,
) -> int:
    for name in PLANETS:
        color = PLANET_COLORS[name]
        kep = planet_kepler(name, epoch)
        x, y = kepler_to_xy(kep, n_points=900)
        label = name if name in label_names else "_nolegend_"
        marker = "o" if name in label_names else None
        ax.plot(
            x,
            y,
            color=color,
            lw=0.9,
            alpha=0.72,
            zorder=2,
            label=label,
            marker=marker,
            markevery=[],
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
        )
        px, py = kepler_position_xy(kep, kep[5])
        size = 60 if name in {"Jupiter", "Saturn"} else 42
        if name in {"Uranus", "Neptune", "Pluto"}:
            size = 34
        ax.scatter(float(px), float(py), s=size, color=color, edgecolor="white", linewidth=0.5, zorder=6)

    if nea_orbit is not None:
        nea_name, nea_kep = nea_orbit
        nx, ny = kepler_to_xy(nea_kep, n_points=1000)
        ax.plot(nx, ny, color="#1f9e89", lw=1.8, alpha=0.86, zorder=7, label=nea_name)
        npx, npy = kepler_position_xy(nea_kep, nea_kep[5])
        ax.scatter(float(npx), float(npy), marker="D", s=70, color="#1f9e89", edgecolor="white", linewidth=0.7, zorder=9)

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

    if best is not None:
        bx, by = kepler_to_xy(best, n_points=1000)
        ax.plot(bx, by, color="#d62728", lw=2.0, alpha=0.7, label="Nominal propagated orbit", zorder=8)

        now = kepler_position_xy(best, best[5])
        ax.scatter(now[0], now[1], marker="*", s=145, color="#d62728", edgecolor="white", linewidth=0.8, zorder=9, label="Nominal epoch state")
    ax.scatter(0.0, 0.0, s=140, color="gold", edgecolor="0.25", linewidth=0.8, zorder=10, label="Sun")
    return finite_samples


def style_axis(ax, limits: tuple[float, float]) -> None:
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.axvline(0, color="0.9", lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Heliocentric ecliptic x (AU)")
    ax.set_ylabel("Heliocentric ecliptic y (AU)")
    ax.grid(True, color="0.92", lw=0.8)


def add_kepler_inset(ax, sample_kepler: np.ndarray, macro_path: Path) -> None:
    text = kepler_annotation_from_macros(read_macro_values(macro_path))
    if text is None:
        text = kepler_annotation(sample_kepler)
    ax.text(
        0.035,
        0.035,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        color="0.08",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 5.0},
        zorder=20,
    )


def plot(args: argparse.Namespace) -> None:
    _, best, epoch_utc, t_ref_unix = read_kepler(args.orbit_h5)
    sample_kepler, _, _, _ = read_kepler(args.samples_h5)
    macros = read_macro_values(args.macro_tex)
    metadata = read_orbit_metadata(args.orbit_h5)
    validate_macro_metadata(macros, metadata)
    epoch = Time(t_ref_unix, scale="utc", format="unix")
    nea_orbit = find_nea_orbit(args.nea_catalog, args.nea_name) if args.nea_name else None
    if args.nea_name and nea_orbit is None:
        raise FileNotFoundError(f"Could not find NEA {args.nea_name!r} in {args.nea_catalog}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    fig, ax_inner = plt.subplots(1, 1, figsize=(8.6, 8.2))
    limits = (-8.0, 8.0)
    finite_samples = draw_orbits(
        ax_inner,
        None,
        sample_kepler,
        epoch,
        limits,
        {"Mercury", "Venus", "Earth", "Mars", "Jupiter"},
        nea_orbit=nea_orbit,
    )
    style_axis(ax_inner, limits)
    add_kepler_inset(ax_inner, sample_kepler, args.macro_tex)
    add_version_stamp(ax_inner, metadata_stamp_label(metadata))
    ax_inner.fill_between([], [], [], color="0.55", alpha=0.34, label="Monte Carlo")
    ax_inner.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=True, framealpha=0.95)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.98)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bolide orbit and planet positions in heliocentric ecliptic x-y.")
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--samples-h5", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--macro-tex", type=Path, default=DEFAULT_MACROS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nea-catalog", type=Path, default=DEFAULT_NEA_CATALOG)
    parser.add_argument("--nea-name", default=DEFAULT_NEA_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
