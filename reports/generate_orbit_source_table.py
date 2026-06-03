#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import h5py
import numpy as np
import pyorb
from astropy.time import Time

import dasst


ROOT = Path(__file__).resolve().parents[1]
AU = 1.495978707e11
ORBIT_H5 = ROOT / "products" / "orbit_determination" / "best_fit_orbit.h5"
NEA_CSV = ROOT / "products" / "orbit_determination" / "nea_jopek_matches.csv"
COMET_CSV = ROOT / "products" / "orbit_determination" / "comet_jopek_matches.csv"
SOURCE_CSV = ROOT / "products" / "orbit_determination" / "meteor_source_matches.csv"
OUT = ROOT / "reports" / "orbit_source_table.tex"
PARAM_OUT = ROOT / "reports" / "orbit_parameter_table.tex"


def read_csv_rows(path: Path, n: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))[:n]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def circular_mean_deg(values_deg):
    values = np.deg2rad(np.asarray(values_deg, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))) % 360.0)


def angular_std_deg(values_deg: np.ndarray, mean_deg: float) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    diff = (values - mean_deg + 180.0) % 360.0 - 180.0
    return float(np.nanstd(diff, ddof=1))


def corrected_radiant(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    return circular_mean_deg(radiant[0, finite]), float(np.nanmean(radiant[1, finite]))


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
        return fmt(mean, min_decimals)
    digits = decimals_for_sigma(sigma, min_decimals=min_decimals)
    return f"{fmt(mean, digits)} $\\pm$ {fmt(sigma, digits)}"


def geocentric_speed_from_kepler(kep: np.ndarray, t_ref_unix: float) -> np.ndarray:
    epoch = Time(t_ref_unix, scale="utc", format="unix")
    out = np.full(kep.shape[1], np.nan, dtype=float)
    orb = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
    for idx in range(kep.shape[1]):
        row = kep[:, idx]
        if not np.all(np.isfinite(row)):
            continue
        try:
            orb.kepler = row[:, None]
            state = dasst.frames.convert(epoch, orb.cartesian, "HeliocentricMeanEcliptic", "GeocentricMeanEcliptic")
            out[idx] = np.linalg.norm(state[3:6, 0]) / 1000.0
        except Exception:
            out[idx] = np.nan
    return out


def write_orbit_parameter_table(path: Path) -> None:
    with h5py.File(path, "r") as h:
        kep = np.asarray(h["kepler_HeliocentricMeanEcliptic_final"][()], dtype=float)
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
        t_ref_unix = float(h.attrs["t_ref_unix"])
    finite = np.all(np.isfinite(kep), axis=0)
    kep = kep[:, finite]
    a_au = kep[0] / AU
    e = kep[1]
    q_au = a_au * (1.0 - e)
    vg_kmps = geocentric_speed_from_kepler(kep, t_ref_unix)

    quantities = [
        ("Semimajor axis", "$a$", "AU", a_au, False, 2),
        ("Perihelion distance", "$q$", "AU", q_au, False, 3),
        ("Eccentricity", "$e$", "", e, False, 3),
        ("Inclination", "$i$", "deg", kep[2], True, 1),
        ("Argument of perihelion", "$\\omega$", "deg", kep[3], True, 1),
        ("Longitude of ascending node", "$\\Omega$", "deg", kep[4], True, 2),
        ("True anomaly at epoch", "$\\nu$", "deg", kep[5], True, 1),
        ("Geocentric speed", "$v_g$", "km s$^{-1}$", vg_kmps, False, 2),
    ]

    rfinite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    ra = radiant[0, rfinite]
    dec = radiant[1, rfinite]
    quantities.extend(
        [
            ("Corrected radiant right ascension", "RA", "deg", ra, True, 2),
            ("Corrected radiant declination", "Dec", "deg", dec, False, 1),
        ]
    )

    lines = [
        "\\begin{tabular}{llrl}\n",
        "\\toprule\n",
        "Quantity & Symbol & Unit & Estimate \\\\\n",
        "\\midrule\n",
    ]
    for name, symbol, unit, values, is_angle, digits in quantities:
        values = np.asarray(values, dtype=float)
        if is_angle:
            mean = circular_mean_deg(values)
            sigma = angular_std_deg(values, mean)
        else:
            mean = float(np.nanmean(values))
            sigma = float(np.nanstd(values, ddof=1))
        lines.append(f"{name} & {symbol} & {unit} & {format_pm(mean, sigma, digits)} \\\\\n")
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    PARAM_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    with h5py.File(ORBIT_H5, "r") as h:
        if "kepler_HeliocentricMeanEcliptic_final_mean" in h:
            kep = h["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        else:
            kep = h["kepler_HeliocentricMeanEcliptic_final"][:, 0]
    ra_deg, dec_deg = corrected_radiant(ORBIT_H5)
    a_au = float(kep[0] / AU)
    e = float(kep[1])
    q_au = a_au * (1.0 - e)
    orbit_rows = [
        ("Bolide best fit", a_au, q_au, e, float(kep[2]), float(kep[3]), float(kep[4]), ra_deg, dec_deg, "", "", ""),
    ]

    nea_rows = []
    for row in read_csv_rows(NEA_CSV, 5):
        nea_rows.append(
            (
                row["name"],
                float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float("nan"),
                float("nan"),
                float(row["D_H"]),
                float("nan"),
                "NEA",
            )
        )

    comet_rows = []
    for row in read_csv_rows(COMET_CSV, 5):
        comet_rows.append(
            (
                row["name"] or row["designation"],
                float("nan") if row["a_au"] == "nan" else float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float("nan"),
                float("nan"),
                float(row["D_H"]),
                float("nan"),
                "comet",
            )
        )

    source_rows = []
    for row in read_csv_rows(SOURCE_CSV, 6):
        source_rows.append(
            (
                f"{row['code']} {row['name']}",
                float("nan") if row["a_au"] == "nan" else float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float(row["ra_deg"]),
                float(row["dec_deg"]),
                float(row["D_H"]),
                float(row["radiant_sep_deg"]),
                "stream",
            )
        )

    lines = [
        "\\begin{tabular}{lrrrrrrrrrrl}\n",
        "\\toprule\n",
        "Object/source & $a$ & $q$ & $e$ & $i$ & $\\omega$ & $\\Omega$ & RA & Dec & $D_H$ & $\\Delta_R$ & Type \\\\\n",
        " & (AU) & (AU) & & (deg) & (deg) & (deg) & (deg) & (deg) & & (deg) & \\\\\n",
        "\\midrule\n",
    ]
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in orbit_rows:
        lines.append(
            f"{name} & {fmt(a)} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & {fmt(ra, 2)} & {fmt(dec, 2)} & -- & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in nea_rows:
        lines.append(
            f"{name} & {fmt(a)} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & -- & -- & {fmt(dh, 3)} & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in comet_rows:
        a_text = "--" if a != a else fmt(a)
        lines.append(
            f"{name} & {a_text} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & -- & -- & {fmt(dh, 3)} & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in source_rows:
        a_text = "--" if a != a else fmt(a)
        lines.append(
            f"{name} & {a_text} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & {fmt(ra, 1)} & {fmt(dec, 1)} & "
            f"{fmt(dh, 3)} & {fmt(d_rad, 1)} & {typ} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    OUT.write_text("".join(lines), encoding="utf-8")
    write_orbit_parameter_table(ORBIT_H5)
    print(OUT)
    print(PARAM_OUT)


if __name__ == "__main__":
    main()
