#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AU = 1.495978707e11
ORBIT_H5 = ROOT / "products" / "orbit_determination" / "best_fit_orbit.h5"
NEA_CSV = ROOT / "products" / "orbit_determination" / "nea_jopek_matches.csv"
SOURCE_CSV = ROOT / "products" / "orbit_determination" / "meteor_source_matches.csv"
OUT = ROOT / "reports" / "orbit_source_table.tex"


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


def corrected_radiant(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    return circular_mean_deg(radiant[0, finite]), float(np.nanmean(radiant[1, finite]))


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
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in source_rows:
        a_text = "--" if a != a else fmt(a)
        lines.append(
            f"{name} & {a_text} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & {fmt(ra, 1)} & {fmt(dec, 1)} & "
            f"{fmt(dh, 3)} & {fmt(d_rad, 1)} & {typ} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    OUT.write_text("".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
