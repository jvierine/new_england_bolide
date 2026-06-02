#!/usr/bin/env python3
"""Rank established meteor streams against the bolide orbit and radiant.

The default catalogue is the 2026 IAU Meteor Data Center established shower
list in ``data/streamestablisheddata2026.txt``.  Shower rows use the catalogue
orbital elements and geocentric J2000 radiant; no shower mean anomaly is used.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np

from nea_jopek_search import Orbit, jopek_dh, read_bolide_orbit, southworth_hawkins_dsh


DEFAULT_ORBIT = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_CATALOG = Path("data/streamestablisheddata2026.txt")
DEFAULT_OUT = Path("products/orbit_determination/meteor_source_matches.csv")
DEFAULT_REFERENCE_FILTERS = ("Jenniskens", "Brown")


def clean(value: str) -> str:
    return value.strip().strip('"')


def parse_float(value: str) -> float:
    value = clean(value)
    return float(value) if value else math.nan


def parse_int(value: str) -> int:
    value = clean(value).replace(" ", "")
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def circular_mean_deg(values_deg: np.ndarray) -> float:
    values = np.deg2rad(values_deg[np.isfinite(values_deg)])
    if values.size == 0:
        return math.nan
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))) % 360.0)


def read_corrected_radiant(path: Path) -> tuple[float, float]:
    """Return mean zenithal-attraction-corrected radiant RA/Dec from the orbit HDF5."""
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    if radiant.ndim != 2 or radiant.shape[0] != 2:
        raise ValueError(f"Unexpected radiant_orbit_GCRS shape in {path}: {radiant.shape}")
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    if not np.any(finite):
        raise ValueError(f"No finite radiant_orbit_GCRS samples in {path}")
    return circular_mean_deg(radiant[0, finite]), float(np.nanmean(radiant[1, finite]))


def reference_allowed(reference: str, filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    reference_l = reference.lower()
    return any(token.lower() in reference_l for token in filters)


def parse_stream_catalog(path: Path, reference_filters: tuple[str, ...] = DEFAULT_REFERENCE_FILTERS) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if not line.startswith('"'):
            continue
        parts = line.split("|")
        if len(parts) < 36:
            continue
        try:
            reference = clean(parts[35])
            if not reference_allowed(reference, reference_filters):
                continue
            ra = parse_float(parts[11])
            dec = parse_float(parts[12])
            q = parse_float(parts[23])
            e = parse_float(parts[24])
            omega = parse_float(parts[25])
            node = parse_float(parts[26])
            inc = parse_float(parts[27])
            if not all(np.isfinite([ra, dec, q, e, omega, node, inc])) or e > 1.0:
                continue
            rows.append(
                {
                    "code": clean(parts[3]),
                    "name": clean(parts[6]),
                    "activity": clean(parts[7]),
                    "solar_lon_deg": parse_float(parts[10]),
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "vg_kmps": parse_float(parts[15]),
                    "a_au": parse_float(parts[22]),
                    "q_au": q,
                    "e": e,
                    "omega_deg": omega,
                    "node_deg": node,
                    "i_deg": inc,
                    "n": parse_int(parts[28]),
                    "parent": clean(parts[34]),
                    "reference": reference,
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def rank_streams(
    catalog: Path,
    bolide: Orbit,
    radiant_ra_deg: float | None = None,
    radiant_dec_deg: float | None = None,
    reference_filters: tuple[str, ...] = DEFAULT_REFERENCE_FILTERS,
) -> list[dict]:
    rows = []
    for row in parse_stream_catalog(catalog, reference_filters=reference_filters):
        stream = Orbit(
            q_au=row["q_au"],
            e=row["e"],
            i_deg=row["i_deg"],
            omega_deg=row["omega_deg"],
            node_deg=row["node_deg"],
            a_au=row["a_au"] if np.isfinite(row["a_au"]) else None,
        )
        row["D_H"] = jopek_dh(bolide, stream)
        row["D_SH"] = southworth_hawkins_dsh(bolide, stream)
        if radiant_ra_deg is not None and radiant_dec_deg is not None:
            row["radiant_sep_deg"] = angular_separation_deg(
                radiant_ra_deg, radiant_dec_deg, row["ra_deg"], row["dec_deg"]
            )
        else:
            row["radiant_sep_deg"] = math.nan
        rows.append(row)
    rows.sort(key=lambda row: row["D_H"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "D_H",
        "D_SH",
        "code",
        "name",
        "activity",
        "solar_lon_deg",
        "ra_deg",
        "dec_deg",
        "radiant_sep_deg",
        "vg_kmps",
        "a_au",
        "q_au",
        "e",
        "i_deg",
        "omega_deg",
        "node_deg",
        "n",
        "parent",
        "reference",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank established meteor streams by D_H and radiant separation.")
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--reference-filter",
        action="append",
        default=None,
        help="Keep only catalogue rows whose References field contains this token. "
        "May be repeated. Defaults to Jenniskens and Brown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bolide = read_bolide_orbit(args.orbit_h5)
    reference_filters = tuple(args.reference_filter) if args.reference_filter is not None else DEFAULT_REFERENCE_FILTERS
    radiant_ra, radiant_dec = read_corrected_radiant(args.orbit_h5)
    rows = rank_streams(args.catalog, bolide, radiant_ra, radiant_dec, reference_filters=reference_filters)
    write_csv(args.out, rows)
    print(f"Bolide corrected radiant: RA={radiant_ra:.3f} deg Dec={radiant_dec:.3f} deg")
    print(f"Searched {len(rows)} stream entries; wrote {args.out}")
    for rank, row in enumerate(rows[: args.top], start=1):
        print(
            f"{rank:2d} D_H={row['D_H']:.4f} D_SH={row['D_SH']:.4f} "
            f"dRad={row['radiant_sep_deg']:.2f} deg "
            f"{row['code']:<4s} {row['name']:<36s} "
            f"RA={row['ra_deg']:.1f} Dec={row['dec_deg']:.1f} "
            f"q={row['q_au']:.3f} e={row['e']:.3f} i={row['i_deg']:.1f} "
            f"omega={row['omega_deg']:.1f} Omega={row['node_deg']:.1f} parent={row['parent'][:28]}"
        )


if __name__ == "__main__":
    main()
