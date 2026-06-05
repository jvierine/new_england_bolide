#!/usr/bin/env python3
"""Search the local MPC NEA catalogue for Jopek D_H orbit matches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from orbit_product_provenance import assert_orbit_h5_matches_trajectory_json


AU = 1.495978707e11
DEFAULT_ORBIT = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_CATALOG = Path("data/NEA.txt")
DEFAULT_OUT = Path("products/orbit_determination/nea_jopek_matches.csv")


@dataclass(frozen=True)
class Orbit:
    q_au: float
    e: float
    i_deg: float
    omega_deg: float
    node_deg: float
    a_au: float | None = None


def wrap_pi(angle_rad: np.ndarray | float) -> np.ndarray | float:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def plane_angle(o1: Orbit, o2: Orbit) -> float:
    i1 = np.deg2rad(o1.i_deg)
    i2 = np.deg2rad(o2.i_deg)
    dnode = np.deg2rad(o1.node_deg - o2.node_deg)
    cos_i = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(dnode)
    return float(np.arccos(np.clip(cos_i, -1.0, 1.0)))


def perihelion_lon_lat(o: Orbit) -> tuple[float, float]:
    i = np.deg2rad(o.i_deg)
    omega = np.deg2rad(o.omega_deg)
    node = np.deg2rad(o.node_deg)
    lon = node + np.arctan2(np.cos(i) * np.sin(omega), np.cos(omega))
    lat = np.arcsin(np.sin(i) * np.sin(omega))
    return float(lon), float(lat)


def perihelion_direction_angle(o1: Orbit, o2: Orbit) -> float:
    lon1, lat1 = perihelion_lon_lat(o1)
    lon2, lat2 = perihelion_lon_lat(o2)
    cos_theta = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def jopek_dh(o1: Orbit, o2: Orbit) -> float:
    """Jopek D_H: D_SH q/e/plane terms with Drummond perihelion-direction term."""
    i_angle = plane_angle(o1, o2)
    theta = perihelion_direction_angle(o1, o2)
    d2 = 0.0
    d2 += (o1.q_au - o2.q_au) ** 2
    d2 += (o1.e - o2.e) ** 2
    d2 += (2.0 * np.sin(0.5 * i_angle)) ** 2
    d2 += (((o1.e + o2.e) / 2.0) * (theta / np.pi)) ** 2
    return float(np.sqrt(d2))


def southworth_hawkins_dsh(o1: Orbit, o2: Orbit) -> float:
    i_angle = plane_angle(o1, o2)
    i1 = np.deg2rad(o1.i_deg)
    i2 = np.deg2rad(o2.i_deg)
    dnode = wrap_pi(np.deg2rad(o2.node_deg - o1.node_deg))
    sign = 1.0 if abs(dnode) <= np.pi else -1.0
    arg = np.cos(0.5 * (i1 + i2)) * np.sin(0.5 * dnode) / max(np.cos(0.5 * i_angle), 1e-12)
    pi_angle = np.deg2rad(o2.omega_deg - o1.omega_deg) + sign * 2.0 * np.arcsin(np.clip(arg, -1.0, 1.0))
    pi_angle = wrap_pi(pi_angle)
    d2 = 0.0
    d2 += (o1.q_au - o2.q_au) ** 2
    d2 += (o1.e - o2.e) ** 2
    d2 += (2.0 * np.sin(0.5 * i_angle)) ** 2
    d2 += (((o1.e + o2.e) / 2.0) * 2.0 * np.sin(0.5 * pi_angle)) ** 2
    return float(np.sqrt(d2))


def read_bolide_orbit(path: Path) -> Orbit:
    assert_orbit_h5_matches_trajectory_json(path)
    with h5py.File(path, "r") as h:
        if "kepler_HeliocentricMeanEcliptic_final_mean" in h:
            kep = h["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        else:
            kep = h["kepler_HeliocentricMeanEcliptic_final"][:, 0]
    a_au = float(kep[0] / AU)
    e = float(kep[1])
    return Orbit(
        q_au=a_au * (1.0 - e),
        e=e,
        i_deg=float(kep[2]),
        omega_deg=float(kep[3]),
        node_deg=float(kep[4]),
        a_au=a_au,
    )


def parse_mpcorb_line(line: str) -> dict | None:
    parts = line.split()
    if len(parts) < 11:
        return None
    try:
        number = parts[0]
        h_mag = float(parts[1])
        epoch = parts[3]
        mean_anomaly = float(parts[4])
        omega = float(parts[5])
        node = float(parts[6])
        inc = float(parts[7])
        ecc = float(parts[8])
        mean_motion = float(parts[9])
        semi_major = float(parts[10])
    except ValueError:
        return None

    name = line[166:194].strip() if len(line) >= 194 else ""
    if not name:
        name = " ".join(parts[20:]) if len(parts) > 20 else number
    return {
        "number": number,
        "name": name,
        "h": h_mag,
        "epoch": epoch,
        "M_deg": mean_anomaly,
        "omega_deg": omega,
        "node_deg": node,
        "i_deg": inc,
        "e": ecc,
        "n_deg_day": mean_motion,
        "a_au": semi_major,
        "q_au": semi_major * (1.0 - ecc),
    }


def search(catalog: Path, bolide: Orbit) -> list[dict]:
    rows = []
    for line in catalog.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = parse_mpcorb_line(line)
        if rec is None:
            continue
        nea = Orbit(
            q_au=rec["q_au"],
            e=rec["e"],
            i_deg=rec["i_deg"],
            omega_deg=rec["omega_deg"],
            node_deg=rec["node_deg"],
            a_au=rec["a_au"],
        )
        rec["D_H"] = jopek_dh(bolide, nea)
        rec["D_SH"] = southworth_hawkins_dsh(bolide, nea)
        rows.append(rec)
    rows.sort(key=lambda r: r["D_H"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "D_H",
        "D_SH",
        "number",
        "name",
        "h",
        "epoch",
        "a_au",
        "q_au",
        "e",
        "i_deg",
        "omega_deg",
        "node_deg",
        "M_deg",
        "n_deg_day",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank local NEAs by Jopek D_H orbital similarity to the bolide orbit.")
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bolide = read_bolide_orbit(args.orbit_h5)
    rows = search(args.catalog, bolide)
    write_csv(args.out, rows)
    print(
        "Bolide: "
        f"a={bolide.a_au:.6f} AU q={bolide.q_au:.6f} AU e={bolide.e:.6f} "
        f"i={bolide.i_deg:.4f} omega={bolide.omega_deg:.4f} Omega={bolide.node_deg:.4f}"
    )
    print(f"Searched {len(rows)} NEAs; wrote {args.out}")
    for rank, row in enumerate(rows[: args.top], start=1):
        print(
            f"{rank:2d} D_H={row['D_H']:.5f} D_SH={row['D_SH']:.5f} "
            f"{row['number']:>7s} {row['name']:<28s} "
            f"a={row['a_au']:.4f} q={row['q_au']:.4f} e={row['e']:.4f} "
            f"i={row['i_deg']:.2f} omega={row['omega_deg']:.2f} Omega={row['node_deg']:.2f}"
        )


if __name__ == "__main__":
    main()
