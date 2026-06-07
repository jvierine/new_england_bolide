#!/usr/bin/env python3
"""Generate orbit-source match CSVs using MC-mean Jopek D_H.

The source-association table should rank catalogue objects by the expectation
of the orbital dissimilarity over the bolide orbit sample distribution,
``mean(D_H(sample, catalogue_orbit))``.  Ranking by ``D_H`` evaluated only at a
mean Keplerian orbit can select a different neighbour when the orbit posterior
is broad or nonlinearly mapped into element space.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


AU = 1.495978707e11
DEFAULT_ORBIT = Path("products/orbit_determination/best_fit_orbit.h5")
DEFAULT_NEA_CATALOG = Path("data/NEA.txt")
DEFAULT_COMET_CATALOG = Path("data/cometels.json.gz")
DEFAULT_STREAM_CATALOG = Path("data/streamestablisheddata2026.txt")
DEFAULT_OUT_DIR = Path("products/orbit_determination")
DEFAULT_STREAM_REFERENCE_FILTERS = ("Jenniskens", "Brown")


@dataclass(frozen=True)
class Orbit:
    q_au: float
    e: float
    i_deg: float
    omega_deg: float
    node_deg: float
    a_au: float | None = None


@dataclass(frozen=True)
class OrbitSamples:
    q_au: np.ndarray
    e: np.ndarray
    i_deg: np.ndarray
    omega_deg: np.ndarray
    node_deg: np.ndarray
    a_au: np.ndarray

    def mean_orbit(self) -> Orbit:
        return Orbit(
            q_au=float(np.nanmean(self.q_au)),
            e=float(np.nanmean(self.e)),
            i_deg=circular_mean_deg(self.i_deg),
            omega_deg=circular_mean_deg(self.omega_deg),
            node_deg=circular_mean_deg(self.node_deg),
            a_au=float(np.nanmean(self.a_au)),
        )


def circular_mean_deg(values_deg: np.ndarray) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    values_rad = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(values_rad)), np.mean(np.cos(values_rad)))) % 360.0)


def read_bolide_samples(path: Path, max_samples: int | None = None) -> OrbitSamples:
    with h5py.File(path, "r") as h:
        kep = np.asarray(h["kepler_HeliocentricMeanEcliptic_final"][()], dtype=float)
    finite = np.all(np.isfinite(kep), axis=0)
    kep = kep[:, finite]
    if kep.size == 0:
        raise ValueError(f"No finite Keplerian orbit samples in {path}")
    if max_samples is not None and kep.shape[1] > max_samples:
        idx = np.linspace(0, kep.shape[1] - 1, max_samples, dtype=int)
        kep = kep[:, idx]
    a_au = kep[0] / AU
    e = kep[1]
    return OrbitSamples(
        q_au=a_au * (1.0 - e),
        e=e,
        i_deg=kep[2],
        omega_deg=kep[3],
        node_deg=kep[4],
        a_au=a_au,
    )


def read_corrected_radiant(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    if not np.any(finite):
        return math.nan, math.nan
    return circular_mean_deg(radiant[0, finite]), float(np.nanmean(radiant[1, finite]))


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_sep = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return float(math.degrees(math.acos(np.clip(cos_sep, -1.0, 1.0))))


def wrap_pi(angle_rad: np.ndarray | float) -> np.ndarray | float:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def plane_angle_samples(samples: OrbitSamples, orbit: Orbit) -> np.ndarray:
    i1 = np.deg2rad(samples.i_deg)
    i2 = np.deg2rad(orbit.i_deg)
    dnode = np.deg2rad(samples.node_deg - orbit.node_deg)
    cos_i = np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(dnode)
    return np.arccos(np.clip(cos_i, -1.0, 1.0))


def perihelion_lon_lat_samples(samples: OrbitSamples) -> tuple[np.ndarray, np.ndarray]:
    i = np.deg2rad(samples.i_deg)
    omega = np.deg2rad(samples.omega_deg)
    node = np.deg2rad(samples.node_deg)
    lon = node + np.arctan2(np.cos(i) * np.sin(omega), np.cos(omega))
    lat = np.arcsin(np.sin(i) * np.sin(omega))
    return lon, lat


def perihelion_lon_lat(orbit: Orbit) -> tuple[float, float]:
    i = math.radians(orbit.i_deg)
    omega = math.radians(orbit.omega_deg)
    node = math.radians(orbit.node_deg)
    lon = node + math.atan2(math.cos(i) * math.sin(omega), math.cos(omega))
    lat = math.asin(math.sin(i) * math.sin(omega))
    return lon, lat


def jopek_dh_samples(samples: OrbitSamples, orbit: Orbit) -> np.ndarray:
    i_angle = plane_angle_samples(samples, orbit)
    lon1, lat1 = perihelion_lon_lat_samples(samples)
    lon2, lat2 = perihelion_lon_lat(orbit)
    cos_theta = np.sin(lat1) * math.sin(lat2) + np.cos(lat1) * math.cos(lat2) * np.cos(lon2 - lon1)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    d2 = (samples.q_au - orbit.q_au) ** 2
    d2 += (samples.e - orbit.e) ** 2
    d2 += (2.0 * np.sin(0.5 * i_angle)) ** 2
    d2 += (((samples.e + orbit.e) / 2.0) * (theta / np.pi)) ** 2
    return np.sqrt(np.maximum(d2, 0.0))


def southworth_hawkins_dsh_samples(samples: OrbitSamples, orbit: Orbit) -> np.ndarray:
    i_angle = plane_angle_samples(samples, orbit)
    i1 = np.deg2rad(samples.i_deg)
    i2 = math.radians(orbit.i_deg)
    dnode = wrap_pi(math.radians(orbit.node_deg) - np.deg2rad(samples.node_deg))
    sign = np.where(np.abs(dnode) <= np.pi, 1.0, -1.0)
    arg = np.cos(0.5 * (i1 + i2)) * np.sin(0.5 * dnode) / np.maximum(np.cos(0.5 * i_angle), 1e-12)
    pi_angle = np.deg2rad(orbit.omega_deg - samples.omega_deg) + sign * 2.0 * np.arcsin(np.clip(arg, -1.0, 1.0))
    pi_angle = wrap_pi(pi_angle)
    d2 = (samples.q_au - orbit.q_au) ** 2
    d2 += (samples.e - orbit.e) ** 2
    d2 += (2.0 * np.sin(0.5 * i_angle)) ** 2
    d2 += (((samples.e + orbit.e) / 2.0) * 2.0 * np.sin(0.5 * pi_angle)) ** 2
    return np.sqrt(np.maximum(d2, 0.0))


def score_orbit(samples: OrbitSamples, mean_orbit: Orbit, orbit: Orbit) -> dict[str, float]:
    dh = jopek_dh_samples(samples, orbit)
    dsh = southworth_hawkins_dsh_samples(samples, orbit)
    dh_mean_orbit = float(jopek_dh_samples(
        OrbitSamples(
            q_au=np.array([mean_orbit.q_au]),
            e=np.array([mean_orbit.e]),
            i_deg=np.array([mean_orbit.i_deg]),
            omega_deg=np.array([mean_orbit.omega_deg]),
            node_deg=np.array([mean_orbit.node_deg]),
            a_au=np.array([mean_orbit.a_au if mean_orbit.a_au is not None else np.nan]),
        ),
        orbit,
    )[0])
    finite_dh = dh[np.isfinite(dh)]
    finite_dsh = dsh[np.isfinite(dsh)]
    return {
        "D_H": float(np.nanmean(finite_dh)),
        "D_H_mean": float(np.nanmean(finite_dh)),
        "D_H_sigma": float(np.nanstd(finite_dh, ddof=1)) if finite_dh.size > 1 else math.nan,
        "D_H_at_mean_orbit": dh_mean_orbit,
        "D_SH": float(np.nanmean(finite_dsh)),
        "D_SH_mean": float(np.nanmean(finite_dsh)),
        "D_SH_sigma": float(np.nanstd(finite_dsh, ddof=1)) if finite_dsh.size > 1 else math.nan,
    }


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


def parse_comet_record(record: dict) -> dict | None:
    try:
        q_au = float(record["Perihelion_dist"])
        e = float(record["e"])
        omega = float(record["Peri"])
        node = float(record["Node"])
        inc = float(record["i"])
    except (KeyError, TypeError, ValueError):
        return None
    a_au = q_au / (1.0 - e) if e < 1.0 else np.nan
    epoch_year = record.get("Epoch_year")
    epoch_month = record.get("Epoch_month")
    epoch_day = record.get("Epoch_day")
    epoch = (
        f"{int(epoch_year):04d}-{int(epoch_month):02d}-{float(epoch_day):05.2f}"
        if epoch_year is not None and epoch_month is not None and epoch_day is not None
        else ""
    )
    designation = str(record.get("Provisional_packed_desig") or record.get("Orbit_type") or "").strip()
    return {
        "designation": designation,
        "name": str(record.get("Designation_and_name") or designation).strip(),
        "reference": str(record.get("Ref") or "").strip(),
        "peri_year": record.get("Year_of_perihelion"),
        "peri_month": record.get("Month_of_perihelion"),
        "peri_day": record.get("Day_of_perihelion"),
        "q_au": q_au,
        "e": e,
        "a_au": a_au,
        "omega_deg": omega,
        "node_deg": node,
        "i_deg": inc,
        "epoch": epoch,
        "h": record.get("H"),
        "g": record.get("G"),
    }


def read_comet_catalog(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [rec for row in payload if (rec := parse_comet_record(row)) is not None]


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


def reference_allowed(reference: str, filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    reference_l = reference.lower()
    return any(token.lower() in reference_l for token in filters)


def parse_stream_catalog(path: Path, reference_filters: tuple[str, ...]) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def orbit_from_row(row: dict) -> Orbit:
    a_au = row.get("a_au")
    return Orbit(
        q_au=float(row["q_au"]),
        e=float(row["e"]),
        i_deg=float(row["i_deg"]),
        omega_deg=float(row["omega_deg"]),
        node_deg=float(row["node_deg"]),
        a_au=float(a_au) if a_au is not None and np.isfinite(float(a_au)) else None,
    )


def rank_rows(rows: list[dict], samples: OrbitSamples) -> list[dict]:
    mean_orbit = samples.mean_orbit()
    out = []
    for row in rows:
        scored = dict(row)
        scored.update(score_orbit(samples, mean_orbit, orbit_from_row(row)))
        out.append(scored)
    out.sort(key=lambda row: row["D_H"])
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def generate_nea(samples: OrbitSamples, catalog: Path, out: Path) -> list[dict]:
    rows = [
        rec
        for line in catalog.read_text(encoding="utf-8", errors="replace").splitlines()
        if (rec := parse_mpcorb_line(line)) is not None
    ]
    rows = rank_rows(rows, samples)
    write_csv(
        out,
        rows,
        [
            "rank", "D_H", "D_H_mean", "D_H_sigma", "D_H_at_mean_orbit", "D_SH", "D_SH_mean", "D_SH_sigma",
            "number", "name", "h", "epoch", "a_au", "q_au", "e", "i_deg", "omega_deg", "node_deg", "M_deg", "n_deg_day",
        ],
    )
    return rows


def generate_comets(samples: OrbitSamples, catalog: Path, out: Path) -> list[dict]:
    rows = rank_rows(read_comet_catalog(catalog), samples)
    write_csv(
        out,
        rows,
        [
            "rank", "D_H", "D_H_mean", "D_H_sigma", "D_H_at_mean_orbit", "D_SH", "D_SH_mean", "D_SH_sigma",
            "designation", "name", "reference", "peri_year", "peri_month", "peri_day", "a_au", "q_au", "e",
            "i_deg", "omega_deg", "node_deg", "epoch", "h", "g",
        ],
    )
    return rows


def generate_streams(
    samples: OrbitSamples,
    radiant_ra_dec_deg: tuple[float, float],
    catalog: Path,
    out: Path,
    reference_filters: tuple[str, ...],
) -> list[dict]:
    rows = rank_rows(parse_stream_catalog(catalog, reference_filters=reference_filters), samples)
    radiant_ra, radiant_dec = radiant_ra_dec_deg
    if np.isfinite(radiant_ra) and np.isfinite(radiant_dec):
        for row in rows:
            row["radiant_sep_deg"] = angular_separation_deg(radiant_ra, radiant_dec, row["ra_deg"], row["dec_deg"])
    write_csv(
        out,
        rows,
        [
            "rank", "D_H", "D_H_mean", "D_H_sigma", "D_H_at_mean_orbit", "D_SH", "D_SH_mean", "D_SH_sigma",
            "code", "name", "activity", "solar_lon_deg", "ra_deg", "dec_deg", "radiant_sep_deg", "vg_kmps",
            "a_au", "q_au", "e", "i_deg", "omega_deg", "node_deg", "n", "parent", "reference",
        ],
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit-h5", type=Path, default=DEFAULT_ORBIT)
    parser.add_argument("--nea-catalog", type=Path, default=DEFAULT_NEA_CATALOG)
    parser.add_argument("--comet-catalog", type=Path, default=DEFAULT_COMET_CATALOG)
    parser.add_argument("--stream-catalog", type=Path, default=DEFAULT_STREAM_CATALOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--skip-nea", action="store_true")
    parser.add_argument("--skip-comets", action="store_true")
    parser.add_argument("--skip-streams", action="store_true")
    parser.add_argument("--stream-reference-filter", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_bolide_samples(args.orbit_h5, max_samples=args.max_samples)
    radiant = read_corrected_radiant(args.orbit_h5)
    filters = tuple(args.stream_reference_filter) if args.stream_reference_filter is not None else DEFAULT_STREAM_REFERENCE_FILTERS
    jobs = []
    if not args.skip_nea:
        jobs.append(("NEA", generate_nea(samples, args.nea_catalog, args.out_dir / "nea_jopek_matches.csv")))
    if not args.skip_comets:
        jobs.append(("comet", generate_comets(samples, args.comet_catalog, args.out_dir / "comet_jopek_matches.csv")))
    if not args.skip_streams:
        jobs.append(("stream", generate_streams(samples, radiant, args.stream_catalog, args.out_dir / "meteor_source_matches.csv", filters)))
    print(f"Used {samples.q_au.size} finite bolide orbit samples")
    for label, rows in jobs:
        print(f"{label}: {len(rows)} rows")
        for rank, row in enumerate(rows[: args.top], start=1):
            name = row.get("name") or row.get("designation") or row.get("number") or row.get("code")
            print(
                f"  {rank:2d} D_H_mean={row['D_H']:.5f} "
                f"D_H(mean orbit)={row['D_H_at_mean_orbit']:.5f} {name}"
            )


if __name__ == "__main__":
    main()
