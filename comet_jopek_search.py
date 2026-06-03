#!/usr/bin/env python3
"""Search the local MPC comet-element catalogue for Jopek D_H orbit matches."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from nea_jopek_search import DEFAULT_ORBIT, Orbit, jopek_dh, read_bolide_orbit, southworth_hawkins_dsh


DEFAULT_CATALOG = Path("data/cometels.json.gz")
DEFAULT_OUT = Path("products/orbit_determination/comet_jopek_matches.csv")


def parse_comet_line(line: str) -> dict | None:
    parts = line.split()
    if len(parts) < 12:
        return None
    try:
        designation = parts[0]
        peri_year = int(parts[1])
        peri_month = int(parts[2])
        peri_day = float(parts[3])
        q_au = float(parts[4])
        e = float(parts[5])
        omega = float(parts[6])
        node = float(parts[7])
        inc = float(parts[8])
        epoch = parts[9]
        h_mag = float(parts[10])
        g_mag = float(parts[11])
    except ValueError:
        return None

    name = line[101:157].strip() if len(line) >= 157 else " ".join(parts[12:]).strip()
    reference = line[157:].strip() if len(line) >= 157 else ""
    if not name:
        name = designation
    a_au = q_au / (1.0 - e) if e < 1.0 else np.nan
    return {
        "designation": designation,
        "name": name,
        "reference": reference,
        "peri_year": peri_year,
        "peri_month": peri_month,
        "peri_day": peri_day,
        "q_au": q_au,
        "e": e,
        "a_au": a_au,
        "omega_deg": omega,
        "node_deg": node,
        "i_deg": inc,
        "epoch": epoch,
        "h": h_mag,
        "g": g_mag,
    }


def parse_json_record(record: dict) -> dict | None:
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


def read_catalog(catalog: Path) -> list[dict]:
    if catalog.suffix == ".gz" or catalog.name.endswith(".json.gz"):
        with gzip.open(catalog, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {catalog}")
        return [rec for row in payload if (rec := parse_json_record(row)) is not None]
    return [
        rec
        for line in catalog.read_text(encoding="utf-8", errors="replace").splitlines()
        if (rec := parse_comet_line(line)) is not None
    ]


def search(catalog: Path, bolide: Orbit) -> list[dict]:
    rows = []
    for rec in read_catalog(catalog):
        comet = Orbit(
            q_au=rec["q_au"],
            e=rec["e"],
            i_deg=rec["i_deg"],
            omega_deg=rec["omega_deg"],
            node_deg=rec["node_deg"],
            a_au=rec["a_au"] if np.isfinite(rec["a_au"]) else None,
        )
        rec["D_H"] = jopek_dh(bolide, comet)
        rec["D_SH"] = southworth_hawkins_dsh(bolide, comet)
        rows.append(rec)
    rows.sort(key=lambda row: row["D_H"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "D_H",
        "D_SH",
        "designation",
        "name",
        "reference",
        "peri_year",
        "peri_month",
        "peri_day",
        "a_au",
        "q_au",
        "e",
        "i_deg",
        "omega_deg",
        "node_deg",
        "epoch",
        "h",
        "g",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank local comet elements by Jopek D_H orbital similarity to the bolide orbit.")
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
    print(f"Searched {len(rows)} comet elements; wrote {args.out}")
    for rank, row in enumerate(rows[: args.top], start=1):
        a_text = f"{row['a_au']:.4f}" if np.isfinite(row["a_au"]) else "--"
        print(
            f"{rank:2d} D_H={row['D_H']:.5f} D_SH={row['D_SH']:.5f} "
            f"{row['designation']:>9s} {row['name']:<36s} "
            f"a={a_text} q={row['q_au']:.4f} e={row['e']:.4f} "
            f"i={row['i_deg']:.2f} omega={row['omega_deg']:.2f} Omega={row['node_deg']:.2f}"
        )


if __name__ == "__main__":
    main()
