#!/usr/bin/env python3
"""Build a separate manual-selection workflow from exported selector JSON files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_retained_ext0p5.csv"
DEFAULT_OUT_DIR = ROOT / "reports" / "head_echo_manual_selection_ext0p5"

SELECTION_RANK = {
    "unknown": 0,
    "reject": 1,
    "real": 2,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_selection(row: dict) -> str:
    value = str(row.get("selection", "")).strip().lower()
    if value in SELECTION_RANK:
        return value
    if row.get("real") is True:
        return "real"
    return "unknown"


def load_selection_files(paths: list[Path]) -> tuple[dict[tuple[str, int], str], list[dict]]:
    merged: dict[tuple[str, int], str] = {}
    manifest = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = {"unknown": 0, "reject": 0, "real": 0}
        for row in data:
            station = str(row["station"])
            idx = int(row["idx"])
            sel = normalized_selection(row)
            counts[sel] += 1
            key = (station, idx)
            old = merged.get(key, "unknown")
            if SELECTION_RANK[sel] >= SELECTION_RANK[old]:
                merged[key] = sel
        manifest.append(
            {
                "path": str(path),
                "name": path.name,
                "sha256": sha256(path),
                "n_rows": len(data),
                "counts": counts,
            }
        )
    return merged, manifest


def build_tables(fit_summary: Path, selections: dict[tuple[str, int], str]) -> pd.DataFrame:
    df = pd.read_csv(fit_summary)
    df = df[df["success"].astype(str).str.lower() == "true"].copy()
    df["idx"] = pd.to_numeric(df["idx"], errors="raise").astype(int)
    df["selection"] = [
        selections.get((str(row.station), int(row.idx)), "unknown")
        for row in df[["station", "idx"]].itertuples(index=False)
    ]
    df["selected_real"] = df["selection"] == "real"
    df["selected_reject"] = df["selection"] == "reject"
    df["fit_range_km"] = pd.to_numeric(df["fit_range_m"], errors="coerce") / 1000.0
    return df


def write_empty_or_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_csv(path, index=False)
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "station",
                "idx",
                "time_unix",
                "time_rel_s",
                "fit_range_m",
                "fit_doppler_hz",
                "sigma_range_m",
                "sigma_doppler_hz",
                "selection",
                "selected_real",
            ]
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection_json", nargs="+", type=Path)
    p.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for i, src in enumerate(args.selection_json, start=1):
        dst = raw_dir / f"head_echo_measurement_selection_export_{i:02d}.json"
        shutil.copy2(src, dst)
        copied.append(dst)

    selections, manifest = load_selection_files(copied)
    df = build_tables(args.fit_summary, selections)

    all_csv = args.out_dir / "head_echo_manual_selection_all.csv"
    real_csv = args.out_dir / "head_echo_manual_selection_real.csv"
    reject_csv = args.out_dir / "head_echo_manual_selection_reject.csv"
    summary_json = args.out_dir / "head_echo_manual_selection_summary.json"

    df.to_csv(all_csv, index=False)
    write_empty_or_table(real_csv, df[df["selected_real"]].copy())
    write_empty_or_table(reject_csv, df[df["selected_reject"]].copy())

    counts = {
        "all": int(len(df)),
        "real": int(df["selected_real"].sum()),
        "reject": int(df["selected_reject"].sum()),
        "unknown": int((df["selection"] == "unknown").sum()),
    }
    by_station = {
        station: {
            "all": int(len(sdf)),
            "real": int(sdf["selected_real"].sum()),
            "reject": int(sdf["selected_reject"].sum()),
            "unknown": int((sdf["selection"] == "unknown").sum()),
        }
        for station, sdf in df.groupby("station", sort=True)
    }
    summary = {
        "schema": "new_england_bolide.head_echo_manual_selection.v1",
        "description": "Manual-selection workflow built from browser selector JSON exports. This workflow intentionally uses separate output paths and does not override retained/ext0p5 Gaussian products or trajectory products.",
        "fit_summary": str(args.fit_summary),
        "out_dir": str(args.out_dir),
        "raw_exports": manifest,
        "outputs": {
            "all_csv": str(all_csv),
            "real_csv": str(real_csv),
            "reject_csv": str(reject_csv),
        },
        "counts": counts,
        "by_station": by_station,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(summary_json)
    print(json.dumps({"counts": counts, "by_station": by_station}, indent=2))


if __name__ == "__main__":
    main()
