#!/usr/bin/env python3
"""Plot best-fit head-echo polarization state versus time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_polarized_residual.csv"
DEFAULT_OUT = ROOT / "reports" / "figures" / "head_echo_polarization_timeseries.png"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("success", "").lower() != "true":
                continue
            if not row.get("polarization_gamma0_abs") or not row.get("polarization_gamma1_abs"):
                continue
            rows.append(row)
    return rows


def as_float(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rows = load_rows(args.summary)
    if not rows:
        raise ValueError(f"No polarization rows found in {args.summary}")
    stations = sorted(set(row["station"] for row in rows))
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True)
    for station in stations:
        srows = [row for row in rows if row["station"] == station]
        t = as_float(srows, "time_rel_s")
        order = np.argsort(t)
        t = t[order]
        gamma0 = as_float(srows, "polarization_gamma0_abs")[order]
        gamma1 = as_float(srows, "polarization_gamma1_abs")[order]
        phase = as_float(srows, "polarization_relative_phase_deg")[order]
        metric = as_float(srows, "polarization_metric")[order]
        axes[0].plot(t, gamma0, label=f"{station} |g0|")
        axes[0].plot(t, gamma1, linestyle="--", label=f"{station} |g1|")
        axes[1].plot(t, phase, label=station)
        axes[2].plot(t, 10.0 * np.log10(np.maximum(metric, 1e-300)), label=station)
    axes[0].set_ylabel("Gamma amplitude")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Relative phase (deg)")
    axes[2].set_ylabel("Contour metric (dB)")
    axes[2].set_xlabel("Seconds after 18:06:20 UTC")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
