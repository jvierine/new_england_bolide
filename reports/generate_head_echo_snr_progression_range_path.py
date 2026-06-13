#!/usr/bin/env python3
"""Plot head-echo SNR along the modeled range path for RDMF combining stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler_2dfft_notched import EVENT_T0  # noqa: E402
from reports.generate_station_head_rcs_figure import STATION_LABELS  # noqa: E402

DEFAULT_INPUT = ROOT / "products" / "sparse_residual_noise_power" / "station_range_time_head_snr_rdmf_baseline.h5"
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "head_echo_snr_rdmf_progression_range_path.png"
CURVES = (
    ("snr_db", "Incoherent TX + incoherent pol.", "#4b5563", "-"),
    ("txsteer_snr_db", "Trajectory TX steering + incoherent pol.", "#0f766e", "-"),
    ("txsteer_pol_snr_db", "Trajectory TX steering + matched pol.", "#b91c1c", "-"),
)


def selected_path_band_snr(group: h5py.Group, key: str, half_width_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = group["time_unix"][()].astype(float)
    snr = group[key][()].astype(float)
    ranges = group["range_km"][()].astype(float)
    model_range = group["model_range_km"][()].astype(float)
    center_cols = np.asarray([int(np.argmin(np.abs(ranges - r))) for r in model_range], dtype=int)
    cols = np.empty_like(center_cols)
    rows = np.arange(len(times))
    for ti, center in enumerate(center_cols):
        lo = max(0, int(center) - int(half_width_px))
        hi = min(snr.shape[1], int(center) + int(half_width_px) + 1)
        window = snr[ti, lo:hi]
        if np.all(~np.isfinite(window)):
            cols[ti] = int(center)
        else:
            cols[ti] = lo + int(np.nanargmax(window))
    return times - EVENT_T0, snr[rows, cols], ranges[cols]


def plot(input_h5: Path, output: Path, half_width_px: int, ymax_db: float | None) -> None:
    with h5py.File(input_h5, "r") as h:
        stations = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]
        data = {station: {key: selected_path_band_snr(h[station], key, half_width_px) for key, *_ in CURVES} for station in stations}
    vals = []
    for station in stations:
        for key, *_rest in CURVES:
            vals.append(data[station][key][1])
    finite = np.concatenate([v[np.isfinite(v)] for v in vals])
    ymin = max(-5.0, float(np.nanpercentile(finite, 2.0)) - 1.0)
    ymax = float(ymax_db) if ymax_db is not None else min(45.0, float(np.nanpercentile(finite, 99.0)) + 2.0)

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(len(stations), 1, figsize=(8.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, station in zip(axes, stations, strict=True):
        for key, label, color, linestyle in CURVES:
            t_rel, snr_db, _range_km = data[station][key]
            ax.plot(t_rel, snr_db, color=color, linestyle=linestyle, linewidth=1.4, alpha=0.92, label=label)
        ax.set_ylabel(f"{STATION_LABELS.get(station, station)}\nSNR (dB)")
        ax.grid(True, color="0.88", linewidth=0.5)
        ax.set_ylim(ymin, ymax)
    axes[0].legend(loc="upper left", ncol=1, fontsize=9, frameon=True)
    axes[0].text(
        0.99,
        0.92,
        rf"best pixel within $\pm {half_width_px}$ range bins of modeled range",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.25",
    )
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--half-width-px", type=int, default=10)
    p.add_argument("--ymax-db", type=float, default=None)
    args = p.parse_args()
    plot(args.input_h5, args.output, args.half_width_px, args.ymax_db)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
