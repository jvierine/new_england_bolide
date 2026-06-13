#!/usr/bin/env python3
"""Range-gate SNR loss for the v0.2.9 range-interpolated RDMF."""

from __future__ import annotations

import argparse
import csv
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

from range_doppler_mf_bolide import bpsk, get_codes, interpolate_rect, read_config  # noqa: E402


DEFAULT_OUT = ROOT / "products" / "synthetic_range_gate_snr_loss_v0p2p9"


def fractional_delay_fft(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = len(x)
    freq = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2.0j * np.pi * freq * float(delay_samples))).astype(np.complex64)


def synthesize_decimated_echo(code: np.ndarray, oversample: int, delay_raw_samples: float, doppler_hz: float, sample_rate: float) -> np.ndarray:
    high_rate = sample_rate * oversample
    n_raw = 4 * len(code)
    n_high = n_raw * oversample
    code_high = np.repeat(code.astype(np.complex64), oversample)
    echo = fractional_delay_fft(code_high, delay_raw_samples * oversample)
    phase = np.exp(2.0j * np.pi * doppler_hz * (np.arange(len(echo)) / high_rate)).astype(np.complex64)
    z_high = np.zeros(n_high, dtype=np.complex64)
    start = len(code) * oversample
    z_high[start : start + len(echo)] += echo * phase
    return z_high.reshape(n_raw, oversample).mean(axis=1).astype(np.complex64)


def measure_peak_power(
    z100: np.ndarray,
    code_interp: np.ndarray,
    center_raw_range: int,
    range_interp: int,
    doppler_hz: float,
    sample_rate: float,
    doppler_zero_pad_factor: int,
) -> tuple[float, int, float]:
    zi = interpolate_rect(z100, range_interp)
    sample_rate_i = sample_rate * range_interp
    win_len = len(code_interp)
    fftlen = doppler_zero_pad_factor * win_len
    fvec = np.fft.fftshift(np.fft.fftfreq(fftlen, d=1.0 / sample_rate_i))
    di = int(np.argmin(np.abs(fvec - doppler_hz)))
    center_i = center_raw_range * range_interp
    ridx = np.arange(center_i - 8, center_i + 9, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(zi, win_len)[ridx, :]
    spec = np.fft.fftshift(np.fft.fft(windows * np.conj(code_interp), n=fftlen, axis=1), axes=1)
    power = np.abs(spec[:, di]) ** 2
    peak_i = int(np.argmax(power))
    return float(power[peak_i]), int(ridx[peak_i]), float(fvec[di])


def compute(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(args.data_dir, args.station)
    code = bpsk(int(config["seeds"][0]), int(config["code_len"]))
    code = code / np.max(np.abs(code))
    code_interp = get_codes([int(config["seeds"][0])], int(config["code_len"]), 1, range_interp=args.range_interp)[0]
    sample_rate = 100_000.0
    center_raw_range = 1200

    offsets = np.linspace(-0.5, 0.5, args.n_offsets, dtype=np.float64)
    rows = []
    for off_interp in offsets:
        delay_raw = center_raw_range + off_interp / float(args.range_interp)
        z100 = synthesize_decimated_echo(code, args.oversample, delay_raw, args.doppler_hz, sample_rate)
        power, peak_interp_idx, dop_bin = measure_peak_power(
            z100,
            code_interp,
            center_raw_range,
            args.range_interp,
            args.doppler_hz,
            sample_rate,
            args.doppler_zero_pad_factor,
        )
        rows.append(
            {
                "fractional_interp_gate": float(off_interp),
                "fractional_raw_gate": float(off_interp / args.range_interp),
                "peak_power": power,
                "peak_interp_idx": peak_interp_idx,
                "peak_offset_interp_gates": float(peak_interp_idx - center_raw_range * args.range_interp),
                "doppler_bin_hz": dop_bin,
            }
        )

    peak_ref = max(row["peak_power"] for row in rows)
    for row in rows:
        row["snr_loss_db"] = 10.0 * np.log10(max(row["peak_power"], 1e-30) / max(peak_ref, 1e-30))
        row["rcs_correction_db"] = -row["snr_loss_db"]

    csv_path = args.out_dir / "range_gate_snr_loss.csv"
    h5_path = args.out_dir / "range_gate_snr_loss.h5"
    fig_path = args.out_dir / "range_gate_snr_loss.png"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with h5py.File(h5_path.with_suffix(".h5.tmp"), "w") as h:
        h.attrs["source"] = "reports/simulate_range_gate_snr_loss_v0p2p9.py"
        h.attrs["description"] = "1 MHz echo, boxcar decimated to 100 kHz, processed with v0.2.9 range_interp=4 matched filter"
        for key in rows[0]:
            h.create_dataset(key, data=np.asarray([row[key] for row in rows]))
        h.create_dataset("station", data=np.bytes_(args.station))
        h.create_dataset("seed", data=int(config["seeds"][0]))
        h.create_dataset("code_len", data=int(config["code_len"]))
        h.create_dataset("range_interp", data=int(args.range_interp))
        h.create_dataset("oversample", data=int(args.oversample))
    h5_path.with_suffix(".h5.tmp").replace(h5_path)

    frac = np.asarray([r["fractional_interp_gate"] for r in rows])
    loss = np.asarray([r["snr_loss_db"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(frac, loss, "o-", ms=3.5, lw=1.4)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("Target offset from nearest interpolated range gate (gates)")
    ax.set_ylabel("Peak SNR loss (dB)")
    ax.set_title("v0.2.9 range interpolation loss, 1 MHz synthetic echo")
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    worst = min(rows, key=lambda r: r["snr_loss_db"])
    print(f"Wrote {csv_path}")
    print(f"Wrote {h5_path}")
    print(f"Wrote {fig_path}")
    print(
        f"Worst sampled loss {worst['snr_loss_db']:.3f} dB "
        f"at fractional_interp_gate={worst['fractional_interp_gate']:.3f}; "
        f"RCS correction {worst['rcs_correction_db']:.3f} dB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    p.add_argument("--station", default="pomfret")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--oversample", type=int, default=10)
    p.add_argument("--range-interp", type=int, default=4)
    p.add_argument("--doppler-zero-pad-factor", type=int, default=16)
    p.add_argument("--doppler-hz", type=float, default=-1500.0)
    p.add_argument("--n-offsets", type=int, default=101)
    args = p.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
