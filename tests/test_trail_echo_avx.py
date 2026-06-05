#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trail_echo_avx import range_doppler_peak_scores_avx


def numpy_reference(
    y_block: np.ndarray,
    shifted_codes: np.ndarray,
    window: np.ndarray,
    window_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_codes, n_ch, code_len = y_block.shape
    n_tx, n_range, _ = shifted_codes.shape
    power = np.zeros(n_range, dtype=np.float64)
    peak_bin = np.zeros(n_range, dtype=np.int32)
    for ri in range(n_range):
        p = np.zeros(n_codes, dtype=np.float64)
        shifted = shifted_codes[:, ri, :]
        for chi in range(n_ch):
            amp = y_block[:, chi, :] @ shifted.conj().T / float(code_len)
            spec = np.fft.fftshift(np.fft.fft(amp * window[:, None], axis=0), axes=0)
            p += np.sum(np.abs(spec) ** 2, axis=1) / float(window_norm * n_ch * n_tx)
        peak_bin[ri] = int(np.argmax(p))
        power[ri] = float(p[peak_bin[ri]])
    return power, peak_bin


def random_complex64(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def run_case(
    rng: np.random.Generator,
    n_codes: int,
    n_ch: int,
    code_len: int,
    n_tx: int,
    n_range: int,
) -> None:
    y = random_complex64(rng, (n_codes, n_ch, code_len))
    shifted = random_complex64(rng, (n_tx, n_range, code_len))
    window = np.hanning(n_codes).astype(np.float32)
    window_norm = float(np.sum(window**2))

    t0 = time.perf_counter()
    avx_power, avx_bin = range_doppler_peak_scores_avx(y, shifted, window, window_norm)
    avx_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_power, ref_bin = numpy_reference(y, shifted, window, window_norm)
    ref_s = time.perf_counter() - t0

    rel = np.abs(avx_power.astype(np.float64) - ref_power) / np.maximum(np.abs(ref_power), 1e-20)
    max_rel = float(np.max(rel))
    if not np.array_equal(avx_bin, ref_bin):
        bad = int(np.flatnonzero(avx_bin != ref_bin)[0])
        raise AssertionError(f"peak-bin mismatch at range {bad}: avx={avx_bin[bad]} ref={ref_bin[bad]}")
    if max_rel > 5e-5:
        raise AssertionError(f"power mismatch: max relative error {max_rel:.3e}")
    print(
        f"case n={n_codes} ch={n_ch} code={code_len} tx={n_tx} range={n_range}: "
        f"max_rel={max_rel:.3e} avx={avx_s:.3f}s numpy={ref_s:.3f}s speedup={ref_s / avx_s:.2f}x"
    )


def run_known_doppler_case() -> None:
    rng = np.random.default_rng(7)
    n_codes = 50
    n_ch = 2
    code_len = 1000
    n_tx = 5
    n_range = 1
    peak_bin = 37
    shifted = random_complex64(rng, (n_tx, n_range, code_len))
    y = np.zeros((n_codes, n_ch, code_len), dtype=np.complex64)
    phase = np.exp(2j * np.pi * ((peak_bin + n_codes // 2) % n_codes) * np.arange(n_codes) / n_codes).astype(np.complex64)
    for ni in range(n_codes):
        y[ni, 0, :] = phase[ni] * shifted[2, 0, :]
    window = np.hanning(n_codes).astype(np.float32)
    window_norm = float(np.sum(window**2))
    _, got_bin = range_doppler_peak_scores_avx(y, shifted, window, window_norm)
    if int(got_bin[0]) != peak_bin:
        raise AssertionError(f"known-Doppler mismatch: got bin {int(got_bin[0])}, expected {peak_bin}")
    print(f"known Doppler case: peak_bin={int(got_bin[0])}")


def main() -> None:
    rng = np.random.default_rng(20260605)
    run_known_doppler_case()
    cases = [
        (8, 1, 31, 2, 7),
        (16, 2, 64, 3, 11),
        (50, 2, 100, 5, 17),
        (50, 2, 1000, 5, 32),
    ]
    for case in cases:
        run_case(rng, *case)


if __name__ == "__main__":
    main()
