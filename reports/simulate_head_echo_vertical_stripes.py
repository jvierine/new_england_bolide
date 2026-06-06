#!/usr/bin/env python3
"""Simulate vertical stripe diagnostics for the 16-code Clinton head echo patch."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from range_doppler_mf_bolide import get_codes  # noqa: E402
from reports.generate_head_microdoppler_2dfft_notched import notch_2d_doppler_repetition  # noqa: E402

EXAMPLE_CACHE = ROOT / "reports" / "head_echo_gaussian_2dfft_example.h5"
RDMF_CLINTON = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431" / "rdmf_peak_clinton_merged.h5"
OUT_FIG = ROOT / "reports" / "figures" / "head_echo_vertical_stripe_simulation.png"
OUT_TEX = ROOT / "reports" / "head_echo_vertical_stripe_simulation_appendix.tex"
OUT_H5 = ROOT / "reports" / "head_echo_vertical_stripe_simulation.h5"


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    signal_codes: tuple[int, ...]
    filter_codes: tuple[int, ...]
    n_channels: int
    direct_path_db: float | None = None
    random_code_phases: bool = False


VARIANTS = (
    Variant(
        key="one_tx_one_code",
        title="1 TX code, 1 RX channel",
        signal_codes=(0,),
        filter_codes=(0,),
        n_channels=1,
    ),
    Variant(
        key="one_tx_one_code_two_rx",
        title="1 TX code, 2 RX channels",
        signal_codes=(0,),
        filter_codes=(0,),
        n_channels=2,
    ),
    Variant(
        key="five_tx_equal",
        title="5 TX codes, equal phases",
        signal_codes=(0, 1, 2, 3, 4),
        filter_codes=(0, 1, 2, 3, 4),
        n_channels=2,
    ),
    Variant(
        key="five_tx_random_phase",
        title="5 TX codes, random phases",
        signal_codes=(0, 1, 2, 3, 4),
        filter_codes=(0, 1, 2, 3, 4),
        n_channels=2,
        random_code_phases=True,
    ),
    Variant(
        key="five_tx_direct_path_stress",
        title="5 TX + strong zero-Doppler leakage",
        signal_codes=(0, 1, 2, 3, 4),
        filter_codes=(0, 1, 2, 3, 4),
        n_channels=2,
        random_code_phases=True,
        direct_path_db=35.0,
    ),
)


def load_geometry() -> dict:
    with h5py.File(EXAMPLE_CACHE, "r") as h:
        raw = h["raw_db"][()]
        range_m = h["range_m"][()]
        rel_freqs = h["rel_freqs"][()]
        seed_range_m = float(h.attrs["seed_range_m"])
        seed_doppler_hz = float(h.attrs["seed_doppler_hz"])
        model_doppler_hz = float(h.attrs["model_doppler_hz"])
        time_rel = float(h.attrs["time_rel"])

    with h5py.File(RDMF_CLINTON, "r") as h:
        seeds = h["seeds"][()]
        code_len = int(h["code_len"][()])
        range_interp = int(h["range_interp"][()])
        sample_rate = float(h["sample_rate"][()])
        range_resolution_m = float(h["range_resolution_m"][()])

    n_coh = 16
    sample_rate_i = sample_rate * range_interp
    code_len_i = code_len * range_interp
    win_len = n_coh * code_len_i
    fvec_full = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    bulk = model_doppler_hz
    fidx = np.array([int(np.argmin(np.abs(fvec_full - (bulk + rel)))) for rel in rel_freqs], dtype=np.int64)
    gate_m = range_resolution_m / range_interp
    ridx = np.rint(range_m / gate_m).astype(np.int64)
    seed_ridx = int(round(seed_range_m / gate_m))

    peak_mask = (
        (np.abs(range_m[:, None] - seed_range_m) <= 8000.0)
        & (np.abs(rel_freqs[None, :] - (seed_doppler_hz - model_doppler_hz)) <= 300.0)
    )
    observed_contrast_db = float(np.nanmax(raw[peak_mask]) - np.nanmedian(raw))

    return {
        "raw": raw,
        "range_m": range_m,
        "rel_freqs": rel_freqs,
        "seed_range_m": seed_range_m,
        "seed_doppler_hz": seed_doppler_hz,
        "model_doppler_hz": model_doppler_hz,
        "time_rel": time_rel,
        "seeds": seeds,
        "code_len": code_len,
        "range_interp": range_interp,
        "sample_rate_i": sample_rate_i,
        "gate_m": gate_m,
        "ridx": ridx,
        "seed_ridx": seed_ridx,
        "win_len": win_len,
        "fvec_full": fvec_full,
        "fidx": fidx,
        "observed_contrast_db": observed_contrast_db,
    }


def target_waveform(codes: list[np.ndarray], code_indices: tuple[int, ...], n: int, doppler_hz: float, sample_rate: float, rng: np.random.Generator, random_phases: bool) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase = np.exp(2.0j * np.pi * doppler_hz * t)
    out = np.zeros(n, dtype=np.complex64)
    for ci in code_indices:
        weight = 1.0 + 0.0j
        if random_phases:
            weight = np.exp(2.0j * np.pi * rng.random())
        out += (weight * codes[ci][:n] * phase).astype(np.complex64)
    if code_indices:
        out /= np.sqrt(float(len(code_indices)))
    return out


def matched_filter_image(z_channels: list[np.ndarray], codes: list[np.ndarray], filter_indices: tuple[int, ...], geom: dict) -> np.ndarray:
    ridx = geom["ridx"]
    win_len = geom["win_len"]
    fidx = geom["fidx"]
    S = np.zeros((len(ridx), len(fidx)), dtype=np.float64)
    for z in z_channels:
        windows = np.lib.stride_tricks.sliding_window_view(z, win_len)[ridx, :]
        for ci in filter_indices:
            Z = np.fft.fftshift(np.fft.fft(windows * np.conj(codes[ci]), axis=1), axes=1)[:, fidx]
            S += np.abs(Z) ** 2
    return S


def scale_target_to_observed_snr(noise_channels: list[np.ndarray], target_channels: list[np.ndarray], codes: list[np.ndarray], variant: Variant, geom: dict) -> float:
    noise = matched_filter_image(noise_channels, codes, variant.filter_codes, geom)
    target_only = matched_filter_image(target_channels, codes, variant.filter_codes, geom)
    rel = geom["rel_freqs"]
    target_rel = geom["seed_doppler_hz"] - geom["model_doppler_hz"]
    ri = int(np.argmin(np.abs(geom["ridx"] - geom["seed_ridx"])))
    fi = int(np.argmin(np.abs(rel - target_rel)))
    noise_floor = float(np.nanmedian(noise))
    signal_peak = max(float(target_only[ri, fi]), 1e-30)
    ratio = 10.0 ** (geom["observed_contrast_db"] / 10.0)
    return float(np.sqrt(max((ratio - 1.0) * noise_floor, 0.0) / signal_peak))


def comb_metric(image_db: np.ndarray, period_bins: int = 16) -> tuple[float, float]:
    img0 = image_db - np.nanmean(image_db)
    F = np.abs(np.fft.fft2(img0)) ** 2
    ny, nx = image_db.shape
    fund = nx // period_bins
    cols = np.zeros(nx, dtype=bool)
    for hidx in range(1, nx // (2 * fund) + 1):
        k = hidx * fund
        if k > nx // 2:
            break
        cols[k] = True
        cols[(-k) % nx] = True
    line_power = float(np.nanmean(F[:, cols])) if np.any(cols) else 0.0
    off_power = float(np.nanmean(F[:, ~cols])) if np.any(~cols) else 1.0
    return line_power / max(off_power, 1e-30), 10.0 * np.log10(max(line_power / max(off_power, 1e-30), 1e-30))


def simulate_variant(variant: Variant, geom: dict, codes: list[np.ndarray], rng: np.random.Generator) -> dict:
    n = int(geom["win_len"] + int(geom["ridx"][-1]) + 1)
    noise_channels = [
        (rng.normal(size=n) + 1.0j * rng.normal(size=n)).astype(np.complex64) / np.sqrt(2.0)
        for _ in range(variant.n_channels)
    ]
    target = target_waveform(
        codes,
        variant.signal_codes,
        geom["win_len"],
        geom["seed_doppler_hz"],
        geom["sample_rate_i"],
        rng,
        variant.random_code_phases,
    )
    target_channels = [np.zeros(n, dtype=np.complex64) for _ in range(variant.n_channels)]
    for chi in range(variant.n_channels):
        phase = np.exp(2.0j * np.pi * rng.random())
        target_channels[chi][geom["seed_ridx"] : geom["seed_ridx"] + geom["win_len"]] += phase * target

    amp = scale_target_to_observed_snr(noise_channels, target_channels, codes, variant, geom)
    z_channels = [noise_channels[chi] + amp * target_channels[chi] for chi in range(variant.n_channels)]

    direct_amp = 0.0
    if variant.direct_path_db is not None:
        direct = target_waveform(
            codes,
            variant.signal_codes,
            geom["win_len"],
            0.0,
            geom["sample_rate_i"],
            rng,
            variant.random_code_phases,
        )
        direct_amp = amp * 10.0 ** (variant.direct_path_db / 20.0)
        for chi in range(variant.n_channels):
            phase = np.exp(2.0j * np.pi * rng.random())
            # This is intentionally crude: a strong stationary code-like leakage term
            # is injected at all ranges so we can test whether such leakage creates
            # columnar Doppler structure in the matched-filter image.
            z_channels[chi][: geom["win_len"]] += phase * direct_amp * direct

    S = matched_filter_image(z_channels, codes, variant.filter_codes, geom)
    raw_db = 10.0 * np.log10(np.maximum(S, 1e-30))
    filtered_db = notch_2d_doppler_repetition(raw_db, 16, 1)
    line_ratio, line_ratio_db = comb_metric(raw_db, 16)
    filtered_ratio, filtered_ratio_db = comb_metric(filtered_db, 16)
    return {
        "raw_db": raw_db,
        "filtered_db": filtered_db,
        "target_amplitude": amp,
        "direct_amplitude": direct_amp,
        "comb_ratio": line_ratio,
        "comb_ratio_db": line_ratio_db,
        "filtered_comb_ratio": filtered_ratio,
        "filtered_comb_ratio_db": filtered_ratio_db,
    }


def plot_results(geom: dict, results: dict[str, dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    range_km = geom["range_m"] / 1000.0
    rel = geom["rel_freqs"]
    observed = geom["raw"] - np.nanmedian(geom["raw"])
    all_images = [observed] + [results[v.key]["raw_db"] - np.nanmedian(results[v.key]["raw_db"]) for v in VARIANTS]
    finite = np.concatenate([img[np.isfinite(img)].ravel() for img in all_images])
    vmin = float(np.nanpercentile(finite, 3.0))
    vmax = float(np.nanpercentile(finite, 99.6))

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), sharex=True, sharey=True, constrained_layout=True)
    observed_metric = comb_metric(geom["raw"], 16)
    panels = [("Observed Clinton raw patch", observed, {"comb_ratio_db": observed_metric[1]})] + [
        (v.title, results[v.key]["raw_db"] - np.nanmedian(results[v.key]["raw_db"]), results[v.key]) for v in VARIANTS
    ]
    for ax, (title, img, result) in zip(axes.ravel(), panels, strict=True):
        im = ax.pcolormesh(rel, range_km, img, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.axhline(geom["seed_range_m"] / 1000.0, color="w", lw=0.7, alpha=0.6)
        ax.axvline(geom["seed_doppler_hz"] - geom["model_doppler_hz"], color="w", lw=0.7, alpha=0.6)
        ax.set_title(title, fontsize=9)
        if result is not None:
            ax.text(
                0.02,
                0.96,
                f"comb/off {result['comb_ratio_db']:.1f} dB",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Range (km)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Doppler relative to model (Hz)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Matched-filter power relative to panel median (dB)")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_h5(path: Path, geom: dict, results: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "Synthetic 16-code Clinton head-echo vertical stripe matched-filter diagnostic"
        h.attrs["station"] = "clinton"
        h.attrs["time_rel_s"] = geom["time_rel"]
        h.attrs["seed_range_m"] = geom["seed_range_m"]
        h.attrs["seed_doppler_hz"] = geom["seed_doppler_hz"]
        h.attrs["model_doppler_hz"] = geom["model_doppler_hz"]
        h.attrs["observed_peak_minus_median_db"] = geom["observed_contrast_db"]
        h["range_m"] = geom["range_m"]
        h["rel_freqs_hz"] = geom["rel_freqs"]
        h["observed_raw_db"] = geom["raw"]
        for variant in VARIANTS:
            g = h.create_group(variant.key)
            result = results[variant.key]
            g.attrs["title"] = variant.title
            g.attrs["signal_code_count"] = len(variant.signal_codes)
            g.attrs["filter_code_count"] = len(variant.filter_codes)
            g.attrs["n_channels"] = variant.n_channels
            g.attrs["direct_path_db"] = np.nan if variant.direct_path_db is None else variant.direct_path_db
            for key in ("target_amplitude", "direct_amplitude", "comb_ratio", "comb_ratio_db", "filtered_comb_ratio", "filtered_comb_ratio_db"):
                g.attrs[key] = result[key]
            g["raw_db"] = result["raw_db"]
            g["filtered_db"] = result["filtered_db"]


def write_tex(path: Path, results: dict[str, dict], geom: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_metric = comb_metric(geom["raw"], 16)
    observed_notched_metric = comb_metric(notch_2d_doppler_repetition(geom["raw"], 16, 1), 16)
    rows = [
        f"Observed Clinton raw patch & {observed_metric[1]:.1f} & "
        f"{observed_notched_metric[1]:.1f} \\\\"
    ]
    for variant in VARIANTS:
        result = results[variant.key]
        rows.append(
            f"{variant.title} & {result['comb_ratio_db']:.1f} & "
            f"{result['filtered_comb_ratio_db']:.1f} \\\\"
        )
    table_rows = "\n".join(rows)
    text = rf"""\subsection{{Synthetic Tests for Vertical Stripes in the 16-Code Matched Filter}}
\label{{app:head-vertical-stripe-simulation}}

The raw Clinton range--Doppler patch in Fig.~\ref{{fig:head-gaussian-example}}
contains weak nearly vertical structures with a spacing associated with the
100 Hz repeated-code ambiguity.  To test whether these stripes are an intrinsic
consequence of the 16-code coherent processing, the same matched-filter grid was
recomputed with synthetic voltage data.  The simulation used the Clinton code
length, interpolation factor, 16-code coherent window, Doppler bins, and range
gates of the 0.400 s example.  A single point-like head echo was inserted at
{geom["seed_range_m"] / 1000.0:.2f} km and {geom["seed_doppler_hz"]:.1f} Hz,
with the target amplitude adjusted so that the peak-to-median contrast of the
synthetic matched-filter image matched the observed raw patch
({geom["observed_contrast_db"]:.1f} dB).

Figure~\ref{{fig:head-vertical-stripe-simulation}} compares the observed raw
patch with several synthetic variants.  The first simulation uses one transmit
code, one matched code, and one receiver channel.  The following simulations add
the second receiver channel and then all five simultaneously transmitted code
seeds used in the experiment.  The final panel is a deliberately crude stress
test in which a strong zero-Doppler code-like leakage term is added; it is not a
physical direct-path model, but it tests whether stationary code leakage can
produce coherent Doppler-column structure.

\begin{{table}}[h]
  \centering
  \caption{{Image-frequency power in the 100 Hz comb columns for the synthetic
  Clinton vertical-stripe tests.  Values are the mean two-dimensional Fourier
  power in the repeated-code comb columns relative to the mean power outside
  those columns.  The same statistic is listed before and after the 2-D FFT
  notch used for the micro-Doppler diagnostic.
  \figsource{{reports/simulate_head_echo_vertical_stripes.py}}}}
  \begin{{tabular}}{{lrr}}
    \toprule
    Simulation & Raw comb/off (dB) & Notched comb/off (dB) \\
    \midrule
{table_rows}
    \bottomrule
  \end{{tabular}}
  \label{{tab:head-vertical-stripe-simulation}}
\end{{table}}

\begin{{figure}}
  \centering
  \includegraphics[width=\linewidth]{{reports/figures/head_echo_vertical_stripe_simulation.png}}
  \caption{{Synthetic Clinton matched-filter tests for the vertical stripes in
  the 16-code coherent range--Doppler output.  The first panel is the observed
  raw Clinton patch from Fig.~\ref{{fig:head-gaussian-example}}.  The remaining
  panels use simulated head echoes with the same range, Doppler shift, matched
  filter grid, and approximate peak-to-background contrast.  White reference
  lines mark the inserted range and Doppler.  Each panel is shown relative to
  its own median matched-filter power.  If the one-code case were already
  stripe dominated, the artifact would be intrinsic to the single repeated-code
  coherent integration.  If the stripes grew only when the other transmitters
  were added, multi-code cross-talk would be implicated.
  \figsource{{reports/simulate_head_echo_vertical_stripes.py}}}}
  \label{{fig:head-vertical-stripe-simulation}}
\end{{figure}}

The controlled synthetic point-target cases do not reproduce the observed
vertical stripe pattern as a dominant feature.  Adding the other simultaneously
transmitted code seeds increases the repeated-code comb statistic only modestly
relative to the one-code case, and the 2-D FFT notch strongly suppresses that
image-frequency component.  This argues against ordinary multi-code
cross-correlation of a compact head echo as the sole source of the observed
vertical stripes.  The stress test shows that a strong stationary code-like
leakage term can create more coherent columnar structure, so the next diagnostic
step should be to estimate direct-path and DC-like components together with the
head echo in a sparse range--Doppler model.  In that model the voltage data would
be represented by a sparse set of range--Doppler atoms plus slowly varying
per-range DC or direct-path terms, rather than by subtracting or notching the
matched-filter image after it has already mixed these components.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-fig", type=Path, default=OUT_FIG)
    parser.add_argument("--out-tex", type=Path, default=OUT_TEX)
    parser.add_argument("--out-h5", type=Path, default=OUT_H5)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    geom = load_geometry()
    codes = get_codes(geom["seeds"], geom["code_len"], 16, geom["range_interp"])
    rng = np.random.default_rng(args.seed)
    results = {variant.key: simulate_variant(variant, geom, codes, rng) for variant in VARIANTS}
    plot_results(geom, results, args.out_fig)
    save_h5(args.out_h5, geom, results)
    write_tex(args.out_tex, results, geom)
    for variant in VARIANTS:
        result = results[variant.key]
        print(f"{variant.key}: comb/off {result['comb_ratio_db']:.1f} dB, notched {result['filtered_comb_ratio_db']:.1f} dB")
    print(f"wrote {args.out_fig}")
    print(f"wrote {args.out_tex}")
    print(f"wrote {args.out_h5}")


if __name__ == "__main__":
    main()
