#!/usr/bin/env python3
"""Generate a head-echo micro-Doppler diagnostic figure.

The diagnostic follows the older meteor-head CW-radar analysis pattern: use the
picked head-echo range gate, form the coded-CW matched-filter Doppler spectrum,
remove the bulk Doppler using the best-fit trajectory Doppler model, and show
the residual spectral structure around the main echo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import geodetic_to_ecef, load_station_yaml  # noqa: E402

RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
TRAJECTORY_JSON = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
FIG_DIR = ROOT / "reports" / "figures"
OUT_FIG = FIG_DIR / "head_echo_microdoppler.png"
OUT_AMBIGUITY_FIG = FIG_DIR / "head_echo_microdoppler_ambiguity.png"
OUT_SUMMARY = ROOT / "reports" / "head_microdoppler_summary.tex"
C = 299792458.0

STATIONS = ("beverly", "clinton", "pomfret")
STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, CT",
}


def detrend_window(z: np.ndarray) -> np.ndarray:
    return z - np.median(z)


def bpsk(seed: int, length: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    code = rng.random(length)
    code = np.exp(2.0j * np.pi * code)
    code = np.angle(code)
    code = -1.0 * np.sign(code)
    return np.array(code, dtype=np.complex64)


def interpolate_rect(z: np.ndarray, interp: int) -> np.ndarray:
    if interp == 1:
        return z.astype(np.complex64, copy=False)
    z_rep = np.repeat(z, interp)
    wf = np.repeat(1.0 / float(interp), interp)
    zi = np.fft.ifft(np.fft.fft(z_rep) * np.fft.fft(wf, len(z_rep)))
    zi = np.roll(zi, -int(interp / 2))
    return zi.astype(np.complex64)


def get_code(seed: int, code_len: int, n_coh: int, range_interp: int) -> np.ndarray:
    code = bpsk(seed, code_len)
    code = interpolate_rect(code, range_interp)
    code = code / np.max(np.abs(code))
    return np.tile(code, n_coh).astype(np.complex64)


def get_codes(seeds: np.ndarray, code_len: int, n_coh: int, range_interp: int) -> list[np.ndarray]:
    return [get_code(int(seed), code_len, n_coh, range_interp) for seed in seeds]


def rf_file_for_time(data_dir: Path, station: str, channel: int, iso_hour: str) -> list[Path]:
    files = sorted((data_dir / station / f"ch{channel:03d}" / iso_hour).glob("rf@*.h5"))
    if not files:
        raise FileNotFoundError(f"No RF files under {data_dir}/{station}/ch{channel:03d}/{iso_hour}")
    return files


def read_channel_files(files: list[Path], start_second: float, seconds: float | None) -> tuple[np.ndarray, float, int]:
    chunks = []
    sample_rate = None
    abs_index0 = None
    for fn in files:
        with h5py.File(fn, "r") as h:
            ds = h["rf_data"]
            sr = float(ds.attrs["sample_rate_numerator"]) / float(ds.attrs["sample_rate_denominator"])
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample-rate mismatch in {fn}: {sr} != {sample_rate}")
            if abs_index0 is None:
                abs_index0 = int(h["rf_data_index"][0, 0])
            chunks.append(ds[:, 0].astype(np.complex64))
    z = np.concatenate(chunks)
    i0 = int(round(start_second * sample_rate))
    i1 = len(z) if seconds is None else min(len(z), i0 + int(round(seconds * sample_rate)))
    if i1 <= i0:
        raise ValueError("Requested span is outside RF data")
    return z[i0:i1], float(sample_rate), int(abs_index0 + i0)


def read_station_channels(data_dir: Path, station: str, channels: np.ndarray, iso_hour: str, start_second: float, seconds: float, range_interp: int):
    out = []
    sample_rate = None
    abs_index0 = None
    for ch in channels:
        z, sr, idx0 = read_channel_files(
            rf_file_for_time(data_dir, station, int(ch), iso_hour),
            start_second=start_second,
            seconds=seconds,
        )
        out.append(interpolate_rect(z, range_interp))
        sample_rate = sr if sample_rate is None else sample_rate
        abs_index0 = idx0 if abs_index0 is None else abs_index0
    return out, float(sample_rate), int(abs_index0)


def estimate_range_gate_dc(channel_data: list[np.ndarray], range_idx: np.ndarray, sample_rate_i: float, seconds: float) -> np.ndarray | None:
    if seconds <= 0:
        return None
    n_cal = int(round(seconds * sample_rate_i))
    if n_cal <= 0:
        return None
    offsets = np.zeros((len(channel_data), len(range_idx)), dtype=np.complex64)
    for chi, z in enumerate(channel_data):
        max_stop = int(range_idx[-1]) + n_cal
        if max_stop > len(z):
            raise ValueError("Not enough data to estimate range-gate DC offsets")
        cs = np.concatenate(([0.0j], np.cumsum(z[:max_stop], dtype=np.complex128)))
        offsets[chi, :] = ((cs[range_idx + n_cal] - cs[range_idx]) / float(n_cal)).astype(np.complex64)
    return offsets


def load_station_observations() -> tuple[dict, dict[str, list[dict]]]:
    payload = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8"))
    by_station = {station: [] for station in STATIONS}
    for obs in payload["observations"]:
        station = obs["station"]
        if station in by_station:
            by_station[station].append(obs)
    for station in by_station:
        by_station[station].sort(key=lambda row: row["unix_time"])
    return payload, by_station


def load_doppler_model(payload: dict, data_dir: Path) -> dict:
    geom = {}
    for station in STATIONS:
        site = load_station_yaml(data_dir, station)
        geom[station] = {
            "tx": geodetic_to_ecef(*site["gps_tx"]),
            "rx": geodetic_to_ecef(*site["gps_rx"]),
            "frequency_hz": float(site["frequency_hz"]),
        }
    return {
        "t_ref_unix": float(payload["t_ref_unix"]),
        "position_ref_ecef_m": np.asarray(payload["position_ref_ecef_m"], dtype=float),
        "velocity_ecef_mps": np.asarray(payload["velocity_ecef_mps"], dtype=float),
        "geom": geom,
    }


def best_fit_doppler_hz(model: dict, station: str, unix_time: float) -> float:
    geom = model["geom"][station]
    dt = float(unix_time) - model["t_ref_unix"]
    pos = model["position_ref_ecef_m"] + model["velocity_ecef_mps"] * dt
    vel = model["velocity_ecef_mps"]
    u_tx = (pos - geom["tx"]) / np.linalg.norm(pos - geom["tx"])
    u_rx = (pos - geom["rx"]) / np.linalg.norm(pos - geom["rx"])
    range_rate_mps = float(np.dot(u_tx + u_rx, vel))
    return geom["frequency_hz"] * range_rate_mps / C


def code_ambiguity_kernel(codes: list[np.ndarray]) -> np.ndarray:
    """Return the multi-code matched-filter power response to a centered tone.

    The micro-Doppler diagnostic sums power over all candidate transmit-code
    seeds.  For an echo from one seed, the matched seed gives the desired
    response, while the other repeated seeds produce a comb-like cross-code
    ambiguity response.  Averaging over possible true seeds gives the
    station-independent point-spread function used below.
    """
    kernel = np.zeros(len(codes[0]), dtype=np.float64)
    for true_code in codes:
        for test_code in codes:
            spec = np.fft.fftshift(np.fft.fft(true_code * np.conj(test_code)))
            kernel += np.abs(spec) ** 2
    kernel /= float(len(codes))
    kernel /= np.sum(kernel)
    return kernel


def centered_kernel_for_frequency_grid(full_kernel: np.ndarray, fvec: np.ndarray, rel_freqs: np.ndarray) -> np.ndarray:
    center = len(full_kernel) // 2
    full_offsets = np.arange(len(full_kernel)) - center
    df = float(np.median(np.diff(fvec)))
    rel_offsets = np.rint(rel_freqs / df).astype(int)
    kernel = np.zeros(len(rel_freqs), dtype=np.float64)
    good = (rel_offsets + center >= 0) & (rel_offsets + center < len(full_kernel))
    kernel[good] = full_kernel[rel_offsets[good] + center]
    if np.sum(kernel) <= 0:
        raise ValueError("Ambiguity kernel has zero weight on the requested frequency grid")
    kernel /= np.sum(kernel)
    return kernel


def clean_deconvolution_rows(
    spectra: np.ndarray,
    kernel: np.ndarray,
    rel_freqs: np.ndarray,
    picked_residuals_hz: np.ndarray,
    n_iter: int = 12,
    gain: float = 0.35,
    search_half_width_hz: float = 180.0,
) -> np.ndarray:
    """CLEAN the ambiguity comb using the manual Doppler picks as a weak prior."""
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel /= np.max(kernel)
    center = len(kernel) // 2
    df = float(np.median(np.diff(rel_freqs)))
    sigma_bins = max(1.0, 12.0 / df)
    x = np.arange(len(kernel)) - center
    restore = np.exp(-0.5 * (x / sigma_bins) ** 2)
    restore /= np.sum(restore)

    out = np.empty_like(spectra, dtype=np.float64)
    cols = np.arange(spectra.shape[1])
    for i, y0 in enumerate(spectra):
        y = np.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0)
        baseline = np.nanpercentile(y[y > 0], 20) if np.any(y > 0) else 0.0
        residual = np.maximum(y - baseline, 0.0)
        clean = np.zeros_like(residual)
        search = np.abs(rel_freqs - picked_residuals_hz[i]) <= search_half_width_hz
        if not np.any(search):
            search = np.abs(rel_freqs) <= search_half_width_hz
        for _ in range(n_iter):
            masked = np.where(search, residual, -np.inf)
            peak_idx = int(np.argmax(masked))
            peak = residual[peak_idx]
            if not np.isfinite(peak) or peak <= 0:
                break
            amp = gain * peak
            clean[peak_idx] += amp
            shift = peak_idx - center
            k_idx = cols - shift
            good = (k_idx >= 0) & (k_idx < len(kernel))
            residual[good] = np.maximum(residual[good] - amp * kernel[k_idx[good]], 0.0)
        restored = np.convolve(clean, restore, mode="same")
        residual_floor = np.minimum(residual, np.nanpercentile(residual, 70) if np.any(residual > 0) else 0.0)
        out[i, :] = np.maximum(restored + residual_floor + baseline, 1e-30)
    return out


def plot_ambiguity_kernel(rel_freqs: np.ndarray, kernel: np.ndarray, n_coh: int) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    k_db = 10.0 * np.log10(np.maximum(kernel / np.max(kernel), 1e-12))
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.plot(rel_freqs, k_db, color="black", lw=0.9)
    df = float(np.median(np.diff(rel_freqs)))
    repeat_hz = n_coh * df
    for x in np.arange(-1800.0, 1800.1, repeat_hz):
        ax.axvline(x, color="#c73e3a", lw=0.55, alpha=0.28)
    ax.set_xlim(float(np.nanmin(rel_freqs)), float(np.nanmax(rel_freqs)))
    ax.set_ylim(-60, 1)
    ax.set_xlabel("Residual Doppler offset (Hz)")
    ax.set_ylabel("Relative response (dB)")
    ax.set_title(f"Multi-code ambiguity kernel; repeat spacing {repeat_hz:.1f} Hz ({n_coh} FFT bins)")
    fig.savefig(OUT_AMBIGUITY_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return OUT_AMBIGUITY_FIG


def station_microdoppler(station: str, observations: list[dict], doppler_model: dict, rel_half_width_hz: float = 1800.0) -> dict:
    rdmf_file = RDMF_DIR / f"rdmf_peak_{station}_merged.h5"
    data_dir = ROOT / "2026_05_30_bolide"
    with h5py.File(rdmf_file, "r") as h:
        channels = h["channels"][()]
        seeds = h["seeds"][()]
        code_len = int(h["code_len"][()])
        product_n_coh = int(h["n_coh"][()])
        n_coh = 16
        n_incoh = 16
        step_codes = 1
        range_interp = int(h["range_interp"][()])
        sample_rate = float(h["sample_rate"][()])
        range_resolution_m = float(h["range_resolution_m"][()])
        start_second = float(h.attrs["start_second"])
        seconds = max(float(h.attrs["seconds"]), 10.0)
        iso_hour = str(h.attrs["iso_hour"])
        t0_sample = float(h["t0_sample"][()])
        dc_cancel = bool(int(h["dc_cancel"][()])) if "dc_cancel" in h else False
        dc_cancel_seconds = float(h["dc_cancel_seconds"][()]) if "dc_cancel_seconds" in h else 5.0
        dc_cancel_start_second = float(h["dc_cancel_start_second"][()]) if "dc_cancel_start_second" in h else 0.0
        range_idx_all = h["range_idx"][()]

    sample_rate_i = sample_rate * range_interp
    code_len_i = code_len * range_interp
    win_len = n_coh * code_len_i
    step_len = step_codes * code_len_i
    fvec = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    df = float(np.median(np.diff(fvec)))
    rel_freqs = np.arange(-rel_half_width_hz, rel_half_width_hz + 0.5 * df, df)
    code_seeds = [int(seed) for seed in seeds]
    codes = get_codes(seeds, code_len, n_coh, range_interp)
    ambiguity_kernel_full = code_ambiguity_kernel(codes)
    ambiguity_kernel = centered_kernel_for_frequency_grid(ambiguity_kernel_full, fvec, rel_freqs)
    zc, _sr, abs_index0 = read_station_channels(
        data_dir,
        station,
        channels,
        iso_hour,
        start_second=start_second,
        seconds=seconds + 0.5,
        range_interp=range_interp,
    )
    if abs(abs_index0 - t0_sample) > 1:
        raise ValueError(f"{station}: raw t0 {abs_index0} does not match RDMF t0 {t0_sample}")

    dc_offsets = None
    if dc_cancel:
        zc_cal, _sr_cal, _idx_cal = read_station_channels(
            data_dir,
            station,
            channels,
            iso_hour,
            start_second=dc_cancel_start_second,
            seconds=dc_cancel_seconds + (float(range_idx_all[-1]) + 1.0) / sample_rate_i,
            range_interp=range_interp,
        )
        dc_offsets = estimate_range_gate_dc(zc_cal, range_idx_all, sample_rate_i, dc_cancel_seconds)

    times = []
    ranges_km = []
    model_dopplers_hz = []
    picked_dopplers_hz = []
    spectra = []
    gate_m = range_resolution_m / range_interp
    for obs in observations:
        raw_sample = int(round(obs["unix_time"] * sample_rate - t0_sample))
        start_i = raw_sample * range_interp
        range_idx = int(round(float(obs["range_m"]) / gate_m))
        incoh_stop_i = start_i + (n_incoh - 1) * step_len + win_len + range_idx
        if start_i < 0 or incoh_stop_i >= min(map(len, zc)):
            continue
        dp = np.zeros(win_len, dtype=np.float64)
        ri_pos = int(np.argmin(np.abs(range_idx_all - range_idx)))
        for incoh_i in range(n_incoh):
            incoh_start_i = start_i + incoh_i * step_len
            for chi, z in enumerate(zc):
                z0 = detrend_window(z[incoh_start_i : incoh_start_i + win_len + range_idx + 1])
                echo = z0[range_idx : range_idx + win_len]
                if dc_offsets is not None:
                    echo = echo - dc_offsets[chi, ri_pos]
                for code in codes:
                    spec = np.fft.fftshift(np.fft.fft(echo * np.conj(code)))
                    dp += np.abs(spec) ** 2
        bulk = best_fit_doppler_hz(doppler_model, station, float(obs["unix_time"]))
        spectra.append(np.interp(rel_freqs, fvec - bulk, dp, left=np.nan, right=np.nan))
        times.append(float(obs["unix_time"]))
        ranges_km.append(float(obs["range_m"]) / 1000.0)
        model_dopplers_hz.append(bulk)
        picked_dopplers_hz.append(float(obs["doppler_hz"]))

    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.size == 0:
        raise ValueError(f"No spectra generated for {station}")
    picked_residuals_hz = np.asarray(picked_dopplers_hz) - np.asarray(model_dopplers_hz)
    deconvolved = clean_deconvolution_rows(spectra, ambiguity_kernel, rel_freqs, picked_residuals_hz)
    spec_db = 10.0 * np.log10(np.maximum(spectra, 1e-30))
    deconvolved_db = 10.0 * np.log10(np.maximum(deconvolved, 1e-30))
    # Remove a per-station robust background level so panels share an SNR-like scale.
    spec_db -= np.nanmedian(spec_db)
    deconvolved_db -= np.nanmedian(deconvolved_db)
    return {
        "station": station,
        "times": np.asarray(times),
        "time_rel": np.asarray(times) - 1780164380.0,
        "ranges_km": np.asarray(ranges_km),
        "model_dopplers_hz": np.asarray(model_dopplers_hz),
        "picked_dopplers_hz": np.asarray(picked_dopplers_hz),
        "rel_freqs": np.asarray(rel_freqs),
        "spec_db": spec_db,
        "deconvolved_db": deconvolved_db,
        "ambiguity_kernel": ambiguity_kernel,
        "code_seeds": code_seeds,
        "product_n_coh": product_n_coh,
        "micro_n_coh": n_coh,
        "micro_n_incoh": n_incoh,
        "micro_step_codes": step_codes,
        "micro_clean_iterations": 12,
        "micro_clean_gain": 0.35,
    }


def plot_microdoppler(results: list[dict]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    mappable = None
    for ax, res in zip(axes, results):
        ax.set_facecolor("black")
        t = res["time_rel"]
        f = res["rel_freqs"]
        z = res["deconvolved_db"].T
        mappable = ax.pcolormesh(t, f, z, shading="auto", cmap="inferno", vmin=0.0, vmax=18.0)
        picked_residual = res["picked_dopplers_hz"] - res["model_dopplers_hz"]
        ax.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.scatter(
            t,
            picked_residual,
            s=10,
            marker="o",
            facecolor="#79e6ff",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            zorder=3,
        )
        ax.set_ylabel("$f-f_{D,model}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo")
        ax.text(
            0.01,
            0.93,
            f"range {np.nanmedian(res['ranges_km']):.0f} km, model Doppler {np.nanmedian(res['model_dopplers_hz']):.0f} Hz",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Seconds after 18:06:20 UTC")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, pad=0.012)
        cb.set_label("Relative matched-filter power (dB)")
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG


def write_summary(results: list[dict]) -> None:
    lines = [
        "% Auto-generated by reports/generate_head_microdoppler.py\n",
        f"\\newcommand{{\\HeadMicroDopplerStationCount}}{{{len(results)}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerCodeSeedCount}}{{{len(results[0]['code_seeds'])}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerNCoherentCodes}}{{{results[0]['micro_n_coh']}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerNIncoherentSpectra}}{{{results[0]['micro_n_incoh']}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerStepCodes}}{{{results[0]['micro_step_codes']}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerCleanIterations}}{{{results[0]['micro_clean_iterations']}}}\n",
        f"\\newcommand{{\\HeadMicroDopplerCleanGain}}{{{results[0]['micro_clean_gain']:.2f}}}\n",
    ]
    for res in results:
        name = res["station"].title()
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}N}}{{{len(res['times'])}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianRangeKm}}{{{np.nanmedian(res['ranges_km']):.0f}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianDopplerHz}}{{{np.nanmedian(res['model_dopplers_hz']):.0f}}}\n")
        lines.append(f"\\newcommand{{\\HeadMicroDoppler{name}MedianPickedDopplerHz}}{{{np.nanmedian(res['picked_dopplers_hz']):.0f}}}\n")
    OUT_SUMMARY.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    data_dir = ROOT / "2026_05_30_bolide"
    payload, observations = load_station_observations()
    doppler_model = load_doppler_model(payload, data_dir)
    results = [station_microdoppler(station, observations[station], doppler_model) for station in STATIONS]
    ambiguity = plot_ambiguity_kernel(results[0]["rel_freqs"], results[0]["ambiguity_kernel"], results[0]["micro_n_coh"])
    out = plot_microdoppler(results)
    write_summary(results)
    print(ambiguity)
    print(out)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
