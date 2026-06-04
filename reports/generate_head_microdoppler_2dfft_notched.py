#!/usr/bin/env python3
"""Generate head-echo micro-Doppler using a 2-D FFT notch per RD patch."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.generate_head_microdoppler import (  # noqa: E402
    STATION_LABELS,
    STATIONS,
    best_fit_doppler_hz,
    detrend_window,
    estimate_range_gate_dc,
    get_codes,
    load_doppler_model,
    load_station_observations,
    read_station_channels,
)

RDMF_DIR = ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431"
DATA_DIR = ROOT / "2026_05_30_bolide"
FIG_DIR = ROOT / "reports" / "figures"
OUT_FIG = FIG_DIR / "head_echo_microdoppler_2dfft_notched.png"
OUT_CACHE = ROOT / "reports" / "head_echo_microdoppler_2dfft_notched.h5"
EVENT_T0 = 1780164380.0

G = {}


def notch_2d_doppler_repetition(img_db: np.ndarray, period_bins: int, notch_half_width: int) -> np.ndarray:
    img0 = img_db - np.mean(img_db)
    F = np.fft.fft2(img0)
    ny, nx = img_db.shape
    fund = nx // period_bins
    if fund <= 0:
        return img_db
    mask = np.ones_like(F, dtype=bool)
    for hidx in range(1, nx // (2 * fund) + 1):
        k = hidx * fund
        if k > nx // 2:
            break
        for kk in (k, (-k) % nx):
            for col in range(kk - notch_half_width, kk + notch_half_width + 1):
                mask[:, col % nx] = False
    return np.fft.ifft2(F * mask).real + np.mean(img_db)


def init_station(payload: dict, station: str, args: argparse.Namespace) -> None:
    doppler_model = load_doppler_model(payload, DATA_DIR)
    rdmf_file = RDMF_DIR / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(rdmf_file, "r") as h:
        channels = h["channels"][()]
        seeds = h["seeds"][()]
        code_len = int(h["code_len"][()])
        range_interp = int(h["range_interp"][()])
        sample_rate = float(h["sample_rate"][()])
        range_resolution_m = float(h["range_resolution_m"][()])
        start_second = float(h.attrs["start_second"])
        seconds = max(float(h.attrs["seconds"]), 10.0)
        iso_hour = str(h.attrs["iso_hour"])
        t0_sample = float(h["t0_sample"][()])
        dc_cancel = bool(int(h["dc_cancel"][()])) if "dc_cancel" in h and args.dc_cancel else False
        dc_cancel_seconds = float(h["dc_cancel_seconds"][()]) if "dc_cancel_seconds" in h else 5.0
        dc_cancel_start_second = float(h["dc_cancel_start_second"][()]) if "dc_cancel_start_second" in h else 0.0

    sample_rate_i = sample_rate * range_interp
    code_len_i = code_len * range_interp
    win_len = args.n_coh * code_len_i
    step_len = args.step_codes * code_len_i
    fvec_full = np.fft.fftshift(np.fft.fftfreq(win_len, d=1.0 / sample_rate_i))
    gate_m = range_resolution_m / range_interp
    codes = get_codes(seeds, code_len, args.n_coh, range_interp)
    zc, _sr, abs_index0 = read_station_channels(
        DATA_DIR,
        station,
        channels,
        iso_hour,
        start_second=start_second,
        seconds=seconds + 0.5,
        range_interp=range_interp,
    )
    if abs(abs_index0 - t0_sample) > 1:
        raise ValueError(f"{station}: raw t0 {abs_index0} does not match RDMF t0 {t0_sample}")

    G.clear()
    G.update(
        {
            "args": args,
            "station": station,
            "doppler_model": doppler_model,
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_rate_i": sample_rate_i,
            "t0_sample": t0_sample,
            "fvec_full": fvec_full,
            "gate_m": gate_m,
            "win_len": win_len,
            "step_len": step_len,
            "codes": codes,
            "zc": zc,
            "dc_cancel": dc_cancel,
            "dc_cancel_seconds": dc_cancel_seconds,
            "dc_cancel_start_second": dc_cancel_start_second,
            "range_interp": range_interp,
            "iso_hour": iso_hour,
        }
    )


def compute_filtered_patch(item: tuple[int, dict]) -> dict | None:
    idx, obs = item
    args = G["args"]
    station = G["station"]
    bulk = best_fit_doppler_hz(G["doppler_model"], station, float(obs["unix_time"]))
    fvec_full = G["fvec_full"]
    fidx = np.where(np.abs(fvec_full - bulk) <= args.doppler_half_width_hz)[0]
    rel_f = fvec_full[fidx] - bulk

    gate_m = G["gate_m"]
    head_range_m = float(obs["range_m"])
    ridx = np.arange(
        max(0, int(np.floor((head_range_m - args.range_half_width_km * 1000.0) / gate_m))),
        int(np.ceil((head_range_m + args.range_half_width_km * 1000.0) / gate_m)) + 1,
        dtype=np.int64,
    )
    raw_sample = int(round(float(obs["unix_time"]) * G["sample_rate"] - G["t0_sample"]))
    start_i = raw_sample * G["range_interp"]
    read_len = G["win_len"] + int(ridx[-1]) + 1
    incoh_stop_i = start_i + (args.n_incoh - 1) * G["step_len"] + read_len
    if start_i < 0 or incoh_stop_i >= min(map(len, G["zc"])):
        return None

    dc_offsets = None
    if G["dc_cancel"]:
        zc_cal, _sr_cal, _idx_cal = read_station_channels(
            DATA_DIR,
            station,
            G["channels"],
            G["iso_hour"],
            start_second=G["dc_cancel_start_second"],
            seconds=G["dc_cancel_seconds"] + (float(ridx[-1]) + 1.0) / G["sample_rate_i"],
            range_interp=G["range_interp"],
        )
        dc_offsets = estimate_range_gate_dc(zc_cal, ridx, G["sample_rate_i"], G["dc_cancel_seconds"])

    S = np.zeros((len(ridx), len(fidx)), dtype=np.float64)
    for incoh_i in range(args.n_incoh):
        incoh_start_i = start_i + incoh_i * G["step_len"]
        for chi, z in enumerate(G["zc"]):
            z0 = detrend_window(z[incoh_start_i : incoh_start_i + read_len])
            windows = np.lib.stride_tricks.sliding_window_view(z0, G["win_len"])[ridx, :]
            if dc_offsets is not None:
                windows = windows - dc_offsets[chi, :, None]
            for code in G["codes"]:
                Z = np.fft.fftshift(np.fft.fft(windows * np.conj(code), axis=1), axes=1)[:, fidx]
                S += np.abs(Z) ** 2

    img_db = 10.0 * np.log10(np.maximum(S, 1e-30))
    filtered_db = notch_2d_doppler_repetition(img_db, args.n_coh, args.notch_half_width)
    range_m = ridx.astype(np.float64) * gate_m
    return {
        "idx": idx,
        "time": float(obs["unix_time"]),
        "range_m": range_m,
        "seed_range_m": head_range_m,
        "model_doppler_hz": bulk,
        "picked_doppler_hz": float(obs["doppler_hz"]),
        "rel_freqs": rel_f,
        "raw_db": img_db,
        "filtered_db": filtered_db,
    }


def process_observation(item: tuple[int, dict]) -> dict | None:
    patch = compute_filtered_patch(item)
    if patch is None:
        return None
    head_range_m = patch["seed_range_m"]
    range_m = patch["range_m"]
    filtered_db = patch["filtered_db"]
    row_idx = int(np.argmin(np.abs(range_m - head_range_m)))
    return {
        "idx": patch["idx"],
        "time": patch["time"],
        "range_km": head_range_m / 1000.0,
        "model_doppler_hz": patch["model_doppler_hz"],
        "picked_doppler_hz": patch["picked_doppler_hz"],
        "rel_freqs": patch["rel_freqs"],
        "row_db": filtered_db[row_idx, :],
    }


def run_station(station: str, obs: list[dict], payload: dict, args: argparse.Namespace) -> dict:
    print(f"{station}: loading raw data", flush=True)
    init_station(payload, station, args)
    n_workers = min(args.workers, len(obs))
    print(f"{station}: processing {len(obs)} rows with {n_workers} workers", flush=True)
    if n_workers <= 1:
        rows = [process_observation(item) for item in enumerate(obs)]
    else:
        ctx = mp.get_context("fork")
        # On Linux/fork this lets workers share the already-read RF arrays via
        # copy-on-write instead of every worker rereading the station files.
        with ctx.Pool(n_workers) as pool:
            rows = []
            for row in pool.imap_unordered(process_observation, list(enumerate(obs)), chunksize=1):
                rows.append(row)
                done = len(rows)
                if done == 1 or done == len(obs) or done % max(1, len(obs) // 10) == 0:
                    print(f"{station}: {done}/{len(obs)} rows returned", flush=True)
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda row: row["time"])
    if not rows:
        raise ValueError(f"{station}: no rows generated")
    rel_ref = rows[0]["rel_freqs"]
    spec = []
    for row in rows:
        vals = row["row_db"]
        rel = row["rel_freqs"]
        if len(rel) != len(rel_ref) or np.nanmax(np.abs(rel - rel_ref)) > 0.51 * np.nanmedian(np.diff(rel_ref)):
            vals = np.interp(rel_ref, rel, vals, left=np.nan, right=np.nan)
        spec.append(vals)
    spec = np.asarray(spec)
    spec -= np.nanmedian(spec)
    return {
        "station": station,
        "times": np.asarray([row["time"] for row in rows]),
        "time_rel": np.asarray([row["time"] - EVENT_T0 for row in rows]),
        "ranges_km": np.asarray([row["range_km"] for row in rows]),
        "model_dopplers_hz": np.asarray([row["model_doppler_hz"] for row in rows]),
        "picked_dopplers_hz": np.asarray([row["picked_doppler_hz"] for row in rows]),
        "rel_freqs": rel_ref,
        "spec_db": spec,
    }


def plot_results(results: list[dict], out_fig: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    finite = np.concatenate([res["spec_db"][np.isfinite(res["spec_db"])] for res in results])
    vmin = 0.0
    vmax = max(8.0, float(np.nanpercentile(finite, 99.4))) if finite.size else 18.0
    mappable = None
    for ax, res in zip(axes, results):
        ax.set_facecolor("black")
        mappable = ax.pcolormesh(
            res["time_rel"],
            res["rel_freqs"],
            res["spec_db"].T,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0.0, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.set_ylabel("$f-f_{D,model}$ (Hz)")
        ax.set_title(f"{STATION_LABELS[res['station']]} head echo")
        ax.text(
            0.01,
            0.93,
            f"range {np.nanmedian(res['ranges_km']):.0f} km, 2D FFT notch",
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
        cb.set_label("2D-FFT-notched matched-filter power (dB rel.)")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["description"] = "2D FFT-notched head-echo micro-Doppler rows"
        h.attrs["event_t0_unix"] = EVENT_T0
        h["stations"] = np.asarray([res["station"].encode("utf-8") for res in results])
        for res in results:
            g = h.create_group(res["station"])
            g["time_rel"] = res["time_rel"]
            g["ranges_km"] = res["ranges_km"]
            g["model_dopplers_hz"] = res["model_dopplers_hz"]
            g["picked_dopplers_hz"] = res["picked_dopplers_hz"]
            g["rel_freqs"] = res["rel_freqs"]
            g["spec_db"] = res["spec_db"]


def load_results(path: Path) -> list[dict]:
    results = []
    with h5py.File(path, "r") as h:
        for station_b in h["stations"][()]:
            station = station_b.decode("utf-8") if isinstance(station_b, bytes) else str(station_b)
            g = h[station]
            results.append(
                {
                    "station": station,
                    "time_rel": g["time_rel"][()],
                    "ranges_km": g["ranges_km"][()],
                    "model_dopplers_hz": g["model_dopplers_hz"][()],
                    "picked_dopplers_hz": g["picked_dopplers_hz"][()],
                    "rel_freqs": g["rel_freqs"][()],
                    "spec_db": g["spec_db"][()],
                }
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations", nargs="+", default=list(STATIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--range-half-width-km", type=float, default=40.0)
    p.add_argument("--doppler-half-width-hz", type=float, default=1800.0)
    p.add_argument("--n-coh", type=int, default=16)
    p.add_argument("--n-incoh", type=int, default=16)
    p.add_argument("--step-codes", type=int, default=1)
    p.add_argument("--notch-half-width", type=int, default=1)
    p.add_argument("--dc-cancel", action="store_true", help="Apply product range-gate DC cancellation before the RD patch FFT.")
    p.add_argument("--out", type=Path, default=OUT_FIG)
    p.add_argument("--cache", type=Path, default=OUT_CACHE)
    p.add_argument("--plot-only", action="store_true", help="Plot from --cache without recomputing filtered RD patches.")
    args = p.parse_args()

    if args.plot_only:
        results = load_results(args.cache)
    else:
        payload, observations = load_station_observations()
        results = [run_station(station, observations[station], payload, args) for station in args.stations]
        save_results(results, args.cache)
        print(args.cache, flush=True)
    plot_results(results, args.out)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
