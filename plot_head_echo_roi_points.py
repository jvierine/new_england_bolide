#!/usr/bin/env python3
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ANNOTATIONS = "annotations/head_echo_rois.json"
C = 299792458.0


def iso_utc(t):
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


def load_rdmf(fname):
    with h5py.File(fname, "r") as h:
        sr = float(h["sample_rate"][()])
        return {
            "station": h["station"][()].decode(),
            "t_unix": h["t_sample"][()] / sr,
            "range_km": h["range_km"][()],
            "P": h["P_head"][()] if "P_head" in h else h["P"][()],
            "N": h["N"][()],
            "D": h["D_head"][()] if "D_head" in h else h["D"][()],
        }


def extract_roi_peak_points(annotation, snr_threshold_db=5.0):
    data = load_rdmf(annotation["source_file"])
    t = data["t_unix"]
    r = data["range_km"]
    t0 = annotation["time_range"]["start_unix"]
    t1 = annotation["time_range"]["end_unix"]
    r0 = annotation["range_km"]["min"]
    r1 = annotation["range_km"]["max"]
    keep_t = np.where((t >= min(t0, t1)) & (t <= max(t0, t1)))[0]
    keep_r = np.where((r >= min(r0, r1)) & (r <= max(r0, r1)))[0]
    if len(keep_t) == 0 or len(keep_r) == 0:
        return []

    snr = data["P"] / np.maximum(data["N"], 1e-12)
    snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
    points = []
    for ti in keep_t:
        local_snr = snr_db[ti, keep_r]
        imax = int(np.nanargmax(local_snr))
        if local_snr[imax] < snr_threshold_db:
            continue
        ri = keep_r[imax]
        points.append(
            {
                "station": data["station"],
                "unix_time": float(t[ti]),
                "range_km": float(r[ri]),
                "doppler_hz": float(data["D"][ti, ri]),
                "snr_db": float(snr_db[ti, ri]),
            }
        )
    return points


def robust_mask_linear(x, y, sigma=4.0, min_points=6):
    if len(x) < min_points:
        return np.ones(len(x), dtype=bool)
    mask = np.isfinite(x) & np.isfinite(y)
    for _ in range(3):
        if np.sum(mask) < min_points:
            break
        coef = np.polyfit(x[mask], y[mask], deg=1)
        resid = y - np.polyval(coef, x)
        med = np.nanmedian(resid[mask])
        mad = 1.4826 * np.nanmedian(np.abs(resid[mask] - med))
        if not np.isfinite(mad) or mad <= 0:
            break
        mask &= np.abs(resid - med) <= sigma * mad
    return mask


def robust_mask_median(y, sigma=4.0, min_points=6):
    y = np.asarray(y, dtype=float)
    if len(y) < min_points:
        return np.ones(len(y), dtype=bool)
    med = np.nanmedian(y)
    mad = 1.4826 * np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.isfinite(y)
    return np.isfinite(y) & (np.abs(y - med) <= sigma * mad)


def reject_outliers(points, delay_sigma=4.0, doppler_sigma=4.0):
    kept = []
    rejected = []
    for station in sorted(set(p["station"] for p in points)):
        ps = [p for p in points if p["station"] == station]
        t0 = min(p["unix_time"] for p in ps)
        t = np.array([p["unix_time"] - t0 for p in ps])
        delay_ms = np.array([p["range_km"] * 1000.0 / C * 1e3 for p in ps])
        doppler_hz = np.array([p["doppler_hz"] for p in ps])
        mask_delay = robust_mask_linear(t, delay_ms, sigma=delay_sigma)
        mask_dop = robust_mask_median(doppler_hz, sigma=doppler_sigma)
        mask = mask_delay & mask_dop
        for p, ok in zip(ps, mask):
            (kept if ok else rejected).append(p)
    return kept, rejected


def load_points(annotation_file, snr_threshold_db, reject=True, delay_sigma=4.0, doppler_sigma=4.0):
    annotations = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    all_points = []
    for station, entry in annotations.get("stations", {}).items():
        if not entry.get("has_head_echo"):
            continue
        points = extract_roi_peak_points(entry, snr_threshold_db=snr_threshold_db)
        all_points.extend(points)
    all_points.sort(key=lambda p: (p["station"], p["unix_time"]))
    if not reject:
        return all_points, []
    kept, rejected = reject_outliers(all_points, delay_sigma=delay_sigma, doppler_sigma=doppler_sigma)
    kept.sort(key=lambda p: (p["station"], p["unix_time"]))
    rejected.sort(key=lambda p: (p["station"], p["unix_time"]))
    return kept, rejected


def plot_points(points):
    if not points:
        raise SystemExit("No ROI points found. Draw/save ROI boxes first.")

    stations = sorted(set(p["station"] for p in points))
    t_ref = min(p["unix_time"] for p in points)
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)

    for station in stations:
        ps = [p for p in points if p["station"] == station]
        t = np.array([p["unix_time"] - t_ref for p in ps])
        delay_ms = np.array([p["range_km"] * 1000.0 / C * 1e3 for p in ps])
        d = np.array([p["doppler_hz"] for p in ps])
        s = np.array([p["snr_db"] for p in ps])
        axes[0].plot(t, delay_ms, lw=0.9, alpha=0.55)
        axes[0].scatter(t, delay_ms, s=28, label=station)
        axes[1].plot(t, d / 1e3, lw=0.9, alpha=0.55)
        axes[1].scatter(t, d / 1e3, s=28, label=station)
        axes[2].plot(t, s, marker=".", lw=1.0, label=station)

    range_axis = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda delay_ms: delay_ms * C / 1e6,
            lambda range_km: range_km * 1e6 / C,
        ),
    )
    range_axis.set_ylabel("Bistatic range (km)")
    axes[0].set_ylabel("Estimated delay (ms)")
    axes[0].set_title("Head echo ROI peak delay/range, Doppler, and SNR")
    axes[0].legend()
    axes[0].grid(True, color="0.9")

    axes[1].set_ylabel("Doppler shift (kHz)")
    axes[1].grid(True, color="0.9")
    axes[1].legend()

    axes[2].set_xlabel(f"Seconds after {iso_utc(t_ref)}")
    axes[2].set_ylabel("SNR (dB)")
    axes[2].grid(True, color="0.9")
    axes[2].legend()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Interactively plot range/Doppler points extracted from saved head-echo ROIs.")
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--snr-threshold-db", type=float, default=5.0)
    p.add_argument("--no-reject-outliers", action="store_true")
    p.add_argument("--delay-sigma", type=float, default=4.0)
    p.add_argument("--doppler-sigma", type=float, default=4.0)
    args = p.parse_args()
    points, rejected = load_points(
        args.annotations,
        args.snr_threshold_db,
        reject=not args.no_reject_outliers,
        delay_sigma=args.delay_sigma,
        doppler_sigma=args.doppler_sigma,
    )
    print(f"Kept {len(points)} ROI peak points; rejected {len(rejected)} outliers")
    for station in sorted(set([p["station"] for p in points] + [p["station"] for p in rejected])):
        ps = [p for p in points if p["station"] == station]
        rs = [p for p in rejected if p["station"] == station]
        if not ps:
            print(f"{station}: 0 kept, {len(rs)} rejected")
            continue
        print(
            f"{station}: {len(ps)} kept, {len(rs)} rejected, "
            f"range {min(p['range_km'] for p in ps):.1f}-{max(p['range_km'] for p in ps):.1f} km, "
            f"doppler {min(p['doppler_hz'] for p in ps):.1f}-{max(p['doppler_hz'] for p in ps):.1f} Hz"
        )
    plot_points(points)


if __name__ == "__main__":
    main()
