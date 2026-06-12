#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_TEX = ROOT / "reports" / "head_echo_rcs_summary.tex"

from angle_departure_estimator import load_station_yaml  # noqa: E402
DEFAULT_PATCH_H5 = (
    ROOT
    / "products"
    / "head_echo_gaussian_patches"
    / "head_echo_gaussian_patches_v0p2p7_1coh_2incoh_zp16_sparse_dcmean_retained_ext0p5.h5"
)

STATION_RANGES_KM = [
    ("beverly", 180.0, 275.0),
    ("clinton", 220.0, 280.0),
    ("pomfret", 350.0, 420.0),
    ("siena", 400.0, 500.0),
    ("amherstnh", 200.0, 350.0),
]
STATION_LABELS = {
    "amherstnh": "Amherst",
    "beverly": "Beverly",
    "clinton": "Clinton",
    "pomfret": "Pomfret",
    "siena": "Siena",
}
HEAD_PLOT_START_UNIX = 1780164379.0
HEAD_PLOT_END_UNIX = 1780164382.0
C_MPS = 299_792_458.0
K_BOLTZMANN = 1.380649e-23
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563


def unix_to_mpl(t: np.ndarray) -> np.ndarray:
    ns = np.asarray(np.asarray(t, dtype=float) * 1e9, dtype="datetime64[ns]")
    return mdates.date2num(ns.astype("datetime64[us]").astype(object))


def bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.r_[first, mids, last]


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_M / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * np.sin(lat)
    return np.column_stack((x, y, z))


def target_ecef_from_fit(fit_json: Path, times_unix: np.ndarray) -> np.ndarray:
    fit = json.loads(fit_json.read_text(encoding="utf-8"))
    model = fit.get("model", {})
    if not all(k in model for k in ("time_unix", "lat_deg", "lon_deg", "alt_m")):
        raise ValueError(f"{fit_json} does not contain model time_unix/lat_deg/lon_deg/alt_m")
    tm = np.asarray(model["time_unix"], dtype=float)
    order = np.argsort(tm)
    tm = tm[order]
    lat = np.interp(times_unix, tm, np.asarray(model["lat_deg"], dtype=float)[order])
    lon = np.interp(times_unix, tm, np.asarray(model["lon_deg"], dtype=float)[order])
    alt = np.interp(times_unix, tm, np.asarray(model["alt_m"], dtype=float)[order])
    return geodetic_to_ecef(lat, lon, alt)


def vertical_common_mode_notch(img_db: np.ndarray) -> np.ndarray:
    """Remove range-common vertical striping with a narrow 2-D FFT notch."""
    valid = np.isfinite(img_db)
    if not np.any(valid):
        return img_db
    fill = float(np.nanmedian(img_db))
    work = np.where(valid, img_db, fill)
    mean = float(np.mean(work))
    fft = np.fft.fftshift(np.fft.fft2(work - mean))
    ky0 = fft.shape[0] // 2
    kx0 = fft.shape[1] // 2
    fft[ky0, :] = 0.0
    fft[ky0, kx0] = np.sum(work - mean)
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fft))) + mean
    out = img_db.copy()
    out[valid] = filtered[valid]
    return out


def load_station_rcs(
    station: str,
    rmin_km: float,
    rmax_km: float,
    merged_dir: Path,
    fit_json: Path,
    pt_w: float,
    tx_gain: float,
    rx_gain: float,
    system_temp_k: float,
    doppler_min_hz: float,
    apply_vertical_notch: bool,
) -> tuple[dict, dict]:
    path = merged_dir / f"rdmf_peak_{station}_merged.h5"
    with h5py.File(path, "r") as h:
        p = h["P_head"][()] if "P_head" in h else h["P"][()]
        d = h["D_head"][()] if "D_head" in h else h["D"][()]
        n = h["N"][()]
        t = h["t_sample"][()] / float(h["sample_rate"][()])
        ranges = h["range_km"][()] if "range_km" in h else h["range_idx"][()] * 3.0
        gps_tx = h["gps_tx"][()]
        gps_rx = h["gps_rx"][()]
        freq_hz = float(h["frequency_hz"][()])
        sample_rate = float(h["sample_rate"][()])
        code_len = int(h["code_len"][()])
        n_coh = int(h["n_coh"][()])
        product_has_head = "P_head" in h
        product_name = path.name

    range_keep = (ranges >= rmin_km) & (ranges <= rmax_km)
    time_keep = (t >= HEAD_PLOT_START_UNIX) & (t <= HEAD_PLOT_END_UNIX)
    p_sel = p[time_keep][:, range_keep]
    n_sel = np.maximum(n[time_keep][:, range_keep], 1e-30)
    d_sel = d[time_keep][:, range_keep]
    t_sel = t[time_keep]

    snr_linear = np.maximum(p_sel / n_sel, 1e-30)
    snr_db = 10.0 * np.log10(snr_linear)
    snr_db[np.abs(d_sel) < doppler_min_hz] = np.nan
    if apply_vertical_notch:
        snr_db = vertical_common_mode_notch(snr_db)
        snr_linear = 10.0 ** (snr_db / 10.0)

    tx_ecef = geodetic_to_ecef(np.array([gps_tx[0]]), np.array([gps_tx[1]]), np.array([gps_tx[2]]))[0]
    rx_ecef = geodetic_to_ecef(np.array([gps_rx[0]]), np.array([gps_rx[1]]), np.array([gps_rx[2]]))[0]
    target_ecef = target_ecef_from_fit(fit_json, t_sel)
    r_tx_m = np.linalg.norm(target_ecef - tx_ecef[None, :], axis=1)
    r_rx_m = np.linalg.norm(target_ecef - rx_ecef[None, :], axis=1)

    wavelength_m = C_MPS / freq_hz
    noise_bandwidth_hz = sample_rate / (code_len * n_coh)
    noise_power_w = K_BOLTZMANN * system_temp_k * noise_bandwidth_hz
    geom = (4.0 * np.pi) ** 3 * (r_tx_m[:, None] ** 2) * (r_rx_m[:, None] ** 2)
    sigma_m2 = snr_linear * noise_power_w * geom / (pt_w * tx_gain * rx_gain * wavelength_m**2)
    rcs_dbsm = 10.0 * np.log10(np.maximum(sigma_m2, 1e-60))
    rcs_dbsm[~np.isfinite(snr_db)] = np.nan

    data = {
        "station": station,
        "t": t_sel,
        "range_km": ranges[range_keep],
        "rcs_dbsm": rcs_dbsm,
        "rmin_km": rmin_km,
        "rmax_km": rmax_km,
    }
    meta = {
        "station": station,
        "product": product_name,
        "product_has_head_dataset": product_has_head,
        "frequency_hz": freq_hz,
        "code_len": code_len,
        "n_coh": n_coh,
        "noise_bandwidth_hz": noise_bandwidth_hz,
    }
    return data, meta


def patch_peak_snr_by_range(raw_db: np.ndarray) -> np.ndarray:
    raw_power = 10.0 ** (np.asarray(raw_db, dtype=float) / 10.0)
    peak_power = np.nanmax(raw_power, axis=1)
    noise_power = np.nanmedian(raw_power, axis=1)
    return np.maximum(peak_power / np.maximum(noise_power, 1e-300), 1e-30)


def load_station_rcs_from_patches(
    station: str,
    patch_h5: Path,
    data_dir: Path,
    fit_json: Path,
    pt_w: float,
    tx_gain: float,
    rx_gain: float,
    system_temp_k: float,
    noise_bandwidth_hz: float,
) -> tuple[dict, dict]:
    site = load_station_yaml(data_dir, station)
    with h5py.File(patch_h5, "r") as h:
        if station not in h:
            raise ValueError(f"{patch_h5} does not contain station {station!r}")
        station_group = h[station]
        patch_names = sorted(station_group.keys())
        times = np.array([float(station_group[name].attrs["time_unix"]) for name in patch_names], dtype=float)
        ranges_m = np.unique(
            np.concatenate([np.rint(np.asarray(station_group[name]["range_m"][()], dtype=float)).astype(np.int64) for name in patch_names])
        )
        ranges_km = ranges_m.astype(float) / 1000.0
        snr_linear = np.full((len(patch_names), len(ranges_m)), np.nan, dtype=float)
        for ti, name in enumerate(patch_names):
            group = station_group[name]
            patch_ranges = np.rint(np.asarray(group["range_m"][()], dtype=float)).astype(np.int64)
            col = np.searchsorted(ranges_m, patch_ranges)
            snr_linear[ti, col] = patch_peak_snr_by_range(group["raw_db"][()])
        product_version = str(h.attrs.get("head_echo_product_version", "unspecified"))
        product_description = str(h.attrs.get("head_echo_product_version_description", ""))
        n_coh = int(h.attrs.get("n_coh", -1))
        n_incoh = int(h.attrs.get("n_incoh", -1))
        doppler_zero_pad_factor = int(h.attrs.get("doppler_zero_pad_factor", -1))
        sparse_voltage_subtract = bool(h.attrs.get("sparse_voltage_subtract", False))
        dc_subtract = str(h.attrs.get("dc_subtract", ""))

    tx_ecef = geodetic_to_ecef(np.array([site["gps_tx"][0]]), np.array([site["gps_tx"][1]]), np.array([site["gps_tx"][2]]))[0]
    rx_ecef = geodetic_to_ecef(np.array([site["gps_rx"][0]]), np.array([site["gps_rx"][1]]), np.array([site["gps_rx"][2]]))[0]
    target_ecef = target_ecef_from_fit(fit_json, times)
    r_tx_m = np.linalg.norm(target_ecef - tx_ecef[None, :], axis=1)
    r_rx_m = np.linalg.norm(target_ecef - rx_ecef[None, :], axis=1)

    wavelength_m = C_MPS / float(site["frequency_hz"])
    noise_power_w = K_BOLTZMANN * system_temp_k * noise_bandwidth_hz
    geom = (4.0 * np.pi) ** 3 * (r_tx_m[:, None] ** 2) * (r_rx_m[:, None] ** 2)
    sigma_m2 = snr_linear * noise_power_w * geom / (pt_w * tx_gain * rx_gain * wavelength_m**2)
    rcs_dbsm = 10.0 * np.log10(np.maximum(sigma_m2, 1e-60))
    rcs_dbsm[~np.isfinite(snr_linear)] = np.nan

    finite_ranges = ranges_km[np.any(np.isfinite(rcs_dbsm), axis=0)]
    if finite_ranges.size:
        rmin_km = float(np.nanmin(finite_ranges))
        rmax_km = float(np.nanmax(finite_ranges))
    else:
        rmin_km, rmax_km = float(ranges_km[0]), float(ranges_km[-1])
    pad_km = max(1.0, 0.025 * (rmax_km - rmin_km))
    data = {
        "station": station,
        "t": times,
        "range_km": ranges_km,
        "rcs_dbsm": rcs_dbsm,
        "rmin_km": rmin_km - pad_km,
        "rmax_km": rmax_km + pad_km,
    }
    meta = {
        "station": station,
        "product": patch_h5.name,
        "product_kind": "zp16_head_echo_patch_cache",
        "product_version": product_version,
        "product_description": product_description,
        "frequency_hz": float(site["frequency_hz"]),
        "code_len": int(site["code_len"]),
        "n_coh": n_coh,
        "n_incoh": n_incoh,
        "doppler_zero_pad_factor": doppler_zero_pad_factor,
        "noise_bandwidth_hz": float(noise_bandwidth_hz),
        "snr_definition": "per-range peak(raw unfiltered RDMF power) / median(raw unfiltered RDMF power over Doppler)",
        "sparse_voltage_subtract": sparse_voltage_subtract,
        "dc_subtract": dc_subtract,
        "n_patches": len(patch_names),
    }
    return data, meta


def plot_rcs(datasets: list[dict], metas: list[dict], out: Path, version_label: str) -> Path:
    vals = np.concatenate([d["rcs_dbsm"][np.isfinite(d["rcs_dbsm"])] for d in datasets])
    vmin = float(np.nanpercentile(vals, 2.0)) if vals.size else -40.0
    vmax = float(np.nanpercentile(vals, 99.5)) if vals.size else 20.0
    vmin = max(-50.0, vmin)
    vmax = min(35.0, max(vmin + 10.0, vmax))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.90")
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12.0, 10.0), sharex=True, constrained_layout=True)
    mesh = None
    for ax, data in zip(axes, datasets, strict=True):
        mesh = ax.pcolormesh(
            bin_edges(unix_to_mpl(data["t"])),
            bin_edges(data["range_km"]),
            data["rcs_dbsm"].T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel(f"{STATION_LABELS.get(data['station'], data['station'])}\nrange (km)")
        ax.set_ylim(data["rmin_km"], data["rmax_km"])
        ax.grid(True, color="0.9", linewidth=0.4)
    for ax in axes:
        ax.set_xlim(unix_to_mpl(np.array([HEAD_PLOT_START_UNIX, HEAD_PLOT_END_UNIX])))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].text(
        0.012,
        0.94,
        f"{version_label}; B_N={metas[0]['noise_bandwidth_hz']:.1f} Hz",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
        alpha=0.65,
    )
    axes[-1].set_xlabel("UTC on 2026-05-30")
    fig.colorbar(mesh, ax=axes, label="Equivalent RCS (dBsm)", shrink=0.93)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize_rcs(datasets: list[dict]) -> dict:
    stations = []
    finite_all = []
    for data in datasets:
        vals = np.asarray(data["rcs_dbsm"], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            finite_all.append(vals)
            stats = {
                "n": int(vals.size),
                "min_dbsm": float(np.nanmin(vals)),
                "p05_dbsm": float(np.nanpercentile(vals, 5.0)),
                "median_dbsm": float(np.nanmedian(vals)),
                "p95_dbsm": float(np.nanpercentile(vals, 95.0)),
                "max_dbsm": float(np.nanmax(vals)),
            }
        else:
            stats = {
                "n": 0,
                "min_dbsm": float("nan"),
                "p05_dbsm": float("nan"),
                "median_dbsm": float("nan"),
                "p95_dbsm": float("nan"),
                "max_dbsm": float("nan"),
            }
        stations.append({"station": data["station"], **stats})

    if finite_all:
        vals = np.concatenate(finite_all)
        global_stats = {
            "n": int(vals.size),
            "min_dbsm": float(np.nanmin(vals)),
            "p05_dbsm": float(np.nanpercentile(vals, 5.0)),
            "median_dbsm": float(np.nanmedian(vals)),
            "p95_dbsm": float(np.nanpercentile(vals, 95.0)),
            "max_dbsm": float(np.nanmax(vals)),
        }
    else:
        global_stats = {
            "n": 0,
            "min_dbsm": float("nan"),
            "p05_dbsm": float("nan"),
            "median_dbsm": float("nan"),
            "p95_dbsm": float("nan"),
            "max_dbsm": float("nan"),
        }
    return {"global": global_stats, "stations": stations}


def tex_number(x: float, ndigits: int = 1) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def write_summary_tex(summary: dict, path: Path) -> None:
    def macro(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}\n"

    global_stats = summary["global"]
    lines = [
        "% Generated by reports/generate_station_head_rcs_figure.py\n",
        macro("HeadRcsNFinite", f"{global_stats['n']:,}"),
        macro("HeadRcsMinDbsm", tex_number(global_stats["min_dbsm"])),
        macro("HeadRcsFifthDbsm", tex_number(global_stats["p05_dbsm"])),
        macro("HeadRcsMedianDbsm", tex_number(global_stats["median_dbsm"])),
        macro("HeadRcsNinetyFifthDbsm", tex_number(global_stats["p95_dbsm"])),
        macro("HeadRcsMaxDbsm", tex_number(global_stats["max_dbsm"])),
    ]
    for station_stats in summary["stations"]:
        prefix = "".join(part.capitalize() for part in station_stats["station"].replace("_", " ").split())
        lines.extend(
            [
                macro(f"HeadRcs{prefix}NFinite", f"{station_stats['n']:,}"),
                macro(f"HeadRcs{prefix}MedianDbsm", tex_number(station_stats["median_dbsm"])),
                macro(f"HeadRcs{prefix}NinetyFifthDbsm", tex_number(station_stats["p95_dbsm"])),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the head-echo equivalent-RCS station RTI figure.")
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=ROOT / "products" / "rdmf_exact_highres_180615_10s_merged_20260602_095431",
    )
    parser.add_argument("--patch-h5", type=Path, default=DEFAULT_PATCH_H5)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "2026_05_30_bolide")
    parser.add_argument(
        "--fit-json",
        type=Path,
        default=ROOT
        / "products"
        / "ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5"
        / "ballistic_head_trail_correlated_fit.json",
    )
    parser.add_argument("--output", type=Path, default=FIG_DIR / "station_range_time_head_rcs.png")
    parser.add_argument("--pt-w", type=float, default=500.0)
    parser.add_argument("--tx-gain", type=float, default=1.0)
    parser.add_argument("--rx-gain", type=float, default=1.0)
    parser.add_argument("--system-temp-k", type=float, default=290.0)
    parser.add_argument("--noise-bandwidth-hz", type=float, default=100.0)
    parser.add_argument("--doppler-min-hz", type=float, default=250.0)
    parser.add_argument("--version-label", default="v0.2.7 zp16 n_coh=1")
    parser.add_argument("--summary-tex", type=Path, default=SUMMARY_TEX)
    parser.add_argument("--no-vertical-notch", action="store_true")
    parser.add_argument("--use-merged-rdmf", action="store_true")
    args = parser.parse_args()

    datasets = []
    metas = []
    if args.use_merged_rdmf:
        for station, rmin, rmax in STATION_RANGES_KM:
            data, meta = load_station_rcs(
                station,
                rmin,
                rmax,
                args.merged_dir,
                args.fit_json,
                args.pt_w,
                args.tx_gain,
                args.rx_gain,
                args.system_temp_k,
                args.doppler_min_hz,
                not args.no_vertical_notch,
            )
            datasets.append(data)
            metas.append(meta)
    else:
        with h5py.File(args.patch_h5, "r") as h:
            stations = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stations"][()]]
        for station in stations:
            data, meta = load_station_rcs_from_patches(
                station,
                args.patch_h5,
                args.data_dir,
                args.fit_json,
                args.pt_w,
                args.tx_gain,
                args.rx_gain,
                args.system_temp_k,
                args.noise_bandwidth_hz,
            )
            datasets.append(data)
            metas.append(meta)
    summary = summarize_rcs(datasets)
    out = plot_rcs(datasets, metas, args.output, args.version_label)
    write_summary_tex(summary, args.summary_tex)
    meta_out = out.with_suffix(".json")
    product_kind = "merged_rdmf" if args.use_merged_rdmf else "zp16_patch_cache"
    meta_out.write_text(
        json.dumps(
            {
                "figure": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
                "source": "reports/generate_station_head_rcs_figure.py",
                "product_kind": product_kind,
                "patch_h5": str(args.patch_h5.relative_to(ROOT)) if args.patch_h5.is_relative_to(ROOT) else str(args.patch_h5),
                "merged_dir_name": args.merged_dir.name,
                "fit_json_name": args.fit_json.name,
                "pt_w": args.pt_w,
                "tx_gain": args.tx_gain,
                "rx_gain": args.rx_gain,
                "system_temp_k": args.system_temp_k,
                "noise_bandwidth_hz": args.noise_bandwidth_hz,
                "doppler_min_hz": args.doppler_min_hz,
                "vertical_common_mode_2d_fft_notch": bool(args.use_merged_rdmf and not args.no_vertical_notch),
                "summary": summary,
                "stations": metas,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out)
    print(meta_out)
    print(args.summary_tex)


if __name__ == "__main__":
    main()
