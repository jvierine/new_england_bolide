#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pyorb
from astropy.time import Time

import dasst


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AU = 1.495978707e11
JUPITER_A_AU = 5.203
OMEGA_EARTH = 7.2921150e-5
ORBIT_H5 = ROOT / "products" / "orbit_determination" / "best_fit_orbit.h5"
BALLISTIC_FIT_JSON = ROOT / "products" / "ballistic_head_trail_correlated_appendix_fit" / "ballistic_head_trail_correlated_fit.json"
NEA_CSV = ROOT / "products" / "orbit_determination" / "nea_jopek_matches.csv"
COMET_CSV = ROOT / "products" / "orbit_determination" / "comet_jopek_matches.csv"
SOURCE_CSV = ROOT / "products" / "orbit_determination" / "meteor_source_matches.csv"
OUT = ROOT / "reports" / "orbit_source_table.tex"
PARAM_OUT = ROOT / "reports" / "orbit_parameter_table.tex"
MACRO_OUT = ROOT / "reports" / "orbit_analysis_macros.tex"
MAX_CROSSING_100KM_SAMPLES = 500
COMPUTE_CROSSING_100KM_UNCERTAINTY = False


def trajectory_version_comments(path: Path) -> list[str]:
    version, description = trajectory_version(path)
    return [
        f"% trajectory_fit_version: {version}\n",
        f"% trajectory_fit_version_description: {description}\n",
    ]


def trajectory_version(path: Path) -> tuple[str, str]:
    version = "unspecified"
    description = "unspecified"
    try:
        with h5py.File(path, "r") as h:
            version = str(h.attrs.get("trajectory_fit_version", version))
            description = str(h.attrs.get("trajectory_fit_version_description", description))
    except Exception:
        pass
    return version, description


def read_csv_rows(path: Path, n: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))[:n]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def circular_mean_deg(values_deg):
    values = np.deg2rad(np.asarray(values_deg, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values)))) % 360.0)


def angular_std_deg(values_deg: np.ndarray, mean_deg: float) -> float:
    values = np.asarray(values_deg, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    diff = (values - mean_deg + 180.0) % 360.0 - 180.0
    return float(np.nanstd(diff, ddof=1))


def corrected_radiant(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as h:
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
    finite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    return circular_mean_deg(radiant[0, finite]), float(np.nanmean(radiant[1, finite]))


def decimals_for_sigma(sigma: float, min_decimals: int = 0) -> int:
    if not np.isfinite(sigma) or sigma <= 0.0:
        return min_decimals
    exponent = int(math.floor(math.log10(abs(sigma))))
    first_digit = abs(sigma) / (10.0**exponent)
    sig_digits = 2 if first_digit < 3.0 else 1
    return max(min_decimals, sig_digits - 1 - exponent)


def format_pm(mean: float, sigma: float, min_decimals: int = 0) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sigma):
        return fmt(mean, min_decimals)
    digits = decimals_for_sigma(sigma, min_decimals=min_decimals)
    return f"{fmt(mean, digits)} $\\pm$ {fmt(sigma, digits)}"


def latex_text(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(ch, ch) for ch in value.strip())


def latex_macro(name: str, value: str | int | float) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def geocentric_speed_from_kepler(kep: np.ndarray, t_ref_unix: float) -> np.ndarray:
    epoch = Time(t_ref_unix, scale="utc", format="unix")
    out = np.full(kep.shape[1], np.nan, dtype=float)
    orb = pyorb.Orbit(M0=pyorb.M_sol, degrees=True, type="true", num=1)
    for idx in range(kep.shape[1]):
        row = kep[:, idx]
        if not np.all(np.isfinite(row)):
            continue
        try:
            orb.kepler = row[:, None]
            state = dasst.frames.convert(epoch, orb.cartesian, "HeliocentricMeanEcliptic", "GeocentricMeanEcliptic")
            out[idx] = np.linalg.norm(state[3:6, 0]) / 1000.0
        except Exception:
            out[idx] = np.nan
    return out


def eci_velocity_to_ecef(pos_gcrs: np.ndarray, vel_gcrs: np.ndarray, t_unix: float) -> np.ndarray:
    from fit_combined_head_trail_eci import gcrs_to_ecef

    pos_ecef = gcrs_to_ecef(pos_gcrs, t_unix)
    vel_rotated = gcrs_to_ecef(vel_gcrs, t_unix)
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    return vel_rotated - np.cross(omega, pos_ecef)


def crossing_100km_samples(path: Path) -> dict[str, np.ndarray]:
    from angle_departure_estimator import enu_to_ecef_matrix
    from fit_ballistic_head_trail import (
        build_density_profile,
        geodetic_path_from_states,
        gcrs_to_geodetic,
        interpolate_crossing_time,
        propagate_state,
        state_at_time,
    )

    if not BALLISTIC_FIT_JSON.exists():
        return {}
    with h5py.File(path, "r") as h:
        samples = np.asarray(h["sample_state_input"][()], dtype=float)
        mean_state = np.asarray(h["fit_mean_state"][()], dtype=float)
        t_ref_unix = float(h.attrs["t_ref_unix"])
    if not COMPUTE_CROSSING_100KM_UNCERTAINTY:
        samples = mean_state[None, :]
    elif samples.shape[0] > MAX_CROSSING_100KM_SAMPLES:
        idx = np.linspace(0, samples.shape[0] - 1, MAX_CROSSING_100KM_SAMPLES, dtype=int)
        samples = samples[idx]
    payload = json.loads(BALLISTIC_FIT_JSON.read_text(encoding="utf-8"))
    log10_cd_area_mass = float(payload["fit_parameters"][6])
    no_drag = bool(payload.get("settings", {}).get("no_drag", False))
    mean_params = np.r_[mean_state, log10_cd_area_mass]
    density_profile = build_density_profile(
        np.array([t_ref_unix, t_ref_unix + 30.0], dtype=float),
        [],
        mean_params,
        t_ref_unix,
    )

    times = []
    lats = []
    lons = []
    azs = []
    els = []
    search_t = np.arange(t_ref_unix, t_ref_unix + 30.0 + 0.05, 0.05)
    for sample in samples:
        params = np.r_[sample, log10_cd_area_mass]
        try:
            pos_grid, _vel_grid = propagate_state(params, search_t, t_ref_unix, density_profile, use_drag=not no_drag, integration_dt_s=0.05)
            llh_grid = geodetic_path_from_states(pos_grid, search_t)
            t_cross = interpolate_crossing_time(search_t, llh_grid[:, 2], 100000.0)
            if t_cross is None:
                continue
            pos, vel = state_at_time(params, t_cross, t_ref_unix, density_profile, use_drag=not no_drag, integration_dt_s=0.025)
            lat, lon, _alt = gcrs_to_geodetic(pos, t_cross)
            vel_ecef = eci_velocity_to_ecef(pos, vel, t_cross)
            rot = enu_to_ecef_matrix(lat, lon)
            velocity_enu = vel_ecef @ rot
            velocity_enu /= np.linalg.norm(velocity_enu)
            east, north, up = velocity_enu
            az = math.degrees(math.atan2(east, north)) % 360.0
            el = math.degrees(math.asin(np.clip(up, -1.0, 1.0)))
        except Exception:
            continue
        times.append(t_cross)
        lats.append(lat)
        lons.append(lon)
        azs.append(az)
        els.append(el)

    return {
        "time_unix": np.asarray(times, dtype=float),
        "lat_deg": np.asarray(lats, dtype=float),
        "lon_deg": np.asarray(lons, dtype=float),
        "az_deg": np.asarray(azs, dtype=float),
        "el_deg": np.asarray(els, dtype=float),
    }


def format_utc_pm(times_unix: np.ndarray) -> str:
    times = np.asarray(times_unix, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return "--"
    mean = float(np.nanmean(times))
    text = datetime.fromtimestamp(mean, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if times.size < 2:
        return f"{text} UTC"
    sigma = float(np.nanstd(times, ddof=1))
    return f"{text} UTC $\\pm$ {fmt(sigma, decimals_for_sigma(sigma, 2))} s"


def write_orbit_parameter_table(path: Path) -> None:
    with h5py.File(path, "r") as h:
        kep = np.asarray(h["kepler_HeliocentricMeanEcliptic_final"][()], dtype=float)
        radiant = np.asarray(h["radiant_orbit_GCRS"][()], dtype=float)
        t_ref_unix = float(h.attrs["t_ref_unix"])
    finite = np.all(np.isfinite(kep), axis=0)
    kep = kep[:, finite]
    a_au = kep[0] / AU
    e = kep[1]
    q_au = a_au * (1.0 - e)
    q_aphelion_au = a_au * (1.0 + e)
    tj = JUPITER_A_AU / a_au + 2.0 * np.cos(np.deg2rad(kep[2])) * np.sqrt((a_au / JUPITER_A_AU) * (1.0 - e * e))
    vg_kmps = geocentric_speed_from_kepler(kep, t_ref_unix)

    quantities = [
        ("Semimajor axis", "$a$", "AU", a_au, False, 2),
        ("Perihelion distance", "$q$", "AU", q_au, False, 3),
        ("Aphelion distance", "$Q$", "AU", q_aphelion_au, False, 2),
        ("Eccentricity", "$e$", "", e, False, 3),
        ("Inclination", "$i$", "deg", kep[2], True, 1),
        ("Argument of perihelion", "$\\omega$", "deg", kep[3], True, 1),
        ("Longitude of ascending node", "$\\Omega$", "deg", kep[4], True, 2),
        ("True anomaly at epoch", "$\\nu$", "deg", kep[5], True, 1),
        ("Geocentric speed", "$v_g$", "km s$^{-1}$", vg_kmps, False, 2),
        ("Tisserand parameter with respect to Jupiter", "$T_J$", "", tj, False, 2),
    ]

    rfinite = np.isfinite(radiant[0]) & np.isfinite(radiant[1])
    ra = radiant[0, rfinite]
    dec = radiant[1, rfinite]
    quantities.extend(
        [
            ("Corrected radiant right ascension", "RA", "deg", ra, True, 2),
            ("Corrected radiant declination", "Dec", "deg", dec, False, 1),
        ]
    )
    crossing = crossing_100km_samples(path)
    if crossing:
        lon_signed = (crossing["lon_deg"] + 180.0) % 360.0 - 180.0
        quantities.extend(
            [
                ("Entry latitude at 100 km", "$\\phi_{100}$", "deg", crossing["lat_deg"], False, 2),
                ("Entry longitude at 100 km", "$\\lambda_{100}$", "deg", lon_signed, False, 2),
                ("Entry azimuth at 100 km", "$A_{100}$", "deg", crossing["az_deg"], True, 1),
                ("Entry elevation at 100 km", "$E_{100}$", "deg", crossing["el_deg"], False, 1),
            ]
        )

    version, _description = trajectory_version(path)
    version_label = (
        f"Estimate {{\\footnotesize (v{version})}}"
        if version and version != "unspecified"
        else "Estimate"
    )
    lines = [
        *trajectory_version_comments(path),
        "\\begin{tabular}{llrl}\n",
        "\\toprule\n",
        f"Quantity & Symbol & Unit & {version_label} \\\\\n",
        "\\midrule\n",
    ]
    for name, symbol, unit, values, is_angle, digits in quantities:
        values = np.asarray(values, dtype=float)
        if is_angle:
            mean = circular_mean_deg(values)
            sigma = angular_std_deg(values, mean)
        else:
            mean = float(np.nanmean(values))
            finite_values = values[np.isfinite(values)]
            sigma = float(np.nanstd(finite_values, ddof=1)) if finite_values.size > 1 else math.nan
        lines.append(f"{name} & {symbol} & {unit} & {format_pm(mean, sigma, digits)} \\\\\n")
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    PARAM_OUT.write_text("".join(lines), encoding="utf-8")


def orbit_parameter_summary(path: Path) -> dict[str, float | int | str]:
    with h5py.File(path, "r") as h:
        kep = np.asarray(h["kepler_HeliocentricMeanEcliptic_final"][()], dtype=float)
        n_samples = int(h.attrs["n_samples"])
        version = str(h.attrs.get("trajectory_fit_version", "unspecified"))
        t_ref_unix = float(h.attrs["t_ref_unix"])
    finite = np.all(np.isfinite(kep), axis=0)
    kep = kep[:, finite]
    a_au = kep[0] / AU
    e = kep[1]
    q_au = a_au * (1.0 - e)
    q_aphelion_au = a_au * (1.0 + e)
    tj = JUPITER_A_AU / a_au + 2.0 * np.cos(np.deg2rad(kep[2])) * np.sqrt((a_au / JUPITER_A_AU) * (1.0 - e * e))
    inc_mean = circular_mean_deg(kep[2])
    omega_mean = circular_mean_deg(kep[3])
    node_mean = circular_mean_deg(kep[4])
    true_anomaly_mean = circular_mean_deg(kep[5])
    return {
        "n_samples": n_samples,
        "version": version,
        "epoch_utc": datetime.fromtimestamp(t_ref_unix, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "epoch_mjd_utc": float(Time(t_ref_unix, scale="utc", format="unix").mjd),
        "a_mean": float(np.nanmean(a_au)),
        "a_sigma": float(np.nanstd(a_au, ddof=1)),
        "q_mean": float(np.nanmean(q_au)),
        "q_sigma": float(np.nanstd(q_au, ddof=1)),
        "aphelion_mean": float(np.nanmean(q_aphelion_au)),
        "aphelion_sigma": float(np.nanstd(q_aphelion_au, ddof=1)),
        "e_mean": float(np.nanmean(e)),
        "e_sigma": float(np.nanstd(e, ddof=1)),
        "tj_mean": float(np.nanmean(tj)),
        "tj_sigma": float(np.nanstd(tj, ddof=1)),
        "inc_mean": inc_mean,
        "inc_sigma": angular_std_deg(kep[2], inc_mean),
        "omega_mean": omega_mean,
        "omega_sigma": angular_std_deg(kep[3], omega_mean),
        "node_mean": node_mean,
        "node_sigma": angular_std_deg(kep[4], node_mean),
        "true_anomaly_mean": true_anomaly_mean,
        "true_anomaly_sigma": angular_std_deg(kep[5], true_anomaly_mean),
    }


def first_csv_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path, 1)
    return rows[0] if rows else {}


def write_orbit_analysis_macros(path: Path) -> None:
    summary = orbit_parameter_summary(path)
    nea = first_csv_row(NEA_CSV)
    comet = first_csv_row(COMET_CSV)
    stream = first_csv_row(SOURCE_CSV)

    lines = [
        *trajectory_version_comments(path),
        "% Generated by reports/generate_orbit_source_table.py; do not edit by hand.\n",
        latex_macro("OrbitTrajectoryFitVersion", latex_text(str(summary["version"]))),
        latex_macro("OrbitSampleCount", str(summary["n_samples"])),
        latex_macro("OrbitEpochUTC", latex_text(str(summary["epoch_utc"]))),
        latex_macro("OrbitEpochMjdUTC", fmt(float(summary["epoch_mjd_utc"]), 8)),
        latex_macro("OrbitTisserandMean", fmt(float(summary["tj_mean"]), 2)),
        latex_macro("OrbitTisserandSigma", fmt(float(summary["tj_sigma"]), 2)),
        latex_macro("OrbitSemiMajorMeanAU", fmt(float(summary["a_mean"]), 2)),
        latex_macro("OrbitSemiMajorSigmaAU", fmt(float(summary["a_sigma"]), 2)),
        latex_macro("OrbitPerihelionMeanAU", fmt(float(summary["q_mean"]), 3)),
        latex_macro("OrbitPerihelionSigmaAU", fmt(float(summary["q_sigma"]), 3)),
        latex_macro("OrbitAphelionMeanAU", fmt(float(summary["aphelion_mean"]), 2)),
        latex_macro("OrbitAphelionSigmaAU", fmt(float(summary["aphelion_sigma"]), 2)),
        latex_macro("OrbitEccentricityMean", fmt(float(summary["e_mean"]), 3)),
        latex_macro("OrbitEccentricitySigma", fmt(float(summary["e_sigma"]), 3)),
        latex_macro("OrbitInclinationMeanDeg", fmt(float(summary["inc_mean"]), 1)),
        latex_macro("OrbitInclinationSigmaDeg", fmt(float(summary["inc_sigma"]), 1)),
        latex_macro("OrbitArgPerihelionMeanDeg", fmt(float(summary["omega_mean"]), 1)),
        latex_macro("OrbitArgPerihelionSigmaDeg", fmt(float(summary["omega_sigma"]), 1)),
        latex_macro("OrbitNodeMeanDeg", fmt(float(summary["node_mean"]), 4)),
        latex_macro("OrbitNodeSigmaDeg", fmt(float(summary["node_sigma"]), 4)),
        latex_macro("OrbitTrueAnomalyMeanDeg", fmt(float(summary["true_anomaly_mean"]), 1)),
        latex_macro("OrbitTrueAnomalySigmaDeg", fmt(float(summary["true_anomaly_sigma"]), 1)),
    ]
    if nea:
        lines.extend(
            [
                latex_macro("OrbitTopNeaName", latex_text(nea["name"])),
                latex_macro("OrbitTopNeaDH", fmt(float(nea["D_H"]), 3)),
            ]
        )
    if comet:
        lines.extend(
            [
                latex_macro("OrbitTopCometName", latex_text(comet["name"] or comet["designation"])),
                latex_macro("OrbitTopCometDH", fmt(float(comet["D_H"]), 3)),
            ]
        )
    if stream:
        lines.extend(
            [
                latex_macro("OrbitTopStreamName", latex_text(stream["name"])),
                latex_macro("OrbitTopStreamDH", fmt(float(stream["D_H"]), 3)),
                latex_macro("OrbitTopStreamRadiantSepDeg", fmt(float(stream["radiant_sep_deg"]), 0)),
            ]
        )
    MACRO_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    with h5py.File(ORBIT_H5, "r") as h:
        if "kepler_HeliocentricMeanEcliptic_final_mean" in h:
            kep = h["kepler_HeliocentricMeanEcliptic_final_mean"][()]
        else:
            kep = h["kepler_HeliocentricMeanEcliptic_final"][:, 0]
        kep_samples = np.asarray(h["kepler_HeliocentricMeanEcliptic_final"][()], dtype=float)
    ra_deg, dec_deg = corrected_radiant(ORBIT_H5)
    finite = np.all(np.isfinite(kep_samples), axis=0)
    if np.any(finite):
        kep_finite = kep_samples[:, finite]
        a_au = float(np.nanmean(kep_finite[0] / AU))
        e = float(np.nanmean(kep_finite[1]))
        q_au = float(np.nanmean((kep_finite[0] / AU) * (1.0 - kep_finite[1])))
        inc = circular_mean_deg(kep_finite[2])
        omega = circular_mean_deg(kep_finite[3])
        node = circular_mean_deg(kep_finite[4])
    else:
        a_au = float(kep[0] / AU)
        e = float(kep[1])
        q_au = a_au * (1.0 - e)
        inc = float(kep[2])
        omega = float(kep[3])
        node = float(kep[4])
    orbit_rows = [
        ("Bolide orbit sample mean", a_au, q_au, e, inc, omega, node, ra_deg, dec_deg, "", "", ""),
    ]

    nea_rows = []
    for row in read_csv_rows(NEA_CSV, 5):
        nea_rows.append(
            (
                row["name"],
                float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float("nan"),
                float("nan"),
                float(row["D_H"]),
                float("nan"),
                "NEA",
            )
        )

    comet_rows = []
    for row in read_csv_rows(COMET_CSV, 5):
        comet_rows.append(
            (
                row["name"] or row["designation"],
                float("nan") if row["a_au"] == "nan" else float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float("nan"),
                float("nan"),
                float(row["D_H"]),
                float("nan"),
                "comet",
            )
        )

    source_rows = []
    for row in read_csv_rows(SOURCE_CSV, 6):
        source_rows.append(
            (
                f"{row['code']} {row['name']}",
                float("nan") if row["a_au"] == "nan" else float(row["a_au"]),
                float(row["q_au"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["omega_deg"]),
                float(row["node_deg"]),
                float(row["ra_deg"]),
                float(row["dec_deg"]),
                float(row["D_H"]),
                float(row["radiant_sep_deg"]),
                "stream",
            )
        )

    lines = [
        *trajectory_version_comments(ORBIT_H5),
        "\\begin{tabular}{lrrrrrrrrrrl}\n",
        "\\toprule\n",
        "Object/source & $a$ & $q$ & $e$ & $i$ & $\\omega$ & $\\Omega$ & RA & Dec & $D_H$ & $\\Delta_R$ & Type \\\\\n",
        " & (AU) & (AU) & & (deg) & (deg) & (deg) & (deg) & (deg) & & (deg) & \\\\\n",
        "\\midrule\n",
    ]
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in orbit_rows:
        lines.append(
            f"{name} & {fmt(a)} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & {fmt(ra, 2)} & {fmt(dec, 2)} & -- & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in nea_rows:
        lines.append(
            f"{name} & {fmt(a)} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & -- & -- & {fmt(dh, 3)} & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in comet_rows:
        a_text = "--" if a != a else fmt(a)
        lines.append(
            f"{name} & {a_text} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & -- & -- & {fmt(dh, 3)} & -- & {typ} \\\\\n"
        )
    lines.append("\\midrule\n")
    for name, a, q, ecc, inc, omega, node, ra, dec, dh, d_rad, typ in source_rows:
        a_text = "--" if a != a else fmt(a)
        lines.append(
            f"{name} & {a_text} & {fmt(q)} & {fmt(ecc)} & {fmt(inc, 2)} & "
            f"{fmt(omega, 2)} & {fmt(node, 2)} & {fmt(ra, 1)} & {fmt(dec, 1)} & "
            f"{fmt(dh, 3)} & {fmt(d_rad, 1)} & {typ} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    OUT.write_text("".join(lines), encoding="utf-8")
    write_orbit_parameter_table(ORBIT_H5)
    write_orbit_analysis_macros(ORBIT_H5)
    print(OUT)
    print(PARAM_OUT)
    print(MACRO_OUT)


if __name__ == "__main__":
    main()
