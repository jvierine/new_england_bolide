#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import argparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import geodetic_to_ecef, enu_to_ecef_matrix, load_station_yaml, unit_vectors_enu
from aod_position_from_merged import aod_delay_position
from estimate_head_echo_trajectory import ecef_to_geodetic_wgs84, model_state_series

DATA_DIR = ROOT / "2026_05_30_bolide"
TRAJECTORY_JSON = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
TRAIL_SUMMARY_JSON = ROOT / "products" / "specular_trail_positions" / "specular_trail_position_summary.json"
STATIONS = ["amherstnh", "beverly", "clinton", "pomfret", "siena"]


def assert_close(name: str, value: float, limit: float) -> None:
    if not np.isfinite(value) or abs(value) > limit:
        raise AssertionError(f"{name}: {value:.6g} exceeds limit {limit:.6g}")


def angular_separation_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def check_wgs84_round_trip() -> dict[str, float]:
    cases = [
        (0.0, 0.0, 0.0),
        (0.0, 90.0, 1000.0),
        (42.0, -70.6, 0.0),
        (42.0, -70.6, 110_000.0),
        (-35.0, 149.0, 500.0),
    ]
    for station in STATIONS:
        site = load_station_yaml(DATA_DIR, station)
        cases.append(tuple(site["gps_tx"]))
        cases.append(tuple(site["gps_rx"]))

    max_lat_err = 0.0
    max_lon_err = 0.0
    max_alt_err = 0.0
    for lat, lon, alt in cases:
        lat2, lon2, alt2 = ecef_to_geodetic_wgs84(geodetic_to_ecef(lat, lon, alt))
        lon_err = ((lon2 - lon + 180.0) % 360.0) - 180.0
        max_lat_err = max(max_lat_err, abs(lat2 - lat))
        max_lon_err = max(max_lon_err, abs(lon_err))
        max_alt_err = max(max_alt_err, abs(alt2 - alt))

    assert_close("max geodetic latitude round-trip error deg", max_lat_err, 1e-9)
    assert_close("max geodetic longitude round-trip error deg", max_lon_err, 1e-9)
    assert_close("max geodetic altitude round-trip error m", max_alt_err, 1e-4)
    return {
        "max_lat_err_deg": max_lat_err,
        "max_lon_err_deg": max_lon_err,
        "max_alt_err_m": max_alt_err,
    }


def check_enu_basis() -> dict[str, float]:
    max_orth_err = 0.0
    max_det_err = 0.0
    max_cross_err = 0.0
    for station in STATIONS:
        site = load_station_yaml(DATA_DIR, station)
        for lat, lon, alt in (site["gps_tx"], site["gps_rx"]):
            rot = enu_to_ecef_matrix(lat, lon)
            max_orth_err = max(max_orth_err, float(np.max(np.abs(rot.T @ rot - np.eye(3)))))
            max_det_err = max(max_det_err, abs(float(np.linalg.det(rot)) - 1.0))
            max_cross_err = max(max_cross_err, float(np.linalg.norm(np.cross(rot[:, 0], rot[:, 1]) - rot[:, 2])))

            origin = geodetic_to_ecef(lat, lon, alt)
            lat_e, lon_e, alt_e = ecef_to_geodetic_wgs84(origin + rot @ np.array([1000.0, 0.0, 0.0]))
            lat_n, lon_n, alt_n = ecef_to_geodetic_wgs84(origin + rot @ np.array([0.0, 1000.0, 0.0]))
            lat_u, lon_u, alt_u = ecef_to_geodetic_wgs84(origin + rot @ np.array([0.0, 0.0, 1000.0]))
            if lon_e <= lon:
                raise AssertionError(f"{station}: +east did not increase longitude at {lat}, {lon}")
            if lat_n <= lat:
                raise AssertionError(f"{station}: +north did not increase latitude at {lat}, {lon}")
            if alt_u <= alt + 999.0:
                raise AssertionError(f"{station}: +up did not increase altitude by about 1 km")
            assert_close(f"{station}: east step altitude drift m", alt_e - alt, 0.1)
            assert_close(f"{station}: north step altitude drift m", alt_n - alt, 0.1)

    assert_close("max ENU basis orthonormality error", max_orth_err, 1e-14)
    assert_close("max ENU basis determinant error", max_det_err, 1e-14)
    assert_close("max ENU east x north - up error", max_cross_err, 1e-14)
    return {
        "max_orthonormality_error": max_orth_err,
        "max_determinant_error": max_det_err,
        "max_cross_product_error": max_cross_err,
    }


def check_az_el_units() -> dict[str, float]:
    cases = [
        (0.0, 0.0, np.array([0.0, 1.0, 0.0])),
        (90.0, 0.0, np.array([1.0, 0.0, 0.0])),
        (180.0, 0.0, np.array([0.0, -1.0, 0.0])),
        (270.0, 0.0, np.array([-1.0, 0.0, 0.0])),
        (0.0, 90.0, np.array([0.0, 0.0, 1.0])),
    ]
    max_err = 0.0
    for az, el, expected in cases:
        got = unit_vectors_enu(np.array([az]), np.array([el]))[0]
        max_err = max(max_err, float(np.linalg.norm(got - expected)))
    assert_close("max az/el basis unit error", max_err, 1e-14)
    return {"max_unit_vector_error": max_err}


def check_trajectory_product() -> dict[str, float]:
    fit = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8"))
    station = fit["used_stations"][0]
    tx_site = load_station_yaml(DATA_DIR, station)
    tx0 = geodetic_to_ecef(*tx_site["gps_tx"])
    rot = enu_to_ecef_matrix(tx_site["gps_tx"][0], tx_site["gps_tx"][1])
    params = np.r_[fit["fit_parameters_enu_at_tx"]["position_ref_m"], fit["fit_parameters_enu_at_tx"]["velocity_mps"]]

    ref_ecef = tx0 + rot @ params[:3]
    stored_ref_ecef = np.array(fit["position_ref_ecef_m"], dtype=float)
    ref_ecef_err = float(np.linalg.norm(ref_ecef - stored_ref_ecef))
    assert_close("trajectory stored ECEF ref error m", ref_ecef_err, 1e-6)

    lat, lon, alt = ecef_to_geodetic_wgs84(stored_ref_ecef)
    geo = fit["geodetic_ref_wgs84"]
    assert_close("trajectory ref latitude product error deg", lat - geo["lat_deg"], 1e-9)
    assert_close("trajectory ref longitude product error deg", lon - geo["lon_deg"], 1e-9)
    assert_close("trajectory ref altitude product error m", alt - geo["alt_m"], 1e-4)

    state = model_state_series(params, [fit["t_ref_unix"]], fit["t_ref_unix"], tx0, rot)
    state_ref_err = float(np.linalg.norm(state["position_ecef_m"][0] - stored_ref_ecef))
    assert_close("model_state_series t_ref ECEF error m", state_ref_err, 1e-6)

    ts = np.array(fit["model_time_series"]["time_unix"], dtype=float)
    product_lat = np.array(fit["model_time_series"]["lat_deg"], dtype=float)
    product_lon = np.array(fit["model_time_series"]["lon_deg"], dtype=float)
    product_alt = np.array(fit["model_time_series"]["alt_km"], dtype=float) * 1000.0
    series = model_state_series(params, ts, fit["t_ref_unix"], tx0, rot)["lat_lon_alt"]
    max_series_lat_err = float(np.max(np.abs(series[:, 0] - product_lat)))
    max_series_lon_err = float(np.max(np.abs(series[:, 1] - product_lon)))
    max_series_alt_err = float(np.max(np.abs(series[:, 2] - product_alt)))
    assert_close("trajectory model series latitude error deg", max_series_lat_err, 1e-9)
    assert_close("trajectory model series longitude error deg", max_series_lon_err, 1e-9)
    assert_close("trajectory model series altitude error m", max_series_alt_err, 1e-3)

    return {
        "ref_ecef_err_m": ref_ecef_err,
        "state_ref_err_m": state_ref_err,
        "max_series_lat_err_deg": max_series_lat_err,
        "max_series_lon_err_deg": max_series_lon_err,
        "max_series_alt_err_m": max_series_alt_err,
    }


def check_trail_aod_positions() -> dict[str, float]:
    payload = json.loads(TRAIL_SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = payload.get("positions", [])
    if not rows:
        raise AssertionError("No trail AoD positions to check")

    first_site = load_station_yaml(DATA_DIR, rows[0]["station"])
    tx_ecef = geodetic_to_ecef(*first_site["gps_tx"])
    rot = enu_to_ecef_matrix(first_site["gps_tx"][0], first_site["gps_tx"][1])

    max_geodetic_err_m = 0.0
    max_enu_err_m = 0.0
    max_path_resid_m = 0.0
    max_unit_angle_err_deg = 0.0
    max_resolve_pos_err_m = 0.0
    for row in rows:
        site = load_station_yaml(DATA_DIR, row["station"])
        rx_ecef = geodetic_to_ecef(*site["gps_rx"])
        pos_ecef = np.array(row["position_ecef_m"], dtype=float)
        pos_enu = rot.T @ (pos_ecef - tx_ecef)
        max_enu_err_m = max(max_enu_err_m, float(np.linalg.norm(pos_enu - np.array(row["position_enu_m"], dtype=float))))

        lat, lon, alt = ecef_to_geodetic_wgs84(pos_ecef)
        dlat_m = (lat - float(row["lat_deg"])) * 111_320.0
        dlon_m = (lon - float(row["lon_deg"])) * 111_320.0 * math.cos(math.radians(lat))
        dalt_m = alt - float(row["alt_m"])
        max_geodetic_err_m = max(max_geodetic_err_m, float(np.linalg.norm([dlat_m, dlon_m, dalt_m])))

        path = np.linalg.norm(pos_ecef - tx_ecef) + np.linalg.norm(pos_ecef - rx_ecef)
        path_resid = path - float(row["range_km"]) * 1000.0
        max_path_resid_m = max(max_path_resid_m, abs(float(path_resid)))

        unit_from_pos_enu = pos_enu / np.linalg.norm(pos_enu)
        unit_angle_err = angular_separation_deg(unit_from_pos_enu, np.array(row["aod_unit_enu"], dtype=float))
        max_unit_angle_err_deg = max(max_unit_angle_err_deg, unit_angle_err)

        solved = aod_delay_position(tx_ecef, rx_ecef, row["aod_unit_enu"], rot, float(row["range_km"]) * 1000.0)
        if solved is None:
            raise AssertionError(f"Could not resolve AoD/range position for {row['station']} at {row['unix_time']}")
        max_resolve_pos_err_m = max(max_resolve_pos_err_m, float(np.linalg.norm(solved[0] - pos_ecef)))

    assert_close("trail position geodetic/ECEF consistency m", max_geodetic_err_m, 1e-3)
    assert_close("trail position ENU/ECEF consistency m", max_enu_err_m, 1e-6)
    assert_close("trail bistatic path residual m", max_path_resid_m, 1e-6)
    assert_close("trail AoD unit-position angular consistency deg", max_unit_angle_err_deg, 1e-5)
    assert_close("trail AoD solver reproducibility m", max_resolve_pos_err_m, 1e-5)
    return {
        "n_positions": len(rows),
        "max_geodetic_err_m": max_geodetic_err_m,
        "max_enu_err_m": max_enu_err_m,
        "max_path_resid_m": max_path_resid_m,
        "max_unit_angle_err_deg": max_unit_angle_err_deg,
        "max_resolve_pos_err_m": max_resolve_pos_err_m,
    }


def sci(value: float, precision: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.{precision}f}\times 10^{{{exponent}}}"


def write_latex_table(checks: dict[str, dict[str, float]], out: Path) -> None:
    rows = [
        (
            "WGS84 geodetic $\\rightarrow$ ECEF $\\rightarrow$ geodetic",
            "max altitude round-trip error",
            checks["wgs84_round_trip"]["max_alt_err_m"],
            "m",
        ),
        (
            "WGS84 geodetic $\\rightarrow$ ECEF $\\rightarrow$ geodetic",
            "max latitude/longitude round-trip error",
            max(checks["wgs84_round_trip"]["max_lat_err_deg"], checks["wgs84_round_trip"]["max_lon_err_deg"]),
            "deg",
        ),
        (
            "ENU rotation matrix",
            "max $|R^T R-I|$, $|\\det R-1|$, and $|{\\bf e}\\times{\\bf n}-{\\bf u}|$",
            max(
                checks["enu_basis"]["max_orthonormality_error"],
                checks["enu_basis"]["max_determinant_error"],
                checks["enu_basis"]["max_cross_product_error"],
            ),
            "dimensionless",
        ),
        (
            "Azimuth/elevation unit vectors",
            "max cardinal-direction unit-vector error",
            checks["az_el_units"]["max_unit_vector_error"],
            "dimensionless",
        ),
        (
            "Trajectory product",
            "reference-state ENU/ECEF mismatch",
            checks["trajectory_product"]["ref_ecef_err_m"],
            "m",
        ),
        (
            "Trajectory product",
            "max stored model-series altitude mismatch",
            checks["trajectory_product"]["max_series_alt_err_m"],
            "m",
        ),
        (
            "Trail AoD positions",
            f"max ECEF/ENU/geodetic mismatch over {int(checks['trail_aod_positions']['n_positions'])} solves",
            max(checks["trail_aod_positions"]["max_geodetic_err_m"], checks["trail_aod_positions"]["max_enu_err_m"]),
            "m",
        ),
        (
            "Trail AoD positions",
            "max bistatic path-length residual",
            checks["trail_aod_positions"]["max_path_resid_m"],
            "m",
        ),
        (
            "Trail AoD positions",
            "max AoD ray versus solved-position angular mismatch",
            checks["trail_aod_positions"]["max_unit_angle_err_deg"],
            "deg",
        ),
    ]
    lines = [
        "% Generated by tests/coordinate_sanity_checks.py --tex-out\n",
        "\\begin{tabularx}{\\linewidth}{lXl}\n",
        "\\toprule\n",
        "Check & Quantity & Maximum residual \\\\\n",
        "\\midrule\n",
    ]
    for check, quantity, value, unit in rows:
        lines.append(f"{check} & {quantity} & ${sci(float(value))}$ {unit} \\\\\n")
    lines.extend(["\\bottomrule\n", "\\end{tabularx}\n"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coordinate-system consistency checks.")
    parser.add_argument("--tex-out", type=Path, default=None, help="Write a LaTeX table of residuals.")
    args = parser.parse_args()

    checks = {
        "wgs84_round_trip": check_wgs84_round_trip(),
        "enu_basis": check_enu_basis(),
        "az_el_units": check_az_el_units(),
        "trajectory_product": check_trajectory_product(),
        "trail_aod_positions": check_trail_aod_positions(),
    }
    if args.tex_out is not None:
        write_latex_table(checks, args.tex_out)
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("coordinate sanity checks passed")


if __name__ == "__main__":
    main()
