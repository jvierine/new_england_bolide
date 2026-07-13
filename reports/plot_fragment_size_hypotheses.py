#!/usr/bin/env python3
"""Plot fragment-size ballistic hypotheses against sparse terminus images."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef  # noqa: E402
from estimate_head_echo_trajectory import ecef_to_geodetic_wgs84  # noqa: E402
from fit_ballistic_head_trail import atmosphere_velocity_gcrs, build_density_profile, density_at_alt, dynamics, propagate_state, sphere_cd_area_over_mass  # noqa: E402
from fit_combined_head_trail_eci import gcrs_to_ecef  # noqa: E402
OUT_FIG = Path("reports/figures/fragment_size_hypotheses.png")
OUT_SUMMARY = Path("reports/fragment_size_hypotheses_summary.json")
DEFAULT_TRAJECTORY = Path("products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json")
DEFAULT_TRAIL_POSITION_JSON = [
    Path("products/specular_trail_positions/specular_trail_position_summary.json"),
]
DEFAULT_DARK_FLIGHT_KMZ = Path("data/marc_fries/cape_cod_bay_dark_flight_20260626.kmz")
DEFAULT_DARK_FLIGHT_JSON = Path("data/marc_fries/cape_cod_bay_dark_flight_20260625.json")
MILLSTONE_TX_LAT_DEG = 42.611
MILLSTONE_TX_LON_DEG = -71.485
MILLSTONE_TX_ALT_M = 80.0


def parse_utc(value: str) -> float:
    """Parse an ISO-8601 UTC timestamp into Unix seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def utc_string(time_unix: float) -> str:
    """Format Unix seconds as an ISO-8601 UTC timestamp."""
    return datetime.fromtimestamp(float(time_unix), tz=timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fit_model_state_at_altitude(fit: dict, altitude_km: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Interpolate the saved fitted-model state at a requested altitude."""
    model = fit["model"]
    altitude_m = np.asarray(model["alt_m"], dtype=float)
    target_m = float(altitude_km) * 1000.0
    if target_m < float(np.nanmin(altitude_m)) or target_m > float(np.nanmax(altitude_m)):
        raise ValueError(
            f"Requested altitude {altitude_km:g} km is outside fitted model range "
            f"{np.nanmin(altitude_m) / 1000.0:.2f}--{np.nanmax(altitude_m) / 1000.0:.2f} km"
        )
    order = np.argsort(altitude_m)
    altitude_sorted = altitude_m[order]
    time_unix = float(np.interp(target_m, altitude_sorted, np.asarray(model["time_unix"], dtype=float)[order]))
    position = np.asarray(
        [np.interp(target_m, altitude_sorted, np.asarray(model["position_gcrs_m"], dtype=float)[order, axis]) for axis in range(3)],
        dtype=float,
    )
    velocity = np.asarray(
        [np.interp(target_m, altitude_sorted, np.asarray(model["velocity_gcrs_mps"], dtype=float)[order, axis]) for axis in range(3)],
        dtype=float,
    )
    return time_unix, position, velocity


def gcrs_path_to_llh(pos_gcrs: np.ndarray, time_unix: np.ndarray) -> np.ndarray:
    return np.array([ecef_to_geodetic_wgs84(gcrs_to_ecef(p, t)) for p, t in zip(pos_gcrs, time_unix, strict=True)], dtype=float)


def air_relative_speed(pos_gcrs: np.ndarray, vel_gcrs: np.ndarray, time_unix: np.ndarray) -> np.ndarray:
    v_air = np.array([atmosphere_velocity_gcrs(p, t) for p, t in zip(pos_gcrs, time_unix, strict=True)], dtype=float)
    return np.linalg.norm(np.asarray(vel_gcrs, dtype=float) - v_air, axis=1)


def meteor_state_summary_at_altitude(fit: dict, altitude_km: float, velocity_reference_altitude_km: float | None = None) -> dict[str, float | str]:
    t_unix, pos_gcrs, vel_gcrs = fit_model_state_at_altitude(fit, altitude_km)
    pos_ecef = gcrs_to_ecef(pos_gcrs, t_unix)
    lat_deg, lon_deg, alt_m = ecef_to_geodetic_wgs84(pos_ecef)
    direction_altitude_km = altitude_km if velocity_reference_altitude_km is None else float(velocity_reference_altitude_km)
    _t_dir, _pos_dir, vel_dir_gcrs = fit_model_state_at_altitude(fit, direction_altitude_km)
    dt_s = 0.01
    # Transform a short inertial displacement into ECEF to include Earth rotation
    # in the local ground-referenced direction of motion.
    next_ecef = gcrs_to_ecef(pos_gcrs + vel_dir_gcrs * dt_s, t_unix + dt_s)
    vel_ecef = (next_ecef - pos_ecef) / dt_s
    vel_enu = enu_to_ecef_matrix(lat_deg, lon_deg).T @ vel_ecef
    east, north, up = vel_enu
    horiz = math.hypot(float(east), float(north))
    az_deg = (math.degrees(math.atan2(float(east), float(north))) + 360.0) % 360.0
    el_deg = math.degrees(math.atan2(float(up), horiz))
    return {
        "time_unix": float(t_unix),
        "time_utc": utc_string(t_unix),
        "lat_deg": float(lat_deg),
        "lon_deg": float(lon_deg),
        "alt_km": float(alt_m) / 1000.0,
        "velocity_reference_alt_km": direction_altitude_km,
        "velocity_azimuth_deg": float(az_deg),
        "velocity_elevation_deg": float(el_deg),
        "ground_referenced_speed_km_s": float(np.linalg.norm(vel_ecef) / 1000.0),
        "air_relative_speed_km_s": float(np.linalg.norm(vel_gcrs - atmosphere_velocity_gcrs(pos_gcrs, t_unix)) / 1000.0),
    }


def lowest_component_summary(fit: dict, components: list[dict]) -> dict[str, object] | None:
    if not components:
        return None
    finite = [row for row in components if np.isfinite(float(row.get("alt_m", np.nan)))]
    if not finite:
        return None
    row = min(finite, key=lambda item: float(item["alt_m"]))
    blob_alt_km = float(row["alt_m"]) / 1000.0
    return {
        "station": str(row.get("station", "")),
        "blob_lat_deg": float(row["lat_deg"]),
        "blob_lon_deg": float(row["lon_deg"]),
        "blob_alt_km": blob_alt_km,
        "meteor_at_blob_altitude": meteor_state_summary_at_altitude(fit, blob_alt_km, velocity_reference_altitude_km=100.0),
    }


def elevation_altitude_km(lat_deg: float, lon_deg: float, elevation_deg: float) -> float:
    """Return target altitude at lat/lon corresponding to a transmitter elevation angle."""
    tx = geodetic_to_ecef(MILLSTONE_TX_LAT_DEG, MILLSTONE_TX_LON_DEG, MILLSTONE_TX_ALT_M)
    tx_up = enu_to_ecef_matrix(MILLSTONE_TX_LAT_DEG, MILLSTONE_TX_LON_DEG)[:, 2]
    surface = geodetic_to_ecef(lat_deg, lon_deg, 0.0)
    target_up = enu_to_ecef_matrix(lat_deg, lon_deg)[:, 2]
    horizontal = surface - tx
    sin_el = math.sin(math.radians(float(elevation_deg)))
    denom = float(np.dot(target_up, tx_up) - sin_el)
    if abs(denom) < 1.0e-9:
        return float("nan")
    rhs = float(sin_el * np.linalg.norm(horizontal) - np.dot(horizontal, tx_up))
    return rhs / denom / 1000.0


def trail_position_h5_from_state(state_path: Path, out_path: Path) -> None:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    rows = payload.get("components", [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_fields = [
        "component_id",
        "time_unix",
        "range_km",
        "range_idx",
        "doppler_hz",
        "snr_db_2hz",
        "lat_deg",
        "lon_deg",
        "alt_m",
        "alt_km",
        "passage_time_unix",
        "plasma_lag_s",
        "passage_path_residual_km",
        "trajectory_visibility_score",
        "trajectory_horizontal_displacement_km",
        "blob_axis_length_along_m",
        "blob_axis_length_cross_m",
        "blob_axis_length_along_uncertainty_m",
        "blob_axis_length_cross_uncertainty_m",
        "centroid_alt_sigma_m",
        "p_blob_real",
        "real_probability_score_z",
        "blob_chi2_reduced",
        "blob_chi2_dof",
    ]
    string_fields = ["station", "time_utc", "position_model", "passage_time_utc"]
    with h5py.File(out_path, "w") as h:
        h.attrs["source_program"] = "reports/plot_fragment_size_hypotheses.py"
        h.attrs["state_file"] = str(state_path)
        h.attrs["station"] = payload.get("station", "")
        h.attrs["n_components"] = len(rows)
        for field in numeric_fields:
            values = []
            for idx, row in enumerate(rows, start=1):
                if field == "component_id":
                    value = idx
                elif field == "alt_km":
                    value = None if row.get("alt_m") is None else float(row["alt_m"]) / 1000.0
                else:
                    value = row.get(field)
                values.append(np.nan if value is None else float(value))
            h.create_dataset(field, data=np.asarray(values, dtype=np.float64))
        str_dtype = h5py.string_dtype(encoding="utf-8")
        for field in string_fields:
            h.create_dataset(field, data=np.asarray([str(row.get(field, "")) for row in rows], dtype=object), dtype=str_dtype)


def ensure_gui_trail_position_h5(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            continue
        state = path.with_name(path.name.replace("_trail_positions.h5", "_interactive_components.json"))
        if state.exists():
            trail_position_h5_from_state(state, path)


def load_components_h5(paths: list[Path], max_height_km: float) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"GUI-derived trail-position HDF5 does not exist: {path}")
        with h5py.File(path, "r") as h:
            n = int(h.attrs.get("n_components", len(h["lat_deg"])))
            for idx in range(n):
                alt_m = float(h["alt_m"][idx])
                alt_km = alt_m / 1000.0
                if not np.isfinite(alt_km) or alt_km < 0.0 or alt_km > max_height_km:
                    continue
                station_value = h["station"][idx]
                if isinstance(station_value, bytes):
                    station_value = station_value.decode("utf-8")
                rows.append(
                    {
                        "station": str(station_value),
                        "lat_deg": float(h["lat_deg"][idx]),
                        "lon_deg": float(h["lon_deg"][idx]),
                        "alt_m": alt_m,
                        "centroid_alt_sigma_m": float(h["centroid_alt_sigma_m"][idx]) if "centroid_alt_sigma_m" in h else np.nan,
                        "blob_axis_length_along_m": float(h["blob_axis_length_along_m"][idx]) if "blob_axis_length_along_m" in h else np.nan,
                        "blob_axis_length_cross_m": float(h["blob_axis_length_cross_m"][idx]) if "blob_axis_length_cross_m" in h else np.nan,
                    }
                )
    return rows


def load_components_json(paths: list[Path], max_height_km: float, station: str = "beverly") -> list[dict]:
    """Load tracked trail-plasma positions from the AoD solution JSON."""
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Trail-position JSON does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("positions", []):
            alt_m = float(row.get("alt_m", np.nan))
            alt_km = alt_m / 1000.0
            if str(row.get("station", "")).lower() != station.lower():
                continue
            if not np.isfinite(alt_km) or alt_km < 0.0 or alt_km > max_height_km:
                continue
            # Reject the known AoD-alias solution far from the trajectory prior.
            if float(row.get("aod_prior_offset_deg", 0.0)) > 10.0:
                continue
            rows.append(
                {
                    "station": str(row.get("station", "")),
                    "lat_deg": float(row["lat_deg"]),
                    "lon_deg": float(row["lon_deg"]),
                    "alt_m": alt_m,
                    "centroid_alt_sigma_m": np.nan,
                    "blob_axis_length_along_m": np.nan,
                    "blob_axis_length_cross_m": np.nan,
                }
            )
    return rows


def add_plasma_blob(ax, x: float, z_km: float, width_x: float, height_km: float, color: str, zorder: float = 1.0) -> None:
    if not all(np.isfinite(v) for v in (x, z_km, width_x, height_km)):
        return
    if width_x <= 0.0 or height_km <= 0.0:
        return
    for scale, alpha in ((2.0, 0.10), (1.0, 0.20)):
        patch = Ellipse(
            (x, z_km),
            width=scale * width_x,
            height=scale * height_km,
            facecolor=color,
            edgecolor=color,
            lw=0.7,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(patch)


def propagate_hypotheses(
    fit: dict,
    diameters_m: list[float],
    density_kg_m3: float,
    launch_altitude_km: float,
    max_duration_s: float,
    cd: float,
    dt_s: float,
    max_step_s: float,
) -> tuple[list[dict], dict]:
    t_launch, pos_launch, vel_launch = fit_model_state_at_altitude(fit, launch_altitude_km)
    stop_time_unix = t_launch + float(max_duration_s)
    density_time = np.linspace(t_launch, stop_time_unix, 400)
    density_profile = build_density_profile(density_time, [], np.r_[pos_launch, vel_launch], t_launch)
    launch_alt_m = launch_altitude_km * 1000.0
    launch_rho = density_at_alt(density_profile, launch_alt_m)

    paths = []
    for diameter_m in diameters_m:
        cd_area_mass = sphere_cd_area_over_mass(diameter_m, density_kg_m3, cd)
        y0 = np.r_[pos_launch, vel_launch]

        def ode(t_rel: float, y: np.ndarray) -> np.ndarray:
            return dynamics(t_rel, y, t_launch, cd_area_mass, density_profile, use_drag=True)

        def sea_level_event(t_rel: float, y: np.ndarray) -> float:
            return float(ecef_to_geodetic_wgs84(gcrs_to_ecef(y[:3], t_launch + float(t_rel)))[2])

        sea_level_event.terminal = True
        sea_level_event.direction = -1.0
        sol = solve_ivp(
            ode,
            (0.0, float(max_duration_s)),
            y0,
            method="RK45",
            rtol=1e-7,
            atol=1e-3,
            max_step=float(max_step_s),
            dense_output=True,
            events=sea_level_event,
        )
        t_end_rel = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
        rel_sample = np.arange(0.0, t_end_rel + dt_s, dt_s)
        if rel_sample.size == 0 or rel_sample[-1] < t_end_rel:
            rel_sample = np.r_[rel_sample, t_end_rel]
        y_sample = sol.sol(rel_sample).T if sol.sol is not None else np.column_stack([np.interp(rel_sample, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])
        t_eval = t_launch + rel_sample
        pos = y_sample[:, :3]
        vel = y_sample[:, 3:]
        llh = gcrs_path_to_llh(pos, t_eval)
        speed_air = air_relative_speed(pos, vel, t_eval)
        rho_air = np.array([density_at_alt(density_profile, alt_m) for alt_m in llh[:, 2]], dtype=float)
        dynamic_pressure_pa = 0.5 * rho_air * speed_air**2
        keep = np.where(
            (llh[:, 2] >= 0.0)
            & (llh[:, 2] <= 150000.0)
            & (llh[:, 0] >= 35.0)
            & (llh[:, 0] <= 50.0)
            & (llh[:, 1] >= -80.0)
            & (llh[:, 1] <= -60.0)
        )[0]
        radius = 0.5 * float(diameter_m)
        mass_kg = (4.0 / 3.0) * math.pi * radius**3 * float(density_kg_m3)
        paths.append(
            {
                "diameter_m": float(diameter_m),
                "mass_kg": float(mass_kg),
                "cd_area_over_mass_m2_kg": float(cd_area_mass),
                "time_unix": t_eval[keep],
                "lat_deg": llh[keep, 0],
                "lon_deg": llh[keep, 1],
                "alt_km": llh[keep, 2] / 1000.0,
                "air_relative_speed_km_s": speed_air[keep] / 1000.0,
                "dynamic_pressure_mpa": dynamic_pressure_pa[keep] / 1.0e6,
                "reached_sea_level": bool(len(sol.t_events[0]) > 0),
                "impact_time_unix": None if len(sol.t_events[0]) == 0 else float(t_launch + sol.t_events[0][0]),
            }
        )
    launch = {
        "time_unix": float(t_launch),
        "time_utc": utc_string(t_launch),
        "altitude_km": float(launch_altitude_km),
        "density_kg_m3": float(launch_rho),
        "inertial_speed_km_s": float(np.linalg.norm(vel_launch) / 1000.0),
        "air_relative_speed_km_s": float(np.linalg.norm(vel_launch - atmosphere_velocity_gcrs(pos_launch, t_launch)) / 1000.0),
    }
    return paths, launch


def interp_path_at_time(path: dict, marker_time_unix: float) -> dict[str, float] | None:
    time = np.asarray(path["time_unix"], dtype=float)
    if time.size == 0 or marker_time_unix < float(time[0]) or marker_time_unix > float(time[-1]):
        return None
    return {
        "lat_deg": float(np.interp(marker_time_unix, time, path["lat_deg"])),
        "lon_deg": float(np.interp(marker_time_unix, time, path["lon_deg"])),
        "alt_km": float(np.interp(marker_time_unix, time, path["alt_km"])),
        "air_relative_speed_km_s": float(np.interp(marker_time_unix, time, path["air_relative_speed_km_s"])),
        "dynamic_pressure_mpa": float(np.interp(marker_time_unix, time, path["dynamic_pressure_mpa"])),
    }


def terminus_marker_summary(fit: dict, altitude_km: float) -> dict[str, float]:
    marker = meteor_state_summary_at_altitude(fit, altitude_km, velocity_reference_altitude_km=100.0)
    t_unix, pos_gcrs, vel_gcrs = fit_model_state_at_altitude(fit, altitude_km)
    density_profile = build_density_profile(np.array([t_unix], dtype=float), [], np.r_[pos_gcrs, vel_gcrs], t_unix)
    air_speed_mps = float(marker["air_relative_speed_km_s"]) * 1000.0
    rho = density_at_alt(density_profile, float(marker["alt_km"]) * 1000.0)
    marker["dynamic_pressure_mpa"] = 0.5 * rho * air_speed_mps**2 / 1.0e6
    return marker


def build_fitted_trajectory(fit: dict, upper_altitude_km: float, lower_altitude_km: float) -> dict[str, np.ndarray]:
    altitudes_km = np.linspace(float(upper_altitude_km), float(lower_altitude_km), 320)
    states = [fit_model_state_at_altitude(fit, float(altitude_km)) for altitude_km in altitudes_km]
    time_unix = np.asarray([state[0] for state in states], dtype=float)
    pos_gcrs = np.asarray([state[1] for state in states], dtype=float)
    vel_gcrs = np.asarray([state[2] for state in states], dtype=float)
    llh = gcrs_path_to_llh(pos_gcrs, time_unix)
    speed_air = air_relative_speed(pos_gcrs, vel_gcrs, time_unix)
    density_profile = build_density_profile(time_unix, [], np.r_[pos_gcrs[0], vel_gcrs[0]], float(time_unix[0]))
    rho_air = np.asarray([density_at_alt(density_profile, altitude_m) for altitude_m in llh[:, 2]], dtype=float)
    return {
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1000.0,
        "air_relative_speed_km_s": speed_air / 1000.0,
        "dynamic_pressure_mpa": 0.5 * rho_air * speed_air**2 / 1.0e6,
    }


def load_dark_flight_kmz(path: Path) -> list[dict]:
    """Load Jormungandr2 radar-constrained flight paths from a KMZ."""
    if not path.exists():
        raise FileNotFoundError(f"Dark-flight KMZ does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    rows = []
    for placemark in root.findall(".//kml:Placemark", namespace):
        name = placemark.findtext("kml:name", default="", namespaces=namespace)
        if not (name.startswith("Flight ") or name.startswith("Path ")):
            continue
        coordinates_text = placemark.findtext(".//kml:LineString/kml:coordinates", default="", namespaces=namespace)
        coordinates = []
        for token in coordinates_text.split():
            values = token.split(",")
            if len(values) < 3:
                continue
            lon_deg, lat_deg, alt_m = map(float, values[:3])
            coordinates.append((lat_deg, lon_deg, alt_m / 1000.0))
        if not coordinates:
            continue
        match = re.search(r"(?:_|Path )([0-9]+(?:\.[0-9]+)?)g$", name)
        mass_g = float(match.group(1)) if match else float("nan")
        path_kind = "radar_constrained" if name.startswith("Flight ") else "generic_mass_hypothesis"
        radar = name.removeprefix("Flight ").split("_", 1)[0] if path_kind == "radar_constrained" else None
        values = np.asarray(coordinates, dtype=float)
        rows.append(
            {
                "name": name,
                "radar": radar,
                "path_kind": path_kind,
                "mass_g": mass_g,
                "lat_deg": values[:, 0],
                "lon_deg": values[:, 1],
                "alt_km": values[:, 2],
                "landing_lat_deg": float(values[-1, 0]),
                "landing_lon_deg": float(values[-1, 1]),
            }
        )
    if not rows:
        raise ValueError(f"No predicted dark-flight paths found in {path}")
    return sorted(rows, key=lambda row: (str(row["path_kind"]), float(row["mass_g"])))


def load_dark_flight_radar_detections(path: Path) -> list[dict]:
    """Load the weather-radar centroids recorded in Marc Fries's JSON output."""
    if not path.exists():
        raise FileNotFoundError(f"Dark-flight JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("Masses", []):
        lat_deg = row.get("obs_blip_lat", row.get("blip_lat"))
        lon_deg = row.get("obs_blip_lon", row.get("blip_lon"))
        alt_m = row.get("obs_blip_alt_m")
        if lat_deg is None or lon_deg is None or alt_m is None:
            continue
        rows.append(
            {
                "radar": str(row.get("Radar", "")),
                "time_utc": str(row.get("Mean Time", "")),
                "lat_deg": float(lat_deg),
                "lon_deg": float(lon_deg),
                "alt_km": float(alt_m) / 1000.0,
            }
        )
    if not rows:
        raise ValueError(f"No weather-radar centroids found in {path}")
    return rows


def component_reference_lonlat(paths: list[dict], components: list[dict]) -> tuple[float, float]:
    points = []
    for row in components:
        lat = float(row["lat_deg"])
        lon = float(row["lon_deg"])
        if np.isfinite(lat) and np.isfinite(lon):
            points.append((lat, lon))
    if not points:
        for path in paths:
            lat = np.asarray(path["lat_deg"], dtype=float)
            lon = np.asarray(path["lon_deg"], dtype=float)
            alt = np.asarray(path["alt_km"], dtype=float)
            keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt) & (alt >= 20.0) & (alt <= 100.0)
            points.extend(zip(lat[keep], lon[keep], strict=True))
    if not points:
        return float("nan"), float("nan")
    return (
        float(np.median([p[0] for p in points])),
        float(np.median([p[1] for p in points])),
    )


def make_plot(
    out: Path,
    paths: list[dict],
    components: list[dict],
    launch: dict,
    marker_time_unix: float,
    terminus_marker: dict[str, float],
    trajectory_min_altitude_km: float,
    dark_flight_paths: list[dict],
    dark_flight_detections: list[dict],
) -> dict[str, float]:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.0), sharey=True, constrained_layout=True)
    ax_lat, ax_lon, ax_q = axes
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]

    marker_label_added = False
    for idx, path in enumerate(paths):
        label = f"{path['diameter_m']:g} m diameter hypothesis"
        style = linestyles[idx % len(linestyles)]
        q = np.asarray(path["dynamic_pressure_mpa"], dtype=float)
        q_plot = np.where(q > 0.0, q, np.nan)
        show_dynamic_pressure = np.isclose(float(path["diameter_m"]), 1.0)
        if show_dynamic_pressure:
            ax_q.plot(q_plot, path["alt_km"], color="0.35", lw=1.55, linestyle=style, label=label, zorder=4)
        marker = interp_path_at_time(path, marker_time_unix) if show_dynamic_pressure else None
        if marker is not None:
            marker_label = None
            if not marker_label_added:
                marker_label = "18:06:23 UTC"
                marker_label_added = True
            if marker["dynamic_pressure_mpa"] > 0.0:
                ax_q.scatter(
                    [marker["dynamic_pressure_mpa"]],
                    [marker["alt_km"]],
                    marker="o",
                    s=34,
                    color="0.15",
                    edgecolor="white",
                    linewidth=0.45,
                    label=marker_label,
                    zorder=7,
                )

    for idx, dark_path in enumerate(dark_flight_paths):
        generic = dark_path["path_kind"] == "generic_mass_hypothesis"
        same_kind_before = any(row["path_kind"] == dark_path["path_kind"] for row in dark_flight_paths[:idx])
        label = None
        if not same_kind_before:
            label = "Jörmungandr2 generic mass hypotheses" if generic else "Jörmungandr2 radar-constrained paths"
        path_style = {
            "color": "#16823b",
            "lw": 1.25 if generic else 1.5,
            "alpha": 0.58 if generic else 0.82,
            "linestyle": "--" if generic else "-",
            "zorder": 5,
        }
        ax_lat.plot(dark_path["lat_deg"], dark_path["alt_km"], **path_style)
        ax_lon.plot(
            dark_path["lon_deg"],
            dark_path["alt_km"],
            label=label,
            **path_style,
        )
        if not generic:
            ax_lat.scatter([dark_path["landing_lat_deg"]], [0.0], marker="s", s=24, color="#16823b", edgecolor="black", linewidth=0.4, zorder=7)
            ax_lon.scatter([dark_path["landing_lon_deg"]], [0.0], marker="s", s=24, color="#16823b", edgecolor="black", linewidth=0.4, zorder=7)

    if dark_flight_detections:
        detection_lat = np.asarray([row["lat_deg"] for row in dark_flight_detections], dtype=float)
        detection_lon = np.asarray([row["lon_deg"] for row in dark_flight_detections], dtype=float)
        detection_alt = np.asarray([row["alt_km"] for row in dark_flight_detections], dtype=float)
        detection_style = {
            "marker": "D",
            "s": 32,
            "color": "#c9342f",
            "edgecolor": "black",
            "linewidth": 0.45,
            "zorder": 8,
        }
        ax_lat.scatter(detection_lat, detection_alt, **detection_style)
        ax_lon.scatter(detection_lon, detection_alt, label="Weather-radar meteorite detections", **detection_style)

    if components:
        stations = sorted(set(str(c["station"]) for c in components))
        colors = {"beverly": "#1f77b4", "clinton": "#d62728"}
        for station in stations:
            rows = [c for c in components if str(c["station"]) == station]
            lat = np.array([float(c["lat_deg"]) for c in rows])
            lon = np.array([float(c["lon_deg"]) for c in rows])
            alt = np.array([float(c["alt_m"]) / 1000.0 for c in rows])
            color = colors.get(station, "C2")
            label = "Plasma trail position (Beverly)" if station == "beverly" else f"Plasma trail position ({station})"
            for row in rows:
                z_km = float(row["alt_m"]) / 1000.0
                alt_sigma_km = row.get("centroid_alt_sigma_m", np.nan) / 1000.0
                horiz_km = max(
                    row.get("blob_axis_length_along_m", np.nan) / 1000.0,
                    row.get("blob_axis_length_cross_m", np.nan) / 1000.0,
                )
                lat_width = horiz_km / 111.0
                lon_width = horiz_km / (111.0 * max(np.cos(np.deg2rad(float(row["lat_deg"]))), 0.1))
                height_km = max(2.0 * alt_sigma_km, row.get("blob_axis_length_cross_m", np.nan) / 1000.0)
                add_plasma_blob(ax_lat, float(row["lat_deg"]), z_km, lat_width, height_km, color, zorder=1)
                add_plasma_blob(ax_lon, float(row["lon_deg"]), z_km, lon_width, height_km, color, zorder=1)
            ax_lat.scatter(lat, alt, s=18, color=color, edgecolor="black", linewidth=0.35, alpha=0.95, zorder=5)
            ax_lon.scatter(lon, alt, s=18, color=color, edgecolor="black", linewidth=0.35, alpha=0.95, label=label, zorder=5)

    ax_lat.set_xlabel("Latitude (deg)")
    ax_lon.set_xlabel("Longitude (deg)")
    ax_q.set_xlabel("Dynamic pressure (MPa)")
    ax_lat.set_ylabel("Height (km)")
    ax_lat.set_title("Latitude")
    ax_lon.set_title("Longitude")
    ax_q.set_title("Dynamic pressure")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.set_ylim(0, 120)
    ax_q.set_xscale("log")
    ax_lon.legend(loc="upper right", fontsize=7.2)
    ax_q.legend(loc="best", fontsize=7.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, facecolor="white", transparent=False)
    plt.close(fig)
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--trail-positions-json", nargs="+", type=Path, default=DEFAULT_TRAIL_POSITION_JSON)
    p.add_argument("--marker-time", default="2026-05-30T18:06:23Z")
    p.add_argument("--terminus-altitude-km", type=float, default=33.61)
    p.add_argument("--right-panel-min-altitude-km", type=float, default=18.0)
    p.add_argument("--dark-flight-kmz", type=Path, default=DEFAULT_DARK_FLIGHT_KMZ)
    p.add_argument("--dark-flight-json", type=Path, default=DEFAULT_DARK_FLIGHT_JSON)
    p.add_argument("--max-duration-s", type=float, default=7200.0)
    p.add_argument("--launch-altitude-km", type=float, default=90.0)
    p.add_argument("--density-kg-m3", type=float, default=3000.0)
    p.add_argument("--diameters-m", nargs="+", type=float, default=[1.0])
    p.add_argument("--cd", type=float, default=1.0)
    p.add_argument("--dt-s", type=float, default=0.25)
    p.add_argument("--max-step-s", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=OUT_FIG)
    p.add_argument("--summary", type=Path, default=OUT_SUMMARY)
    args = p.parse_args()

    fit = json.loads(args.trajectory.read_text(encoding="utf-8"))
    marker_time = parse_utc(args.marker_time)
    paths, launch = propagate_hypotheses(
        fit,
        [float(v) for v in args.diameters_m],
        float(args.density_kg_m3),
        float(args.launch_altitude_km),
        float(args.max_duration_s),
        float(args.cd),
        float(args.dt_s),
        float(args.max_step_s),
    )
    components = load_components_json([Path(v) for v in args.trail_positions_json], max_height_km=120.0)
    dark_flight_paths = load_dark_flight_kmz(args.dark_flight_kmz)
    dark_flight_detections = load_dark_flight_radar_detections(args.dark_flight_json)
    terminus_marker = terminus_marker_summary(fit, float(args.terminus_altitude_km))
    elevation_summary = make_plot(
        args.out,
        paths,
        components,
        launch,
        marker_time,
        terminus_marker,
        float(args.right_panel_min_altitude_km),
        dark_flight_paths,
        dark_flight_detections,
    )
    lowest_blob = lowest_component_summary(fit, components)

    payload = {
        "source_program": "reports/plot_fragment_size_hypotheses.py",
        "trajectory": str(args.trajectory),
        "trail_positions_json": [str(v) for v in args.trail_positions_json],
        "figure": str(args.out),
        "launch": launch,
        "marker_time_unix": float(marker_time),
        "marker_time_utc": utc_string(marker_time),
        "terminus_marker": terminus_marker,
        "dark_flight_kmz": str(args.dark_flight_kmz),
        "dark_flight_json": str(args.dark_flight_json),
        "dark_flight_paths": [
            {
                "radar": row["radar"],
                "mass_g": row["mass_g"],
                "landing_lat_deg": row["landing_lat_deg"],
                "landing_lon_deg": row["landing_lon_deg"],
            }
            for row in dark_flight_paths
        ],
        "dark_flight_radar_detections": dark_flight_detections,
        "millstone_elevation_height_km": elevation_summary,
        "millstone_tx_lat_deg": MILLSTONE_TX_LAT_DEG,
        "millstone_tx_lon_deg": MILLSTONE_TX_LON_DEG,
        "millstone_tx_alt_m": MILLSTONE_TX_ALT_M,
        "max_duration_s": float(args.max_duration_s),
        "density_kg_m3": float(args.density_kg_m3),
        "cd": float(args.cd),
        "diameters": [
            {
                "diameter_m": path["diameter_m"],
                "mass_kg": path["mass_kg"],
                "cd_area_over_mass_m2_kg": path["cd_area_over_mass_m2_kg"],
                "final_height_km": float(path["alt_km"][-1]) if len(path["alt_km"]) else None,
                "final_air_relative_speed_km_s": float(path["air_relative_speed_km_s"][-1]) if len(path["air_relative_speed_km_s"]) else None,
                "final_lat_deg": float(path["lat_deg"][-1]) if len(path["lat_deg"]) else None,
                "final_lon_deg": float(path["lon_deg"][-1]) if len(path["lon_deg"]) else None,
                "marker": interp_path_at_time(path, marker_time),
                "final_dynamic_pressure_mpa": float(path["dynamic_pressure_mpa"][-1]) if len(path["dynamic_pressure_mpa"]) else None,
                "reached_sea_level": bool(path.get("reached_sea_level", False)),
            }
            for path in paths
        ],
        "n_components": len(components),
        "lowest_altitude_blob": lowest_blob,
    }
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
