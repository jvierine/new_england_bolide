#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from station_config import STATION_ORDER, load_all_station_configs, transmitter_config
from estimate_head_echo_trajectory import model_state_series
from angle_departure_estimator import enu_to_ecef_matrix, geodetic_to_ecef


DATA_DIR = ROOT / "2026_05_30_bolide"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TRAJECTORY_JSON = ROOT / "products" / "head_echo_trajectory_roi_fit" / "head_echo_trajectory.json"
C = 299792458.0


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def write_station_coordinate_table(configs: dict) -> Path:
    lines = [
        r"\begin{tabular}{lrrrl}",
        "\n",
        r"\toprule",
        "\n",
        r"Place & Latitude (deg) & Longitude (deg) & Altitude (m) & Role \\",
        "\n",
        r"\midrule",
        "\n",
    ]
    tx = transmitter_config(DATA_DIR)
    lines.append(
        f"{tex_escape(tx.tx_place)} & {tx.gps_tx[0]:.6f} & {tx.gps_tx[1]:.6f} & {tx.gps_tx[2]:.1f} & transmitter \\\\\n"
    )
    for station in STATION_ORDER:
        cfg = configs[station]
        lines.append(
            f"{tex_escape(cfg.rx_place)} & {cfg.gps_rx[0]:.6f} & {cfg.gps_rx[1]:.6f} & {cfg.gps_rx[2]:.1f} & receiver \\\\\n"
        )
    lines.extend([r"\bottomrule", "\n", r"\end{tabular}", "\n"])
    out = REPORT_DIR / "station_coordinates_table.tex"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def write_tx_antenna_table(tx) -> Path:
    lines = [
        r"\begin{tabular}{rrrr}",
        "\n",
        r"\toprule",
        "\n",
        r"TX index & East (m) & North (m) & Seed \\",
        "\n",
        r"\midrule",
        "\n",
    ]
    for idx, (enu, seed) in enumerate(zip(tx.antenna_pos_tx, tx.seeds)):
        lines.append(f"{idx} & {enu[0]:.2f} & {enu[1]:.2f} & {seed} \\\\\n")
    lines.extend([r"\bottomrule", "\n", r"\end{tabular}", "\n"])
    out = REPORT_DIR / "tx_antenna_coordinates_table.tex"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def plot_tx_antennas(tx) -> Path:
    ant = tx.antenna_pos_tx
    fig, ax = plt.subplots(figsize=(6.6, 6.2), constrained_layout=True)
    ax.scatter(ant[:, 0], ant[:, 1], s=95, color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=3)
    for idx, (enu, seed) in enumerate(zip(ant, tx.seeds)):
        ax.annotate(
            f"TX{idx}\nseed {seed}",
            (enu[0], enu[1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0, color="0.78", lw=0.8)
    ax.axvline(0, color="0.78", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    pad = 8.0
    ax.set_xlim(float(np.min(ant[:, 0]) - pad), float(np.max(ant[:, 0]) + pad))
    ax.set_ylim(float(np.min(ant[:, 1]) - pad), float(np.max(ant[:, 1]) + pad))
    ax.set_xlabel("East offset from TX reference (m)")
    ax.set_ylabel("North offset from TX reference (m)")
    ax.set_title("Haystack transmit antenna offsets and code seeds")
    ax.grid(True, color="0.9", linewidth=0.5)
    out = FIG_DIR / "tx_antenna_coordinates.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def trajectory_reference_direction_enu(tx) -> tuple[np.ndarray, dict]:
    fit = json.loads(TRAJECTORY_JSON.read_text(encoding="utf-8"))
    params = np.r_[
        fit["fit_parameters_enu_at_tx"]["position_ref_m"],
        fit["fit_parameters_enu_at_tx"]["velocity_mps"],
    ]
    tx_ecef = geodetic_to_ecef(*tx.gps_tx)
    rot = enu_to_ecef_matrix(tx.gps_tx[0], tx.gps_tx[1])
    model = model_state_series(params, [fit["t_ref_unix"]], fit["t_ref_unix"], tx_ecef, rot)
    pos_enu = model["position_enu_m"][0]
    direction = pos_enu / np.linalg.norm(pos_enu)
    az_deg = (np.rad2deg(np.arctan2(direction[0], direction[1])) + 360.0) % 360.0
    el_deg = np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    return direction, {"az_deg": float(az_deg), "el_deg": float(el_deg), "t_ref_utc": fit["t_ref_utc"]}


def plot_aod_match_function(tx, elevation_floor_deg: float = 5.0, grid_size: int = 421) -> Path:
    ant = tx.antenna_pos_tx
    wavelength = C / tx.frequency_hz
    k = 2.0 * np.pi / wavelength
    ref, ref_meta = trajectory_reference_direction_enu(tx)

    u = np.linspace(-1.0, 1.0, grid_size)
    east, north = np.meshgrid(u, u)
    h2 = east * east + north * north
    min_up = np.sin(np.deg2rad(elevation_floor_deg))
    keep = h2 <= 1.0 - min_up * min_up
    up = np.sqrt(np.maximum(0.0, 1.0 - h2))

    dirs = np.column_stack((east[keep], north[keep], up[keep]))
    steering_ref = np.exp(1.0j * k * (ant @ ref))
    steering = np.exp(1.0j * k * (dirs @ ant.T))
    match = np.abs(steering @ steering_ref.conj()) ** 2 / float(len(ant) ** 2)

    z = np.full_like(east, np.nan, dtype=float)
    z[keep] = 10.0 * np.log10(np.maximum(match, 1e-6))
    peak_mask = np.isclose(match, np.nanmax(match), rtol=0.0, atol=1e-8)
    sidelobe = float(np.nanmax(match[~peak_mask])) if np.any(~peak_mask) else float("nan")

    fig, ax = plt.subplots(figsize=(7.4, 6.8), constrained_layout=True)
    mesh = ax.pcolormesh(east, north, z, shading="auto", cmap="viridis", vmin=-30.0, vmax=0.0)
    circle = plt.Circle((0, 0), np.cos(np.deg2rad(elevation_floor_deg)), color="white", fill=False, lw=1.1)
    ax.add_patch(circle)
    ax.scatter([ref[0]], [ref[1]], marker="+", s=120, color="white", linewidth=2.0, label="tri-static t0 look direction")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East direction cosine")
    ax.set_ylabel("North direction cosine")
    ax.set_title(f"Transmit AoD match function, elevation >= {elevation_floor_deg:g} deg")
    ax.grid(True, color="white", alpha=0.22, linewidth=0.5)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Normalized match (dB)")
    txt = (
        f"wavelength {wavelength:.2f} m\n"
        f"look az {ref_meta['az_deg']:.1f} deg, el {ref_meta['el_deg']:.1f} deg\n"
        f"max off-peak {10*np.log10(max(sidelobe, 1e-12)):.1f} dB"
    )
    ax.text(
        0.03,
        0.03,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox=dict(fc="white", ec="0.6", alpha=0.88),
    )
    ax.legend(loc="upper right")
    out = FIG_DIR / "tx_aod_match_function.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = load_all_station_configs(DATA_DIR)
    tx = transmitter_config(DATA_DIR)
    for path in (
        write_station_coordinate_table(configs),
        write_tx_antenna_table(tx),
        plot_tx_antennas(tx),
        plot_aod_match_function(tx),
    ):
        print(path)


if __name__ == "__main__":
    main()
