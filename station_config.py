#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


DEFAULT_DATA_DIR = Path("2026_05_30_bolide")

STATION_ORDER = ["beverly", "clinton", "pomfret", "siena", "amherstnh"]

STATION_PLACE_NAMES = {
    "amherstnh": "Amherst, New Hampshire",
    "beverly": "Beverly, Massachusetts",
    "clinton": "Clinton, Massachusetts",
    "pomfret": "Pomfret, Vermont",
    "siena": "Siena College, New York",
}

TRANSMITTER_PLACE_NAMES = {
    "haystack": "Haystack, Massachusetts",
}


@dataclass(frozen=True)
class StationConfig:
    station: str
    rx_place: str
    tx_name: str
    tx_place: str
    gps_rx: np.ndarray
    gps_tx: np.ndarray
    antenna_pos_tx: np.ndarray
    seeds: list[int]
    frequency_hz: float
    range_resolution_m: float
    code_len: int


def human_station_name(station: str) -> str:
    return STATION_PLACE_NAMES.get(station, station)


def human_tx_name(tx_name: str) -> str:
    return TRANSMITTER_PLACE_NAMES.get(tx_name, tx_name)


def station_yaml_path(data_dir: Path | str, station: str) -> Path:
    return Path(data_dir) / f"analysis-config-millstone_{station}.yaml"


def load_station_config(data_dir: Path | str, station: str) -> StationConfig:
    path = station_yaml_path(data_dir, station)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    system = cfg["System"]
    coded = cfg["Decoding"]["coded-cw"]
    tx_name = str(system["name_tx"])
    return StationConfig(
        station=station,
        rx_place=human_station_name(station),
        tx_name=tx_name,
        tx_place=human_tx_name(tx_name),
        gps_rx=np.asarray(system["gps_rx"], dtype=float),
        gps_tx=np.asarray(system["gps_tx"], dtype=float),
        antenna_pos_tx=np.asarray(system["antenna_pos_tx"], dtype=float),
        seeds=[int(seed) for seed in coded["seeds"]],
        frequency_hz=float(system["frequency"]),
        range_resolution_m=float(system.get("range_resolution", 3000.0)),
        code_len=int(coded["n_code"]),
    )


def load_all_station_configs(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, StationConfig]:
    return {station: load_station_config(data_dir, station) for station in STATION_ORDER}


def transmitter_config(data_dir: Path | str = DEFAULT_DATA_DIR) -> StationConfig:
    return load_station_config(data_dir, STATION_ORDER[0])
