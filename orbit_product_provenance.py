from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _resolve_reference(path: Path, reference: str) -> Path:
    ref = Path(reference)
    if ref.is_absolute():
        return ref
    return path.resolve().parents[2] / ref


def assert_orbit_h5_matches_trajectory_json(path: Path, *, atol_state: float = 1e-9, atol_cov: float = 1e-8) -> None:
    """Raise if an orbit-sample HDF5 is not traceable to its ballistic JSON."""
    with h5py.File(path, "r") as h:
        trajectory_json = str(h.attrs.get("trajectory_json", ""))
        h5_version = str(h.attrs.get("trajectory_fit_version", "unspecified"))
        h5_description = str(h.attrs.get("trajectory_fit_version_description", "unspecified"))
        h5_state = np.asarray(h["fit_mean_state"][()], dtype=float)
        h5_cov = np.asarray(h["fit_covariance_state"][()], dtype=float)

    if not trajectory_json:
        raise ValueError(f"{path} has no trajectory_json attribute")
    if h5_version == "unspecified":
        raise ValueError(f"{path} has unspecified trajectory_fit_version")

    json_path = _resolve_reference(path, trajectory_json)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    json_version = str(payload.get("trajectory_fit_version", "unspecified"))
    json_description = str(payload.get("trajectory_fit_version_description", "unspecified"))
    json_state = np.asarray(payload["state"], dtype=float)
    json_cov = np.asarray(payload["covariance"], dtype=float)

    if json_version != h5_version:
        raise ValueError(
            f"Orbit product version mismatch: {path} has trajectory_fit_version={h5_version!r}, "
            f"but {json_path} has trajectory_fit_version={json_version!r}"
        )
    if json_description != h5_description:
        raise ValueError(
            f"Orbit product version-description mismatch for {path} and {json_path}"
        )
    if not np.allclose(h5_state, json_state, rtol=0.0, atol=atol_state):
        raise ValueError(
            f"Orbit product state mismatch: {path} fit_mean_state differs from {json_path} state "
            f"(max abs diff {np.max(np.abs(h5_state - json_state)):.6g})"
        )
    if not np.allclose(h5_cov, json_cov, rtol=0.0, atol=atol_cov):
        raise ValueError(
            f"Orbit product covariance mismatch: {path} fit_covariance_state differs from "
            f"{json_path} covariance (max abs diff {np.max(np.abs(h5_cov - json_cov)):.6g})"
        )
