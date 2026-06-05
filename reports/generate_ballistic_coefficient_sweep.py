#!/usr/bin/env python3
"""Generate a standalone drag-area sweep report for the bolide fit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit_ballistic_head_trail import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_TRAIL,
    build_density_profile,
    ecef_to_gcrs,
    is_excluded_trail_row,
    load_trail_positions,
)
from fit_ballistic_head_trail_correlated_appendix import (  # noqa: E402
    DEFAULT_HEAD_SELECTION,
    DEFAULT_OUT_DIR,
    C,
    build_measurement_covariance,
    cache_head_station_states_gcrs,
    load_selected_head_observations,
    pre_head_reference_time,
    raw_residuals,
    whiten,
    whitening_from_covariance,
)
from fit_combined_head_trail_eci import station_geometry  # noqa: E402


FIT_JSON = DEFAULT_OUT_DIR / "ballistic_head_trail_correlated_fit.json"
OUT_DIR = ROOT / "reports" / "ballistic_coefficient_sweep"
FIG_DIR = OUT_DIR / "figures"
REPORT_TEX = OUT_DIR / "ballistic_coefficient_sweep_report.tex"
DENSITY_KG_M3 = 3000.0
CD = 1.0


def sphere_mass_from_cd_area_over_mass(cd_area_over_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return diameter and mass for a sphere with fixed density and drag coefficient."""
    b = np.asarray(cd_area_over_mass, dtype=float)
    diameter_m = 3.0 * CD / (2.0 * DENSITY_KG_M3 * b)
    mass_kg = (math.pi / 6.0) * DENSITY_KG_M3 * diameter_m**3
    return diameter_m, mass_kg


def load_measurements(args: argparse.Namespace, fit_payload: dict):
    fit_args = SimpleNamespace(
        data_dir=args.data_dir,
        trail_json=args.trail_json,
        trail_min_snr_db=args.trail_min_snr_db,
        max_aod_offset_deg=args.max_aod_offset_deg,
        min_trail_alt_km=args.min_trail_alt_km,
        max_trail_alt_km=args.max_trail_alt_km,
        model_padding_s=args.model_padding_s,
        model_dt_s=args.model_dt_s,
        head_coherent_codes=args.head_coherent_codes,
        trail_coherent_codes=args.trail_coherent_codes,
        code_period_s=args.code_period_s,
        trail_correlation_range_gate_m=args.trail_correlation_range_gate_m,
        no_drag=False,
    )
    obs = load_selected_head_observations(args.head_selection_csv)
    used_stations = sorted({o["station"] for o in obs})
    geom = station_geometry(args.data_dir, used_stations)
    obs = cache_head_station_states_gcrs(obs, geom)
    t_ref = pre_head_reference_time(obs)
    trail_rows = load_trail_positions(
        args.trail_json,
        args.trail_min_snr_db,
        args.max_aod_offset_deg,
        args.min_trail_alt_km,
        args.max_trail_alt_km,
    )
    trail_rows = [row for row in trail_rows if not is_excluded_trail_row(row)]
    for row in trail_rows:
        row["gcrs_ref_m"] = ecef_to_gcrs(row["position_ecef_m"], row["unix_time"])

    params = np.asarray(fit_payload["fit_parameters"], dtype=float)
    t_head = np.array([o["unix_time"] for o in obs], dtype=float)
    v2 = float(np.dot(params[3:6], params[3:6]))
    t_line = []
    for row in trail_rows:
        tau = float(np.dot(np.asarray(row["gcrs_ref_m"]) - params[:3], params[3:6]) / v2)
        t_line.append(t_ref + tau)
    t_model = np.arange(
        min(np.min(t_head), np.min(t_line)) - args.model_padding_s,
        max(np.max(t_head), np.max(t_line)) + args.model_padding_s + 0.5 * args.model_dt_s,
        args.model_dt_s,
    )
    density_profile = build_density_profile(np.r_[t_head, t_model], trail_rows, params, t_ref)
    cov, cov_meta = build_measurement_covariance(obs, trail_rows, fit_args)
    chol = whitening_from_covariance(cov)
    return fit_args, obs, trail_rows, t_ref, t_model, density_profile, chol, cov_meta


def summarize_residuals(raw: np.ndarray, obs: list[dict], trail_rows: list[dict], chol: np.ndarray) -> dict[str, float]:
    n_head = len(obs)
    range_m = raw[:n_head]
    doppler_hz = raw[n_head : 2 * n_head]
    trail_vec = raw[2 * n_head :].reshape(len(trail_rows), 3)
    trail_norm_m = np.linalg.norm(trail_vec, axis=1)
    white = whiten(raw, chol)
    dof = max(1, raw.size - 6)
    return {
        "range_rms_km": float(np.sqrt(np.mean(range_m**2)) / 1000.0),
        "range_p95_abs_km": float(np.percentile(np.abs(range_m), 95.0) / 1000.0),
        "doppler_rms_hz": float(np.sqrt(np.mean(doppler_hz**2))),
        "doppler_p95_abs_hz": float(np.percentile(np.abs(doppler_hz), 95.0)),
        "trail_median_km": float(np.median(trail_norm_m) / 1000.0),
        "trail_p95_km": float(np.percentile(trail_norm_m, 95.0) / 1000.0),
        "reduced_chi2": float(np.sum(white**2) / dof),
    }


def fixed_drag_whitened_delta(
    delta: np.ndarray,
    base_state: np.ndarray,
    log10_cd_area_over_mass: float,
    obs: list[dict],
    trail_rows: list[dict],
    t_ref: float,
    t_model: np.ndarray,
    density_profile: dict,
    fit_args: SimpleNamespace,
    chol: np.ndarray,
) -> np.ndarray:
    params = np.r_[base_state + delta, log10_cd_area_over_mass]
    return whiten(raw_residuals(params, obs, trail_rows, t_ref, t_model, density_profile, fit_args), chol)


def refit_state_for_drag(
    q0: np.ndarray,
    base_state: np.ndarray,
    log10_cd_area_over_mass: float,
    obs: list[dict],
    trail_rows: list[dict],
    t_ref: float,
    t_model: np.ndarray,
    density_profile: dict,
    fit_args: SimpleNamespace,
    chol: np.ndarray,
    max_nfev: int,
):
    lower = np.r_[np.full(3, -30000.0), np.full(3, -4000.0)]
    upper = np.r_[np.full(3, 30000.0), np.full(3, 4000.0)]
    return least_squares(
        fixed_drag_whitened_delta,
        q0,
        args=(
            base_state,
            log10_cd_area_over_mass,
            obs,
            trail_rows,
            t_ref,
            t_model,
            density_profile,
            fit_args,
            chol,
        ),
        bounds=(lower, upper),
        x_scale=np.r_[np.full(3, 3000.0), np.full(3, 500.0)],
        diff_step=np.r_[np.full(3, 100.0), np.full(3, 10.0)],
        loss="linear",
        max_nfev=max_nfev,
        verbose=0,
    )


def threshold_row(df: pd.DataFrame, tolerance: float) -> tuple[pd.Series, bool]:
    base = df.iloc[0]
    grown = (df["range_rms_km"] > base["range_rms_km"] * (1.0 + tolerance)) | (
        df["doppler_rms_hz"] > base["doppler_rms_hz"] * (1.0 + tolerance)
    )
    grown.iloc[0] = False
    if grown.any():
        return df.loc[grown].iloc[0], True
    return df.iloc[-1], False


def last_unflagged_row(df: pd.DataFrame, threshold: pd.Series, threshold_found: bool) -> pd.Series:
    if not threshold_found:
        return df.iloc[-1]
    before = df[df["scale"] < threshold["scale"]]
    if before.empty:
        return df.iloc[0]
    return before.iloc[-1]


def write_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "scale",
        "cd_area_over_mass",
        "mass_kg",
        "diameter_m",
        "range_rms_km",
        "doppler_rms_hz",
        "trail_median_km",
        "reduced_chi2",
        "state_delta_pos_km",
        "state_delta_vel_mps",
    ]
    lines = [
        "\\begin{tabular}{rrrrrrrrrr}\n",
        "\\toprule\n",
        "Scale & $C_D A/m$ & Mass & Diameter & Range RMS & Doppler RMS & Trail median & $\\chi^2_\\nu$ & $|\\Delta r_0|$ & $|\\Delta v_0|$ \\\\\n",
        " & (m$^2$ kg$^{-1}$) & (kg) & (m) & (km) & (Hz) & (km) & & (km) & (m s$^{-1}$) \\\\\n",
        "\\midrule\n",
    ]
    for row in df[cols].itertuples(index=False):
        lines.append(
            f"{row.scale:.3g} & {row.cd_area_over_mass:.3e} & {row.mass_kg:.2g} & "
            f"{row.diameter_m:.3f} & {row.range_rms_km:.2f} & {row.doppler_rms_hz:.0f} & "
            f"{row.trail_median_km:.2f} & {row.reduced_chi2:.3f} & "
            f"{row.state_delta_pos_km:.2f} & {row.state_delta_vel_mps:.1f} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def plot_sweep(df: pd.DataFrame, threshold: pd.Series, threshold_found: bool, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    x = df["scale"]
    panels = [
        (axes[0, 0], "range_rms_km", "Head range RMS (km)", "C0"),
        (axes[0, 1], "doppler_rms_hz", "Head Doppler RMS (Hz)", "C1"),
        (axes[1, 0], "trail_median_km", "Trail median distance (km)", "C2"),
        (axes[1, 1], "mass_kg", "Equivalent spherical mass (kg)", "C3"),
    ]
    for ax, col, ylabel, color in panels:
        ax.plot(x, df[col], marker="o", lw=1.8, color=color)
        if threshold_found:
            ax.axvline(threshold["scale"], color="0.25", ls="--", lw=1.2)
        ax.set_xscale("log")
        if col == "mass_kg":
            ax.set_yscale("log")
        ax.set_xlabel("$C_D A/m$ scale factor")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color="0.9", lw=0.6)
    fig.suptitle("Fixed-drag refit sweep of drag-area-to-mass ratio")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    threshold: pd.Series,
    last_ok: pd.Series,
    threshold_found: bool,
    fit_payload: dict,
    args: argparse.Namespace,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "ballistic_coefficient_sweep_table.tex"
    write_table(df, table_path)
    nominal = df.iloc[0]
    fit_path_tex = f"\\texttt{{\\detokenize{{{args.fit_json.relative_to(ROOT)}}}}}"
    if threshold_found:
        threshold_text = (
            f"With this rule, the first flagged scale factor is {threshold['scale']:.3g}, "
            f"giving $C_D A/m={threshold['cd_area_over_mass']:.3e}$ m$^2$ kg$^{{-1}}$.  "
            f"For a spherical object with $C_D=1$ and density $3000$ kg m$^{{-3}}$, this "
            f"corresponds to diameter {threshold['diameter_m']:.3f} m and mass "
            f"{threshold['mass_kg']:.2g} kg.  The largest unflagged scale factor is "
            f"{last_ok['scale']:.3g}, corresponding to diameter {last_ok['diameter_m']:.3f} m "
            f"and mass {last_ok['mass_kg']:.2g} kg; this is the sweep-grid estimate of the "
            "smallest mass that remains undetectable under the present rule."
        )
        threshold_caption = "The dashed vertical line marks the simple RMS-growth threshold used in this note."
    else:
        threshold_text = (
            f"The rule is not triggered by the largest tested scale factor, {threshold['scale']:.3g}.  "
            f"At this endpoint, $C_D A/m={threshold['cd_area_over_mass']:.3e}$ m$^2$ kg$^{{-1}}$, "
            f"corresponding to diameter {threshold['diameter_m']:.3f} m and mass "
            f"{threshold['mass_kg']:.2g} kg for the spherical-density conversion."
        )
        threshold_caption = "No RMS-growth threshold was reached within the tested sweep range."
    lines = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\usepackage{booktabs}\n",
        "\\usepackage{graphicx}\n",
        "\\usepackage{amsmath}\n",
        "\\begin{document}\n",
        "\\title{Ballistic-Coefficient Sweep Test for the Cape Cod Bay Bolide}\n",
        "\\author{Generated analysis note}\n",
        "\\maketitle\n",
        "\\section*{Purpose}\n",
        "This test is separate from the manuscript draft.  For each fixed value of the drag-area-to-mass parameter $C_D A/m$, the six GCRS initial-state components are refit while $C_D A/m$ is held fixed.  Increasing $C_D A/m$ corresponds to decreasing the conventional ballistic coefficient $m/(C_D A)$ and therefore to a smaller dynamically allowed mass for a fixed spherical density.  The manuscript fit product is only read; it is not overwritten.\n\n",
        "\\section*{Setup}\n",
        f"The baseline fit product is {fit_path_tex}.  The baseline version is \\texttt{{{fit_payload.get('trajectory_fit_version', 'unspecified')}}}.  The nominal $C_D A/m$ is {nominal['cd_area_over_mass']:.3e} m$^2$ kg$^{{-1}}$, corresponding to the NASA-motivated 1.6 m, 5 ton body model used in the main fit.  For each scale factor, the same retained head-echo range--Doppler measurements and trail-position measurements are refit with a fixed drag parameter.\n\n",
        "\\section*{Simple detectability rule}\n",
        f"As a deliberately simple diagnostic, the first sweep point where either head-echo range RMS or Doppler RMS exceeds the nominal value by more than {100.0 * args.rms_growth_threshold:.0f}\\% is flagged.  {threshold_text}  This is a sensitivity test, not a final physical bound.\n\n",
        "\\begin{figure}[ht]\n",
        "\\centering\n",
        "\\includegraphics[width=\\linewidth]{figures/ballistic_coefficient_sweep.png}\n",
        "\\caption{Residual growth after refitting the six initial-state components at each fixed $C_D A/m$.  "
        f"{threshold_caption}}}\n",
        "\\end{figure}\n\n",
        "\\begin{table}[ht]\n",
        "\\centering\n",
        "\\caption{Residuals for fixed-$C_D A/m$ refits in the ballistic-coefficient sweep.}\n",
        "\\input{ballistic_coefficient_sweep_table.tex}\n",
        "\\end{table}\n",
        "\\end{document}\n",
    ]
    REPORT_TEX.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-json", type=Path, default=FIT_JSON)
    parser.add_argument("--head-selection-csv", type=Path, default=DEFAULT_HEAD_SELECTION)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--trail-json", type=Path, default=DEFAULT_TRAIL)
    parser.add_argument("--trail-min-snr-db", type=float, default=8.0)
    parser.add_argument("--max-aod-offset-deg", type=float, default=35.0)
    parser.add_argument("--min-trail-alt-km", type=float, default=20.0)
    parser.add_argument("--max-trail-alt-km", type=float, default=115.0)
    parser.add_argument("--model-padding-s", type=float, default=0.8)
    parser.add_argument("--model-dt-s", type=float, default=0.10)
    parser.add_argument("--head-coherent-codes", type=int, default=16)
    parser.add_argument("--trail-coherent-codes", type=int, default=8)
    parser.add_argument("--code-period-s", type=float, default=0.01)
    parser.add_argument("--trail-correlation-range-gate-m", type=float, default=3000.0)
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=1000.0)
    parser.add_argument("--n-scales", type=int, default=31)
    parser.add_argument("--rms-growth-threshold", type=float, default=0.10)
    parser.add_argument("--max-nfev", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fit_payload = json.loads(args.fit_json.read_text(encoding="utf-8"))
    fit_args, obs, trail_rows, t_ref, t_model, density_profile, chol, cov_meta = load_measurements(args, fit_payload)
    base_params = np.asarray(fit_payload["fit_parameters"], dtype=float)
    base_state = base_params[:6].copy()
    nominal_cd_area_over_mass = float(fit_payload["cd_area_over_mass_m2_kg"])
    rows = []
    scales = np.geomspace(args.scale_min, args.scale_max, args.n_scales)
    q0 = np.zeros(6, dtype=float)
    for scale in scales:
        cd_area_over_mass = nominal_cd_area_over_mass * float(scale)
        log_drag = math.log10(cd_area_over_mass)
        res = refit_state_for_drag(
            q0,
            base_state,
            log_drag,
            obs,
            trail_rows,
            t_ref,
            t_model,
            density_profile,
            fit_args,
            chol,
            args.max_nfev,
        )
        q0 = res.x.copy()
        params = np.r_[base_state + res.x, log_drag]
        raw = raw_residuals(params, obs, trail_rows, t_ref, t_model, density_profile, fit_args)
        row = {
            "scale": float(scale),
            "cd_area_over_mass": float(cd_area_over_mass),
            "success": bool(res.success),
            "nfev": int(res.nfev),
            "state_delta_pos_km": float(np.linalg.norm(res.x[:3]) / 1000.0),
            "state_delta_vel_mps": float(np.linalg.norm(res.x[3:6])),
            **summarize_residuals(raw, obs, trail_rows, chol),
        }
        diameter_m, mass_kg = sphere_mass_from_cd_area_over_mass(cd_area_over_mass)
        row["diameter_m"] = float(diameter_m)
        row["mass_kg"] = float(mass_kg)
        rows.append(row)
    df = pd.DataFrame(rows)
    threshold, threshold_found = threshold_row(df, args.rms_growth_threshold)
    last_ok = last_unflagged_row(df, threshold, threshold_found)
    df.to_csv(OUT_DIR / "ballistic_coefficient_sweep.csv", index=False)
    plot_sweep(df, threshold, threshold_found, FIG_DIR / "ballistic_coefficient_sweep.png")
    write_report(df, threshold, last_ok, threshold_found, fit_payload, args)
    (OUT_DIR / "ballistic_coefficient_sweep_summary.json").write_text(
        json.dumps(
            {
                "fit_json": str(args.fit_json.relative_to(ROOT)),
                "trajectory_fit_version": fit_payload.get("trajectory_fit_version", "unspecified"),
                "n_head_observations": len(obs),
                "n_trail_positions": len(trail_rows),
                "covariance": cov_meta,
                "rms_growth_threshold": args.rms_growth_threshold,
                "threshold_found": bool(threshold_found),
                "threshold_scale": float(threshold["scale"]),
                "threshold_cd_area_over_mass_m2_kg": float(threshold["cd_area_over_mass"]),
                "threshold_diameter_m": float(threshold["diameter_m"]),
                "threshold_mass_kg": float(threshold["mass_kg"]),
                "largest_unflagged_scale": float(last_ok["scale"]),
                "largest_unflagged_cd_area_over_mass_m2_kg": float(last_ok["cd_area_over_mass"]),
                "largest_unflagged_diameter_m": float(last_ok["diameter_m"]),
                "largest_unflagged_mass_kg": float(last_ok["mass_kg"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(REPORT_TEX)
    print(FIG_DIR / "ballistic_coefficient_sweep.png")
    print(OUT_DIR / "ballistic_coefficient_sweep.csv")
    print(
        "threshold: "
        f"found={threshold_found}, "
        f"scale={threshold['scale']:.3g}, "
        f"CdA/m={threshold['cd_area_over_mass']:.3e} m^2/kg, "
        f"diameter={threshold['diameter_m']:.2f} m, "
        f"mass={threshold['mass_kg']:.2g} kg; "
        f"last_unflagged_mass={last_ok['mass_kg']:.2g} kg"
    )


if __name__ == "__main__":
    main()
