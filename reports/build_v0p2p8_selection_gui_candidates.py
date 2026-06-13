#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "head_echo_python_selection_v0p2p8_highsnr"
PREVIOUS_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_v0p2p7_1coh_2incoh_zp16_sparse_dcmean.csv"
SELECTION = OUT_DIR / "head_echo_python_measurement_selection.csv"
OUTPUT = OUT_DIR / "head_echo_measurement_candidates.csv"


def main() -> None:
    prev_summary = pd.read_csv(PREVIOUS_SUMMARY)
    selection = pd.read_csv(SELECTION)
    keep = prev_summary[prev_summary["station"].isin(["beverly", "clinton"])].copy()
    pomfret = selection[selection["station"] == "pomfret"].copy()
    cols = list(prev_summary.columns)

    rows = []
    for row_in in pomfret.itertuples(index=False):
        row = {col: pd.NA for col in cols}
        row.update(
            {
                "station": row_in.station,
                "idx": int(row_in.idx),
                "time_unix": float(row_in.time_unix),
                "time_rel_s": float(row_in.time_rel_s),
                "patch_time_unix": float(row_in.time_unix),
                "patch_time_rel_s": float(row_in.time_rel_s),
                "seed_range_m": float(row_in.fit_range_m),
                "fit_range_m": float(row_in.fit_range_m),
                "range_correction_m": 0.0,
                "seed_doppler_hz": float(row_in.fit_doppler_hz),
                "fit_doppler_hz": float(row_in.fit_doppler_hz),
                "doppler_correction_hz": 0.0,
                "model_doppler_hz": float(row_in.fit_doppler_hz),
                "fit_rel_doppler_hz": float(row_in.fit_doppler_hz),
                "sigma_range_m": float(row_in.sigma_range_m),
                "sigma_doppler_hz": float(row_in.sigma_doppler_hz),
                "range_doppler_correlation": 0.0,
                "amplitude_db": float(row_in.amplitude_db),
                "snr_db": float(row_in.amplitude_db),
                "cost": float(row_in.cost),
                "success": True,
                "error": "",
                "n_coh": 1,
                "n_incoh": 2,
                "step_codes": 1,
                "code_period_s": 0.01,
                "figure": "",
            }
        )
        rows.append(row)

    pomfret_summary = pd.DataFrame(rows, columns=cols)
    candidates = (
        pd.concat([keep[cols], pomfret_summary], ignore_index=True)
        .sort_values(["station", "time_unix", "idx"])
        .reset_index(drop=True)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT, index=False)
    print(OUTPUT)
    print(candidates.groupby("station").size())


if __name__ == "__main__":
    main()
