#!/usr/bin/env python3
"""Generate an interactive head-echo range/Doppler time-series plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_retained_ext0p5.csv"
DEFAULT_OUT = ROOT / "reports" / "figures" / "head_echo_range_doppler_interactive.html"

STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, CT",
}

STATION_COLORS = {
    "beverly": "#1f77b4",
    "clinton": "#d62728",
    "pomfret": "#2ca02c",
}


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["success"].astype(str).str.lower() == "true"].copy()
    numeric = [
        "idx",
        "time_rel_s",
        "fit_range_m",
        "seed_range_m",
        "range_correction_m",
        "fit_doppler_hz",
        "seed_doppler_hz",
        "doppler_correction_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "amplitude_db",
        "cost",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fit_range_km"] = df["fit_range_m"] / 1000.0
    df["sigma_range_km"] = df["sigma_range_m"] / 1000.0
    df = df.sort_values(["station", "time_rel_s", "idx"])
    return df


def make_figure(df: pd.DataFrame, show_error_bars: bool) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Fitted bistatic range", "Fitted Doppler"),
    )

    for station, sdf in df.groupby("station", sort=False):
        label = STATION_LABELS.get(station, station)
        color = STATION_COLORS.get(station)
        customdata = sdf[
            [
                "idx",
                "seed_range_m",
                "range_correction_m",
                "seed_doppler_hz",
                "doppler_correction_hz",
                "amplitude_db",
                "cost",
            ]
        ].to_numpy()

        range_error = None
        doppler_error = None
        if show_error_bars:
            range_error = {"type": "data", "array": sdf["sigma_range_km"], "visible": True, "thickness": 0.8}
            doppler_error = {"type": "data", "array": sdf["sigma_doppler_hz"], "visible": True, "thickness": 0.8}

        fig.add_trace(
            go.Scattergl(
                x=sdf["time_rel_s"],
                y=sdf["fit_range_km"],
                mode="markers+lines",
                name=label,
                legendgroup=station,
                marker={"size": 6, "color": color},
                line={"width": 1.4, "color": color},
                error_y=range_error,
                customdata=customdata,
                hovertemplate=(
                    f"{label}<br>"
                    "t=%{x:.3f} s<br>"
                    "range=%{y:.3f} km<br>"
                    "seed range=%{customdata[1]:.1f} m<br>"
                    "range correction=%{customdata[2]:+.1f} m<br>"
                    "amp=%{customdata[5]:.2f} dB<br>"
                    "cost=%{customdata[6]:.3f}<br>"
                    "idx=%{customdata[0]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=sdf["time_rel_s"],
                y=sdf["fit_doppler_hz"],
                mode="markers+lines",
                name=label,
                legendgroup=station,
                showlegend=False,
                marker={"size": 6, "color": color},
                line={"width": 1.4, "color": color},
                error_y=doppler_error,
                customdata=customdata,
                hovertemplate=(
                    f"{label}<br>"
                    "t=%{x:.3f} s<br>"
                    "Doppler=%{y:.2f} Hz<br>"
                    "seed Doppler=%{customdata[3]:.2f} Hz<br>"
                    "Doppler correction=%{customdata[4]:+.2f} Hz<br>"
                    "amp=%{customdata[5]:.2f} dB<br>"
                    "cost=%{customdata[6]:.3f}<br>"
                    "idx=%{customdata[0]:.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Seconds after 18:06:20 UTC", row=2, col=1)
    fig.update_yaxes(title_text="Bistatic range (km)", row=1, col=1)
    fig.update_yaxes(title_text="Doppler (Hz)", row=2, col=1)
    fig.update_layout(
        title="Head-echo Gaussian range-Doppler measurements",
        template="plotly_white",
        hovermode="closest",
        height=820,
        width=1100,
        legend_title_text="Receiver station",
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-error-bars", action="store_true")
    args = p.parse_args()

    df = load_rows(args.summary)
    fig = make_figure(df, show_error_bars=not args.no_error_bars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn", full_html=True)
    print(args.out)
    print(f"{len(df)} successful fitted head-echo measurements")


if __name__ == "__main__":
    main()
