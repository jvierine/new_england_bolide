#!/usr/bin/env python3
"""Interactive matplotlib selector for head-echo range/Doppler measurements.

Keyboard shortcuts:
  r  switch brush mode to real
  x  switch brush mode to reject
  u  switch brush mode to unknown
  c  clear current selection
  a  select all points in current station
  n  next station
  p  previous station
  s  save JSON/CSV
  q  save and close

Use the lasso tool on either panel, or click near a point, to apply the current
brush mode. Classifications are saved to disk immediately after every edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, LassoSelector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "reports" / "head_echo_gaussian_fit_summary_retained_ext0p5.csv"
DEFAULT_OUT_DIR = ROOT / "reports" / "head_echo_python_selection_ext0p5"

STATION_LABELS = {
    "beverly": "Beverly, MA",
    "clinton": "Clinton, MA",
    "pomfret": "Pomfret, VT",
}

STATION_COLORS = {
    "beverly": "tab:blue",
    "clinton": "tab:red",
    "pomfret": "tab:green",
}

STATUS_COLORS = {
    "unknown": "0.55",
    "real": "tab:green",
    "reject": "tab:red",
}


def load_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["success"].astype(str).str.lower() == "true"].copy()
    numeric = [
        "idx",
        "time_unix",
        "time_rel_s",
        "fit_range_m",
        "seed_range_m",
        "fit_doppler_hz",
        "seed_doppler_hz",
        "sigma_range_m",
        "sigma_doppler_hz",
        "amplitude_db",
        "cost",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_rel_s", "fit_range_m", "fit_doppler_hz"]).copy()
    df["idx"] = df["idx"].astype(int)
    df["fit_range_km"] = df["fit_range_m"] / 1000.0
    df["seed_range_km"] = df["seed_range_m"] / 1000.0
    df["key"] = df["station"].astype(str) + ":" + df["idx"].astype(str)
    return df.sort_values(["station", "time_rel_s", "idx"]).reset_index(drop=True)


def default_state(df: pd.DataFrame) -> dict[str, str]:
    return {key: "unknown" for key in df["key"]}


def load_state(path: Path, df: pd.DataFrame) -> dict[str, str]:
    state = default_state(df)
    if not path.exists():
        return state
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "selections" in data:
        data = data["selections"]
    if isinstance(data, list):
        for row in data:
            key = f"{row['station']}:{int(row['idx'])}"
            sel = str(row.get("selection", "unknown")).lower()
            if key in state and sel in {"unknown", "real", "reject"}:
                state[key] = sel
    elif isinstance(data, dict):
        for key, sel in data.items():
            sel = str(sel).lower()
            if key in state and sel in {"unknown", "real", "reject"}:
                state[key] = sel
    return state


class HeadEchoSelector:
    def __init__(self, df: pd.DataFrame, state: dict[str, str], out_dir: Path):
        self.df = df
        self.state = state
        self.out_dir = out_dir
        self.stations = list(dict.fromkeys(df["station"].tolist()))
        self.station_i = 0
        self.selected: set[str] = set()
        self.brush_mode = "real"
        self.scatters = []

        self.fig, self.axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
        plt.subplots_adjust(bottom=0.17, hspace=0.22)
        self.axes[0].set_ylabel("Bistatic range (km)")
        self.axes[1].set_ylabel("Doppler (Hz)")
        self.axes[1].set_xlabel("Seconds after 18:06:20 UTC")
        for ax in self.axes:
            ax.grid(True, color="0.86", linewidth=0.8)

        self.lassos = [LassoSelector(ax, onselect=lambda verts, axis_i=i: self.on_lasso(verts, axis_i)) for i, ax in enumerate(self.axes)]
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.make_buttons()
        self.draw()

    @property
    def station(self) -> str:
        return self.stations[self.station_i]

    def station_df(self) -> pd.DataFrame:
        return self.df[self.df["station"] == self.station].copy()

    def make_buttons(self) -> None:
        specs = [
            ("Brush Real (r)", self.brush_real),
            ("Brush Reject (x)", self.brush_reject),
            ("Brush Unknown (u)", self.brush_unknown),
            ("Clear (c)", self.clear_selection),
            ("Apply All (a)", self.apply_all_station),
            ("Prev (p)", self.prev_station),
            ("Next (n)", self.next_station),
            ("Save (s)", self.save),
        ]
        x0 = 0.055
        width = 0.105
        for i, (label, callback) in enumerate(specs):
            ax = self.fig.add_axes([x0 + i * (width + 0.006), 0.045, width, 0.055])
            btn = Button(ax, label)
            btn.on_clicked(lambda _event, cb=callback: cb())
            setattr(self, f"_button_{i}", btn)

    def draw(self) -> None:
        for ax in self.axes:
            ax.cla()
            ax.grid(True, color="0.86", linewidth=0.8)
        self.axes[0].set_ylabel("Bistatic range (km)")
        self.axes[1].set_ylabel("Doppler (Hz)")
        self.axes[1].set_xlabel("Seconds after 18:06:20 UTC")

        sdf = self.station_df()
        station_color = STATION_COLORS.get(self.station, "tab:blue")
        self.scatters = []
        for status in ["unknown", "real", "reject"]:
            ssdf = sdf[[self.state[k] == status for k in sdf["key"]]]
            if ssdf.empty:
                continue
            edge = [("gold" if key in self.selected else "white") for key in ssdf["key"]]
            linewidth = [(2.2 if key in self.selected else 0.4) for key in ssdf["key"]]
            size = [(76 if key in self.selected else 34) for key in ssdf["key"]]
            color = station_color if status == "unknown" else STATUS_COLORS[status]
            alpha = 0.55 if status == "unknown" else 0.90
            sc0 = self.axes[0].scatter(
                ssdf["time_rel_s"],
                ssdf["fit_range_km"],
                s=size,
                c=color,
                alpha=alpha,
                edgecolors=edge,
                linewidths=linewidth,
                label=f"{status} ({len(ssdf)})",
                zorder=3,
            )
            sc1 = self.axes[1].scatter(
                ssdf["time_rel_s"],
                ssdf["fit_doppler_hz"],
                s=size,
                c=color,
                alpha=alpha,
                edgecolors=edge,
                linewidths=linewidth,
                zorder=3,
            )
            self.scatters.append((sc0, ssdf["key"].tolist(), 0))
            self.scatters.append((sc1, ssdf["key"].tolist(), 1))

        self.axes[0].scatter(sdf["time_rel_s"], sdf["seed_range_km"], s=42, facecolors="none", edgecolors="black", linewidths=0.9, alpha=0.55, label="original seed", zorder=4)
        self.axes[1].scatter(sdf["time_rel_s"], sdf["seed_doppler_hz"], s=42, facecolors="none", edgecolors="black", linewidths=0.9, alpha=0.55, zorder=4)

        counts = {status: sum(self.state[k] == status for k in sdf["key"]) for status in ["unknown", "real", "reject"]}
        self.axes[0].set_title(
            f"{STATION_LABELS.get(self.station, self.station)}: "
            f"unknown={counts['unknown']} real={counts['real']} reject={counts['reject']} "
            f"brush={self.brush_mode} selected={len(self.selected)}"
        )
        self.axes[1].set_title("Click/lasso paints current brush; r=real, x=reject, u=unknown, s=save, n/p=station")
        self.axes[0].legend(loc="best", framealpha=0.90)
        self.fig.canvas.draw_idle()

    def points_for_axis(self, axis_i: int) -> tuple[np.ndarray, list[str]]:
        sdf = self.station_df()
        if axis_i == 0:
            pts = np.column_stack([sdf["time_rel_s"].to_numpy(), sdf["fit_range_km"].to_numpy()])
        else:
            pts = np.column_stack([sdf["time_rel_s"].to_numpy(), sdf["fit_doppler_hz"].to_numpy()])
        return pts, sdf["key"].tolist()

    def on_lasso(self, verts, axis_i: int) -> None:
        pts, keys = self.points_for_axis(axis_i)
        path = MplPath(verts)
        mask = path.contains_points(pts)
        touched = [key for key, keep in zip(keys, mask) if keep]
        self.apply_status(touched, self.brush_mode)

    def on_click(self, event) -> None:
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        axis_i = 0 if event.inaxes == self.axes[0] else 1
        sdf = self.station_df()
        if axis_i == 0:
            x = sdf["time_rel_s"].to_numpy()
            y = sdf["fit_range_km"].to_numpy()
        else:
            x = sdf["time_rel_s"].to_numpy()
            y = sdf["fit_doppler_hz"].to_numpy()
        display = event.inaxes.transData.transform(np.column_stack([x, y]))
        dist = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
        if len(dist) == 0 or float(np.min(dist)) > 12.0:
            return
        key = sdf.iloc[int(np.argmin(dist))]["key"]
        status = self.brush_mode
        if event.button == 3:
            status = "reject"
        elif event.button == 2:
            status = "unknown"
        self.apply_status([key], status)

    def apply_status(self, keys: list[str], status: str) -> None:
        self.selected = set(keys)
        for key in keys:
            self.state[key] = status
        self.save()
        self.draw()

    def brush_real(self) -> None:
        self.brush_mode = "real"
        self.draw()

    def brush_reject(self) -> None:
        self.brush_mode = "reject"
        self.draw()

    def brush_unknown(self) -> None:
        self.brush_mode = "unknown"
        self.draw()

    def clear_selection(self) -> None:
        self.selected.clear()
        self.draw()

    def apply_all_station(self) -> None:
        self.apply_status(list(self.station_df()["key"]), self.brush_mode)

    def next_station(self) -> None:
        self.station_i = (self.station_i + 1) % len(self.stations)
        self.selected.clear()
        self.draw()

    def prev_station(self) -> None:
        self.station_i = (self.station_i - 1) % len(self.stations)
        self.selected.clear()
        self.draw()

    def on_key(self, event) -> None:
        if event.key == "r":
            self.brush_real()
        elif event.key == "x":
            self.brush_reject()
        elif event.key == "u":
            self.brush_unknown()
        elif event.key == "c":
            self.clear_selection()
        elif event.key == "a":
            self.apply_all_station()
        elif event.key == "n":
            self.next_station()
        elif event.key == "p":
            self.prev_station()
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in self.df.to_dict(orient="records"):
            rows.append(
                {
                    "station": row["station"],
                    "idx": int(row["idx"]),
                    "time_unix": float(row["time_unix"]),
                    "time_rel_s": float(row["time_rel_s"]),
                    "fit_range_m": float(row["fit_range_m"]),
                    "fit_doppler_hz": float(row["fit_doppler_hz"]),
                    "sigma_range_m": float(row["sigma_range_m"]),
                    "sigma_doppler_hz": float(row["sigma_doppler_hz"]),
                    "amplitude_db": float(row["amplitude_db"]),
                    "cost": float(row["cost"]),
                    "selection": self.state[row["key"]],
                    "real": self.state[row["key"]] == "real",
                }
            )
        json_path = self.out_dir / "head_echo_python_measurement_selection.json"
        csv_path = self.out_dir / "head_echo_python_measurement_selection.csv"
        state_path = self.out_dir / "head_echo_python_selection_state.json"
        json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        state_path.write_text(json.dumps({"selections": self.state}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {json_path}")
        print(f"Saved {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--state", type=Path, default=None)
    args = p.parse_args()

    df = load_measurements(args.summary)
    state_path = args.state if args.state is not None else args.out_dir / "head_echo_python_selection_state.json"
    state = load_state(state_path, df)
    selector = HeadEchoSelector(df, state, args.out_dir)
    plt.show()
    selector.save()


if __name__ == "__main__":
    main()
