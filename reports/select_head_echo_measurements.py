#!/usr/bin/env python3
"""Interactive matplotlib selector for head-echo range/Doppler measurements.

Keyboard shortcuts:
  r  switch brush mode to real
  x  switch brush mode to reject
  u  switch brush mode to unknown
  c  clear all selections in current station
  n  next station
  p  previous station
  s  save JSON/CSV
  q  save and close

Use the lasso tool on either panel, or click near a point, to apply the current
brush mode. Painting the range panel selects range measurements; painting the
Doppler panel selects Doppler measurements. Classifications are saved to disk
immediately after every edit.
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
    "unknown": "0.45",
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


def default_state(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {key: {"range": "unknown", "doppler": "unknown"} for key in df["key"]}


def normalize_status(value: object) -> str:
    status = str(value).lower()
    return status if status in {"unknown", "real", "reject"} else "unknown"


def load_state(path: Path, df: pd.DataFrame) -> dict[str, dict[str, str]]:
    state = default_state(df)
    if not path.exists():
        return state
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "selections" in data:
        data = data["selections"]
    if isinstance(data, list):
        for row in data:
            key = f"{row['station']}:{int(row['idx'])}"
            if key not in state:
                continue
            if "selection_range" in row or "selection_doppler" in row:
                state[key]["range"] = normalize_status(row.get("selection_range", "unknown"))
                state[key]["doppler"] = normalize_status(row.get("selection_doppler", "unknown"))
            else:
                sel = normalize_status(row.get("selection", "unknown"))
                state[key] = {"range": sel, "doppler": sel}
    elif isinstance(data, dict):
        for key, sel in data.items():
            if key not in state:
                continue
            if isinstance(sel, dict):
                state[key]["range"] = normalize_status(sel.get("range", "unknown"))
                state[key]["doppler"] = normalize_status(sel.get("doppler", "unknown"))
            else:
                status = normalize_status(sel)
                state[key] = {"range": status, "doppler": status}
    return state


def load_reference_measurements(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "real_range" not in df.columns:
        df["real_range"] = df.get("selection", "unknown").astype(str).str.lower() == "real"
    if "real_doppler" not in df.columns:
        df["real_doppler"] = df.get("selection", "unknown").astype(str).str.lower() == "real"
    df["real_range"] = df["real_range"].astype(str).str.lower().isin({"true", "1", "yes", "real"})
    df["real_doppler"] = df["real_doppler"].astype(str).str.lower().isin({"true", "1", "yes", "real"})
    for col in ("time_rel_s", "fit_range_m", "fit_doppler_hz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_rel_s"]).copy()
    df["fit_range_km"] = df["fit_range_m"] / 1000.0
    return df


class HeadEchoSelector:
    def __init__(self, df: pd.DataFrame, state: dict[str, dict[str, str]], out_dir: Path, reference_df: pd.DataFrame | None = None):
        self.df = df
        self.state = state
        self.out_dir = out_dir
        self.reference_df = reference_df
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
            ("Clear All (c)", self.clear_all_station),
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

    def draw(self, preserve_limits: bool = False) -> None:
        old_limits = None
        if preserve_limits and all(ax.has_data() for ax in self.axes):
            old_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self.axes]
        for ax in self.axes:
            ax.cla()
            ax.grid(True, color="0.86", linewidth=0.8)
        self.axes[0].set_ylabel("Bistatic range (km)")
        self.axes[1].set_ylabel("Doppler (Hz)")
        self.axes[1].set_xlabel("Seconds after 18:06:20 UTC")

        sdf = self.station_df()
        ref_range_n = 0
        ref_doppler_n = 0
        if self.reference_df is not None:
            ref = self.reference_df[self.reference_df["station"] == self.station]
            ref_r = ref[ref["real_range"]]
            ref_range_n = len(ref_r)
            if not ref_r.empty:
                self.axes[0].scatter(
                    ref_r["time_rel_s"],
                    ref_r["fit_range_km"],
                    s=72,
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    alpha=0.85,
                    linewidths=1.35,
                    label=f"v0.2.7 prior ({len(ref_r)})",
                    zorder=6,
                )
            ref_d = ref[ref["real_doppler"]]
            ref_doppler_n = len(ref_d)
            if not ref_d.empty:
                self.axes[1].scatter(
                    ref_d["time_rel_s"],
                    ref_d["fit_doppler_hz"],
                    s=72,
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    alpha=0.85,
                    linewidths=1.35,
                    zorder=6,
                )
        self.scatters = []
        for status in ["unknown", "real", "reject"]:
            ssdf = sdf[[self.state[k]["range"] == status for k in sdf["key"]]]
            color = STATUS_COLORS[status]
            alpha = 0.55 if status == "unknown" else 0.90
            if not ssdf.empty:
                edge = [("gold" if key in self.selected else "white") for key in ssdf["key"]]
                linewidth = [(2.2 if key in self.selected else 0.4) for key in ssdf["key"]]
                size = [(76 if key in self.selected else 34) for key in ssdf["key"]]
                sc0 = self.axes[0].scatter(
                    ssdf["time_rel_s"],
                    ssdf["fit_range_km"],
                    s=size,
                    c=color,
                    alpha=0.82 if status == "unknown" else alpha,
                    edgecolors=edge,
                    linewidths=linewidth,
                    label=f"range {status} ({len(ssdf)})",
                    zorder=5,
                )
                self.scatters.append((sc0, ssdf["key"].tolist(), 0))

            ssdf_dop = sdf[[self.state[k]["doppler"] == status for k in sdf["key"]]]
            if ssdf_dop.empty:
                continue
            edge = [("gold" if key in self.selected else "white") for key in ssdf_dop["key"]]
            linewidth = [(2.2 if key in self.selected else 0.4) for key in ssdf_dop["key"]]
            size = [(76 if key in self.selected else 34) for key in ssdf_dop["key"]]
            sc1 = self.axes[1].scatter(
                ssdf_dop["time_rel_s"],
                ssdf_dop["fit_doppler_hz"],
                s=size,
                c=color,
                alpha=0.82 if status == "unknown" else alpha,
                edgecolors=edge,
                linewidths=linewidth,
                zorder=5,
            )
            self.scatters.append((sc1, ssdf_dop["key"].tolist(), 1))

        model = sdf.sort_values("time_rel_s")
        self.axes[0].plot(
            model["time_rel_s"],
            model["seed_range_km"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
            label="trajectory estimate",
            zorder=4,
        )
        self.axes[1].plot(
            model["time_rel_s"],
            model["seed_doppler_hz"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
            zorder=4,
        )

        range_counts = {status: sum(self.state[k]["range"] == status for k in sdf["key"]) for status in ["unknown", "real", "reject"]}
        doppler_counts = {status: sum(self.state[k]["doppler"] == status for k in sdf["key"]) for status in ["unknown", "real", "reject"]}
        self.axes[0].set_title(
            f"{STATION_LABELS.get(self.station, self.station)}: "
            f"range real={range_counts['real']} reject={range_counts['reject']} unknown={range_counts['unknown']} "
            f"ref={ref_range_n} brush={self.brush_mode} selected={len(self.selected)}"
        )
        self.axes[1].set_title(
            f"doppler real={doppler_counts['real']} reject={doppler_counts['reject']} unknown={doppler_counts['unknown']} "
            f"ref={ref_doppler_n} paint range panel for ranges; paint Doppler panel for Dopplers"
        )
        self.axes[0].legend(loc="best", framealpha=0.90)
        if old_limits is not None:
            for ax, (xlim, ylim) in zip(self.axes, old_limits, strict=True):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        else:
            self.set_default_limits(sdf)
        self.fig.canvas.draw_idle()

    def set_default_limits(self, sdf: pd.DataFrame) -> None:
        ref = None if self.reference_df is None else self.reference_df[self.reference_df["station"] == self.station]
        if ref is not None and not ref.empty:
            ref_used = ref[ref["real_range"] | ref["real_doppler"]]
            xvals = ref_used["time_rel_s"].to_numpy(dtype=float)
            xvals = xvals[np.isfinite(xvals)]
        else:
            xvals = np.array([], dtype=float)
        if xvals.size:
            xmin = float(np.nanmin(xvals)) - 0.5
            xmax = float(np.nanmax(xvals)) + 0.5
        else:
            xmin = float(np.nanpercentile(sdf["time_rel_s"], 5.0))
            xmax = float(np.nanpercentile(sdf["time_rel_s"], 95.0))
        if xmin >= xmax:
            xmin -= 0.5
            xmax += 0.5
        for ax in self.axes:
            ax.set_xlim(xmin, xmax)

        in_window = (sdf["time_rel_s"] >= xmin) & (sdf["time_rel_s"] <= xmax)
        range_vals = [sdf.loc[in_window, "fit_range_km"].to_numpy(dtype=float)]
        doppler_vals = [sdf.loc[in_window, "fit_doppler_hz"].to_numpy(dtype=float)]
        if ref is not None and not ref.empty:
            ref_window = (ref["time_rel_s"] >= xmin) & (ref["time_rel_s"] <= xmax)
            range_vals.append(ref.loc[ref_window & ref["real_range"], "fit_range_km"].to_numpy(dtype=float))
            doppler_vals.append(ref.loc[ref_window & ref["real_doppler"], "fit_doppler_hz"].to_numpy(dtype=float))
        self.set_robust_ylim(self.axes[0], np.concatenate(range_vals))
        self.set_robust_ylim(self.axes[1], np.concatenate(doppler_vals))

    @staticmethod
    def set_robust_ylim(ax, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        lo = float(np.nanpercentile(values, 2.0))
        hi = float(np.nanpercentile(values, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if lo == hi:
            pad = max(1.0, abs(lo) * 0.02)
        else:
            pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

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
        self.apply_status(touched, self.brush_mode, axis_i)

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
        self.apply_status([key], status, axis_i)

    def apply_status(self, keys: list[str], status: str, axis_i: int) -> None:
        self.selected = set(keys)
        measurement = "range" if axis_i == 0 else "doppler"
        for key in keys:
            self.state[key][measurement] = status
        self.save()
        self.draw(preserve_limits=True)

    def brush_real(self) -> None:
        self.brush_mode = "real"
        self.draw(preserve_limits=True)

    def brush_reject(self) -> None:
        self.brush_mode = "reject"
        self.draw(preserve_limits=True)

    def brush_unknown(self) -> None:
        self.brush_mode = "unknown"
        self.draw(preserve_limits=True)

    def clear_all_station(self) -> None:
        for key in self.station_df()["key"]:
            self.state[key]["range"] = "unknown"
            self.state[key]["doppler"] = "unknown"
        self.selected.clear()
        self.save()
        self.draw(preserve_limits=True)

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
            self.clear_all_station()
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
            range_selection = self.state[row["key"]]["range"]
            doppler_selection = self.state[row["key"]]["doppler"]
            both_selection = range_selection if range_selection == doppler_selection else "mixed"
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
                    "selection": both_selection,
                    "selection_range": range_selection,
                    "selection_doppler": doppler_selection,
                    "real": range_selection == "real" and doppler_selection == "real",
                    "real_range": range_selection == "real",
                    "real_doppler": doppler_selection == "real",
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
    p.add_argument("--reference-selection", type=Path, default=None)
    args = p.parse_args()

    df = load_measurements(args.summary)
    state_path = args.state if args.state is not None else args.out_dir / "head_echo_python_selection_state.json"
    state = load_state(state_path, df)
    reference_df = load_reference_measurements(args.reference_selection)
    selector = HeadEchoSelector(df, state, args.out_dir, reference_df=reference_df)
    plt.show()
    selector.save()


if __name__ == "__main__":
    main()
