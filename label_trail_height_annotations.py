#!/usr/bin/env python3
"""Interactively place height labels for the trail echo report figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from station_config import short_station_name
from trail_echo_roi import (
    DEFAULT_OUT,
    DEFAULT_HEIGHT_LABELS,
    DEFAULT_MERGED_DIR,
    DEFAULT_TRAJECTORY_JSON,
    HEIGHT_LABELS_KM,
    REPORT_FONT_SCALE,
    REPORT_PLOT_START_REL_S,
    REPORT_RANGE_WINDOWS_KM,
    height_label_positions,
    iso_utc,
    load_head_echo_range_tracks,
    load_height_label_annotations,
    load_merged,
)


def default_payload():
    return {
        "schema": "new_england_bolide.trail_height_annotations.v1",
        "description": "Manual text positions for height labels in reports/figures/trail_echo_roi.png.",
        "stations": {},
    }


def load_payload(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default_payload()


def save_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class HeightLabelEditor:
    def __init__(self, args):
        self.args = args
        self.datasets = [load_merged(args.merged_dir, station) for station in args.stations]
        self.t_ref = min(float(data["t_unix"][0]) for data in self.datasets)
        self.t_max = max(float(data["t_unix"][-1]) for data in self.datasets) - self.t_ref
        self.tracks = load_head_echo_range_tracks(self.datasets, args.trajectory_json)
        for data in self.datasets:
            station = data["station"]
            if station in self.tracks:
                self.tracks[station]["t_rel"] = data["t_unix"] - self.t_ref

        self.payload = load_payload(args.out)
        self.saved_labels = load_height_label_annotations(args.out)
        self.rois = json.loads(Path(args.rois).read_text(encoding="utf-8")).get("stations", {}) if Path(args.rois).exists() else {}
        self.artists = {}
        self.selected = None
        self.dragging = False

        self.fig, self.axes = plt.subplots(
            len(self.datasets),
            1,
            figsize=(12.0, max(2.15 * len(self.datasets), 8.5)),
            sharex=True,
            constrained_layout=True,
        )
        if len(self.datasets) == 1:
            self.axes = np.array([self.axes])
        self.fig.canvas.manager.set_window_title("Trail height annotation editor")
        self.draw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        save_ax = self.fig.add_axes([0.82, 0.01, 0.08, 0.035])
        reset_ax = self.fig.add_axes([0.91, 0.01, 0.08, 0.035])
        self.save_button = Button(save_ax, "Save")
        self.reset_button = Button(reset_ax, "Reset")
        self.save_button.on_clicked(lambda _event: self.save())
        self.reset_button.on_clicked(lambda _event: self.reset_defaults())

    def draw(self):
        axis_label_size = 10 * REPORT_FONT_SCALE
        tick_label_size = 10 * REPORT_FONT_SCALE
        label_size = 8.0 * REPORT_FONT_SCALE
        snr_all = []
        for data in self.datasets:
            snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
            snr_all.append(10.0 * np.log10(np.maximum(snr, 1e-12)).ravel())
        snr_all = np.concatenate(snr_all)
        vmin = max(-5.0, float(np.nanpercentile(snr_all, 5)))
        vmax = min(30.0, float(np.nanpercentile(snr_all, 99.7)))

        for ax, data in zip(self.axes, self.datasets):
            station = data["station"]
            t_rel = data["t_unix"] - self.t_ref
            range_km = data["range_km"]
            snr = data["P_rti"] / np.maximum(data["N"], 1e-12)
            snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
            ax.pcolormesh(t_rel, range_km, snr_db.T, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            if station in self.tracks:
                track = self.tracks[station]
                plotted_range = np.asarray(track["range_km"], dtype=float).copy()
                plotted_range[np.asarray(track["alt_km"], dtype=float) < 0.0] = np.nan
                ax.plot(t_rel, plotted_range, color="white", linestyle="--", linewidth=1.2, alpha=0.3, zorder=5)
            ax.set_ylabel(f"{short_station_name(station)}\nrange (km)", fontsize=axis_label_size)
            ax.tick_params(axis="both", labelsize=tick_label_size)
            roi = self.rois.get(station)
            if roi:
                y0 = roi["range_km"]["min"]
                y1 = roi["range_km"]["max"]
                ax.set_ylim(max(float(range_km[0]), y0 - 8.0), min(float(range_km[-1]), y1 + 8.0))
            else:
                r0, r1 = REPORT_RANGE_WINDOWS_KM[station]
                ax.set_ylim(r0, r1)
            ax.grid(True, color="0.85", lw=0.4)
            ax.set_xlim(REPORT_PLOT_START_REL_S, self.t_max)

        self.axes[-1].set_xlabel(f"Seconds after {iso_utc(self.t_ref)}", fontsize=axis_label_size)
        self.fig.suptitle("Drag labels. Press s to save, r to reset defaults, q to close.", fontsize=13)
        self.create_annotations(label_size)

    def create_annotations(self, label_size):
        for ax, data in zip(self.axes, self.datasets):
            station = data["station"]
            if station not in self.tracks:
                continue
            track = self.tracks[station]
            station_saved = self.saved_labels.get(station, {})
            for pos in height_label_positions(track["t_rel"], track):
                height_km = pos["height_km"]
                saved = station_saved.get(height_km)
                if saved:
                    text_xy = (float(saved["text_t_rel"]), float(saved["text_range_km"]))
                    ha = saved.get("ha", "center")
                    va = saved.get("va", "center")
                else:
                    text_xy = self.default_text_position(ax, pos)
                    ha = "center"
                    va = "center"
                ann = ax.annotate(
                    f"{height_km:.0f} km height",
                    xy=(pos["t_rel"], pos["range_km"]),
                    xytext=text_xy,
                    textcoords="data",
                    ha=ha,
                    va=va,
                    fontsize=label_size,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="none", alpha=0.35),
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.45, shrinkA=0, shrinkB=0),
                    zorder=7,
                    picker=True,
                )
                self.artists[ann] = {
                    "station": station,
                    "height_km": height_km,
                    "anchor_t_rel": pos["t_rel"],
                    "anchor_range_km": pos["range_km"],
                }

    def default_text_position(self, ax, pos):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = min(max(pos["t_rel"] + 0.45, x_min + 0.65), x_max - 0.65)
        y = min(max(pos["range_km"] + 0.12 * (y_max - y_min), y_min + 0.1 * (y_max - y_min)), y_max - 0.1 * (y_max - y_min))
        return x, y

    def on_press(self, event):
        if event.inaxes is None:
            return
        for ann in self.artists:
            hit, _info = ann.contains(event)
            if hit:
                self.selected = ann
                self.dragging = True
                return

    def on_motion(self, event):
        if not self.dragging or self.selected is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.selected.set_position((event.xdata, event.ydata))
        self.fig.canvas.draw_idle()

    def on_release(self, _event):
        self.dragging = False
        self.selected = None

    def on_key(self, event):
        if event.key == "s":
            self.save()
        elif event.key == "r":
            self.reset_defaults()
        elif event.key == "q":
            plt.close(self.fig)

    def reset_defaults(self):
        self.saved_labels = {}
        for ann in list(self.artists):
            ann.remove()
        self.artists.clear()
        self.create_annotations(8.0 * REPORT_FONT_SCALE)
        self.fig.canvas.draw_idle()

    def save(self):
        payload = default_payload()
        payload["t_ref_unix"] = self.t_ref
        payload["t_ref_utc"] = iso_utc(self.t_ref)
        payload["height_labels_km"] = list(HEIGHT_LABELS_KM)
        for ann, meta in self.artists.items():
            text_x, text_y = ann.get_position()
            payload["stations"].setdefault(meta["station"], []).append(
                {
                    "height_km": float(meta["height_km"]),
                    "anchor_t_rel": float(meta["anchor_t_rel"]),
                    "anchor_range_km": float(meta["anchor_range_km"]),
                    "text_t_rel": float(text_x),
                    "text_range_km": float(text_y),
                    "ha": ann.get_ha(),
                    "va": ann.get_va(),
                }
            )
        save_payload(self.args.out, payload)
        print(f"Wrote {self.args.out}")


def main():
    parser = argparse.ArgumentParser(description="Interactively place trail-height labels for Figure 6.")
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--trajectory-json", default=DEFAULT_TRAJECTORY_JSON)
    parser.add_argument("--out", default=DEFAULT_HEIGHT_LABELS)
    parser.add_argument("--rois", default=DEFAULT_OUT)
    parser.add_argument("--stations", nargs="+", default=["amherstnh", "siena", "pomfret", "clinton", "beverly"])
    args = parser.parse_args()
    HeightLabelEditor(args)
    plt.show()


if __name__ == "__main__":
    main()
