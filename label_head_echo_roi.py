#!/usr/bin/env python3
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector


DEFAULT_MERGED_DIR = "products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
DEFAULT_OUT = "annotations/head_echo_rois.json"


def iso_utc(t):
    return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")


def load_station_file(fname):
    with h5py.File(fname, "r") as h:
        sr = float(h["sample_rate"][()])
        station = h["station"][()].decode()
        return {
            "file": str(fname),
            "station": station,
            "t_unix": h["t_sample"][()] / sr,
            "range_km": h["range_km"][()],
            "P": h["P"][()],
            "N": h["N"][()],
            "D": h["D"][()],
        }


def station_files(merged_dir, stations=None):
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob("rdmf_peak_*_merged.h5"))
    if stations:
        wanted = set(stations)
        files = [f for f in files if f.name.removeprefix("rdmf_peak_").removesuffix("_merged.h5") in wanted]
    return files


def load_annotations(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema": "new_england_bolide.head_echo_roi.v1",
        "description": "Manual head-echo ROI labels from range-time-Doppler matched-filter products.",
        "stations": {},
    }


def save_annotations(path, annotations):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(annotations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize_box(data, t0, t1, r0, r1):
    t = data["t_unix"]
    r = data["range_km"]
    keep_t = (t >= min(t0, t1)) & (t <= max(t0, t1))
    keep_r = (r >= min(r0, r1)) & (r <= max(r0, r1))
    if not np.any(keep_t) or not np.any(keep_r):
        return None
    P = data["P"][np.ix_(keep_t, keep_r)]
    N = data["N"][np.ix_(keep_t, keep_r)]
    D = data["D"][np.ix_(keep_t, keep_r)]
    snr = P / np.maximum(N, 1e-12)
    imax = int(np.nanargmax(snr))
    ti, ri = np.unravel_index(imax, snr.shape)
    t_idx = np.where(keep_t)[0]
    r_idx = np.where(keep_r)[0]
    good = np.isfinite(D) & np.isfinite(snr)
    doppler = D[good]
    snr_good = snr[good]
    return {
        "n_time_bins": int(np.sum(keep_t)),
        "n_range_bins": int(np.sum(keep_r)),
        "peak": {
            "unix_time": float(t[t_idx[ti]]),
            "utc": iso_utc(t[t_idx[ti]]),
            "range_km": float(r[r_idx[ri]]),
            "snr_db": float(10.0 * np.log10(max(snr[ti, ri], 1e-12))),
            "doppler_hz": float(D[ti, ri]),
        },
        "doppler_hz": {
            "min": float(np.nanmin(doppler)) if doppler.size else None,
            "p05": float(np.nanpercentile(doppler, 5)) if doppler.size else None,
            "median": float(np.nanmedian(doppler)) if doppler.size else None,
            "p95": float(np.nanpercentile(doppler, 95)) if doppler.size else None,
            "max": float(np.nanmax(doppler)) if doppler.size else None,
        },
        "snr_db": {
            "max": float(10.0 * np.log10(max(np.nanmax(snr_good), 1e-12))) if snr_good.size else None,
            "median": float(10.0 * np.log10(max(np.nanmedian(snr_good), 1e-12))) if snr_good.size else None,
        },
    }


class HeadEchoLabeler:
    def __init__(self, files, out_path, snr_threshold=5.0):
        self.files = files
        self.out_path = Path(out_path)
        self.snr_threshold = snr_threshold
        self.annotations = load_annotations(self.out_path)
        self.index = 0
        self.data = None
        self.t0_unix = None
        self.rectangles = []
        self.current_box = None
        self.selector = None

        self.fig = plt.figure(figsize=(14.5, 8.4), constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.0, 0.10])
        self.ax_rti = self.fig.add_subplot(gs[0, 0])
        self.ax_dop = self.fig.add_subplot(gs[0, 1], sharex=self.ax_rti, sharey=self.ax_rti)
        self.ax_prev = self.fig.add_subplot(gs[1, 0])
        self.ax_next = self.fig.add_subplot(gs[1, 1])

        pos = self.ax_prev.get_position()
        self.ax_save = self.fig.add_axes([pos.x0, pos.y0, 0.12, pos.height])
        self.ax_no = self.fig.add_axes([pos.x0 + 0.14, pos.y0, 0.16, pos.height])
        self.ax_clear = self.fig.add_axes([pos.x0 + 0.32, pos.y0, 0.12, pos.height])
        self.ax_back = self.fig.add_axes([pos.x0 + 0.46, pos.y0, 0.12, pos.height])
        pos2 = self.ax_next.get_position()
        self.ax_forward = self.fig.add_axes([pos2.x1 - 0.12, pos2.y0, 0.12, pos2.height])
        self.ax_status = self.fig.add_axes([pos2.x0, pos2.y0, 0.52, pos2.height])
        self.ax_prev.remove()
        self.ax_next.remove()
        self.ax_status.axis("off")

        self.btn_save = Button(self.ax_save, "Save ROI")
        self.btn_no = Button(self.ax_no, "No Head Echo")
        self.btn_clear = Button(self.ax_clear, "Clear")
        self.btn_back = Button(self.ax_back, "Previous")
        self.btn_forward = Button(self.ax_forward, "Next")
        self.btn_save.on_clicked(self.save_roi)
        self.btn_no.on_clicked(self.no_head_echo)
        self.btn_clear.on_clicked(self.clear_current)
        self.btn_back.on_clicked(self.previous_station)
        self.btn_forward.on_clicked(self.next_station)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw_station(self):
        if self.selector:
            self.selector.disconnect_events()
            self.selector = None
        self.clear_rectangles()
        self.data = load_station_file(self.files[self.index])
        self.t0_unix = float(self.data["t_unix"][0])
        self.current_box = None
        self.ax_rti.clear()
        self.ax_dop.clear()

        tplot = self.data["t_unix"] - self.t0_unix
        r = self.data["range_km"]
        snr = self.data["P"] / np.maximum(self.data["N"], 1e-12)
        snr_db = 10.0 * np.log10(np.maximum(snr, 1e-12))
        dop = self.data["D"].copy()
        dop[snr < 10.0 ** (self.snr_threshold / 10.0)] = np.nan

        snr_lim = np.nanpercentile(snr_db, [2, 99.7])
        self.ax_rti.pcolormesh(tplot, r, snr_db.T, shading="auto", cmap="inferno", vmin=snr_lim[0], vmax=snr_lim[1])
        self.ax_rti.set_title(f"{self.data['station']} RTI / SNR")
        self.ax_rti.set_ylabel("Bistatic group range (km)")
        self.ax_rti.set_xlabel(f"Seconds after {iso_utc(self.t0_unix)}")

        vmax = np.nanpercentile(np.abs(dop), 98) if np.any(np.isfinite(dop)) else 1000.0
        vmax = max(1000.0, float(vmax))
        cmap = plt.cm.seismic.copy()
        cmap.set_bad("0.65")
        self.ax_dop.pcolormesh(tplot, r, dop.T, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        self.ax_dop.set_title(f"{self.data['station']} peak Doppler, SNR >= {self.snr_threshold:g} dB")
        self.ax_dop.set_xlabel(f"Seconds after {iso_utc(self.t0_unix)}")

        for ax in (self.ax_rti, self.ax_dop):
            ax.set_xlim(float(np.min(tplot)), float(np.max(tplot)))
            ax.set_ylim(float(np.min(r)), float(np.max(r)))
            ax.grid(True, color="0.85", lw=0.4)

        self.draw_existing_annotation()
        self.set_status()
        self.selector = RectangleSelector(
            self.ax_rti,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=0.000001,
            minspany=0.1,
            spancoords="data",
            interactive=True,
        )
        self.fig.canvas.draw_idle()

    def clear_rectangles(self):
        for rect in self.rectangles:
            try:
                rect.remove()
            except (NotImplementedError, ValueError):
                pass
        self.rectangles = []

    def add_box_patches(self, x0, x1, y0, y1, color="cyan"):
        for ax in (self.ax_rti, self.ax_dop):
            rect = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                ec=color,
                lw=2.0,
            )
            ax.add_patch(rect)
            self.rectangles.append(rect)

    def draw_existing_annotation(self):
        station = self.data["station"]
        entry = self.annotations.get("stations", {}).get(station)
        if not entry:
            return
        if entry.get("has_head_echo") and "time_range" in entry and "range_km" in entry:
            x0 = float(entry["time_range"]["start_unix"]) - self.t0_unix
            x1 = float(entry["time_range"]["end_unix"]) - self.t0_unix
            y0 = entry["range_km"]["min"]
            y1 = entry["range_km"]["max"]
            self.add_box_patches(x0, x1, y0, y1, color="lime")

    def on_select(self, eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        self.clear_rectangles()
        x0, x1 = float(eclick.xdata), float(erelease.xdata)
        y0, y1 = float(eclick.ydata), float(erelease.ydata)
        self.current_box = (x0, x1, y0, y1)
        self.add_box_patches(x0, x1, y0, y1, color="cyan")
        self.set_status("ROI selected. Click Save ROI or press s.")
        self.fig.canvas.draw_idle()

    def annotation_for_current_box(self):
        x0, x1, y0, y1 = self.current_box
        t0, t1 = self.t0_unix + x0, self.t0_unix + x1
        r0, r1 = y0, y1
        summary = summarize_box(self.data, t0, t1, r0, r1)
        return {
            "has_head_echo": True,
            "source_file": self.data["file"],
            "time_range": {
                "start_unix": float(min(t0, t1)),
                "end_unix": float(max(t0, t1)),
                "start_utc": iso_utc(min(t0, t1)),
                "end_utc": iso_utc(max(t0, t1)),
            },
            "range_km": {
                "min": float(min(r0, r1)),
                "max": float(max(r0, r1)),
            },
            "summary": summary,
        }

    def save_roi(self, _event=None):
        if self.current_box is None:
            self.set_status("No ROI box selected. Drag on the RTI panel first.")
            return
        station = self.data["station"]
        self.annotations.setdefault("stations", {})[station] = self.annotation_for_current_box()
        save_annotations(self.out_path, self.annotations)
        self.set_status(f"Saved ROI for {station} -> {self.out_path}")
        self.draw_station()

    def no_head_echo(self, _event=None):
        station = self.data["station"]
        self.annotations.setdefault("stations", {})[station] = {
            "has_head_echo": False,
            "source_file": self.data["file"],
            "marked_utc": iso_utc(datetime.now(UTC).timestamp()),
        }
        save_annotations(self.out_path, self.annotations)
        self.set_status(f"Saved no-head-echo label for {station} -> {self.out_path}")
        self.draw_station()

    def clear_current(self, _event=None):
        station = self.data["station"]
        self.annotations.setdefault("stations", {}).pop(station, None)
        save_annotations(self.out_path, self.annotations)
        self.clear_rectangles()
        self.current_box = None
        self.set_status(f"Cleared annotation for {station}")
        self.fig.canvas.draw_idle()

    def next_station(self, _event=None):
        self.index = min(self.index + 1, len(self.files) - 1)
        self.draw_station()

    def previous_station(self, _event=None):
        self.index = max(self.index - 1, 0)
        self.draw_station()

    def on_key(self, event):
        if event.key == "s":
            self.save_roi()
        elif event.key == "n":
            self.no_head_echo()
        elif event.key == "c":
            self.clear_current()
        elif event.key in ("right", "pagedown"):
            self.next_station()
        elif event.key in ("left", "pageup"):
            self.previous_station()

    def set_status(self, msg=None):
        station = self.data["station"] if self.data else ""
        entry = self.annotations.get("stations", {}).get(station)
        if msg is None:
            if entry is None:
                label = "unlabeled"
            elif entry.get("has_head_echo"):
                label = "head echo ROI saved"
            else:
                label = "marked no head echo"
            msg = f"{self.index + 1}/{len(self.files)} {station}: {label}"
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(0.0, 0.5, msg, va="center", ha="left", fontsize=10)
        self.fig.canvas.draw_idle()

    def run(self):
        self.draw_station()
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Manually label head-echo range-time ROIs in RDMF products.")
    p.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--stations", nargs="+")
    p.add_argument("--snr-threshold", type=float, default=5.0)
    args = p.parse_args()

    files = station_files(args.merged_dir, args.stations)
    if not files:
        raise SystemExit(f"No merged RDMF files found in {args.merged_dir}")
    HeadEchoLabeler(files, args.out, snr_threshold=args.snr_threshold).run()


if __name__ == "__main__":
    main()
