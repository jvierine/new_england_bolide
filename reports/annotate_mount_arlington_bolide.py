#!/usr/bin/env python3
"""Interactively annotate the bolide in the Mount Arlington screen grab."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, PolygonSelector
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "data/security_camera/mount_arlington_20260530_180622.png"
DEFAULT_OUTPUT = ROOT / "annotations/mount_arlington_bolide.h5"
DEFAULT_FIGURE = ROOT / "reports/figures/mount_arlington_bolide_annotation.png"
DEFAULT_UTC = "2026-05-30T18:06:22Z"
DEFAULT_LAT_DEG = 40.0
DEFAULT_LON_DEG = -74.0
DEFAULT_CROP = (150, 310, 100, 300)  # xmin, xmax, ymin, ymax in source pixels
DEFAULT_LOCATION_NAME = "Mount Arlington, New Jersey, US"
DEFAULT_TITLE = "Mount Arlington security-camera frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw separate polygons around the bolide and a clear blue-sky reference. "
            "Click vertices in the enlarged right panel and click the first vertex "
            "to close each polygon, then save both masks."
        )
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--assumed-utc", default=DEFAULT_UTC)
    parser.add_argument("--latitude-deg", type=float, default=DEFAULT_LAT_DEG)
    parser.add_argument("--longitude-deg", type=float, default=DEFAULT_LON_DEG)
    parser.add_argument("--location-name", default=DEFAULT_LOCATION_NAME)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=DEFAULT_CROP,
        help="Enlarged crop in original-image pixel coordinates.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_annotation(
    output_path: Path,
    bolide_vertices_xy: np.ndarray,
    blue_sky_vertices_xy: np.ndarray,
    image_path: Path,
    image_shape: tuple[int, int, int],
    assumed_utc: str,
    latitude_deg: float,
    longitude_deg: float,
    location_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = "new_england_bolide.optical_bolide_annotation.v1"
        h5.attrs["created_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        h5.attrs["source_image"] = str(image_path.relative_to(ROOT))
        h5.attrs["source_image_sha256"] = sha256(image_path)
        h5.attrs["location_name"] = location_name
        h5.attrs["observer_latitude_deg"] = float(latitude_deg)
        h5.attrs["observer_longitude_deg"] = float(longitude_deg)
        h5.attrs["event_time_utc"] = assumed_utc
        h5.attrs["event_time_status"] = "known Cape Cod Bay bolide event epoch"
        h5.attrs["coordinate_convention"] = "image pixels: x rightward, y downward, zero based"
        h5.attrs["annotation_method"] = "manual Matplotlib PolygonSelector; two labeled polygons"
        h5.create_dataset("image_shape_yxc", data=np.asarray(image_shape, dtype=np.int64))
        h5.create_dataset("bolide_polygon_xy", data=bolide_vertices_xy)
        h5.create_dataset("blue_sky_polygon_xy", data=blue_sky_vertices_xy)


def write_figure(
    figure_path: Path,
    image: np.ndarray,
    bolide_vertices_xy: np.ndarray,
    blue_sky_vertices_xy: np.ndarray,
    crop: tuple[int, int, int, int],
    title: str,
    event_utc: str,
) -> None:
    xmin, xmax, ymin, ymax = crop
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax in axes:
        ax.imshow(image)
        ax.add_patch(
            Polygon(
                bolide_vertices_xy, closed=True, fill=False, edgecolor="cyan",
                linewidth=2.0, label="bolide",
            )
        )
        ax.add_patch(
            Polygon(
                blue_sky_vertices_xy, closed=True, fill=False, edgecolor="blue",
                linewidth=2.0, label="blue-sky reference",
            )
        )
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
    axes[0].add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="yellow", linewidth=1.5,
        )
    )
    axes[0].set_title("Full frame")
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymax, ymin)
    axes[1].set_xticks(np.arange(xmin, xmax + 1, 10), minor=True)
    axes[1].set_yticks(np.arange(ymin, ymax + 1, 10), minor=True)
    axes[1].grid(which="minor", color="white", alpha=0.3, linewidth=0.5)
    axes[1].set_title("Manual bolide and blue-sky polygons")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"{title}; {event_utc}")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    figure_path = args.figure.expanduser().resolve()
    image = np.asarray(Image.open(image_path).convert("RGB"))
    xmin, xmax, ymin, ymax = map(int, args.crop)
    if not (0 <= xmin < xmax <= image.shape[1] and 0 <= ymin < ymax <= image.shape[0]):
        raise ValueError(f"Crop {args.crop} lies outside image shape {image.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    fig.subplots_adjust(bottom=0.24, wspace=0.22)
    full_ax, zoom_ax = axes
    full_ax.imshow(image)
    full_ax.add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="yellow", linewidth=1.5,
        )
    )
    full_ax.set_title("Full frame; yellow box is enlarged at right")
    full_ax.set_xlabel("pixel x")
    full_ax.set_ylabel("pixel y")

    zoom_ax.imshow(image)
    zoom_ax.set_xlim(xmin, xmax)
    zoom_ax.set_ylim(ymax, ymin)
    zoom_ax.set_xticks(np.arange(xmin, xmax + 1, 10), minor=True)
    zoom_ax.set_yticks(np.arange(ymin, ymax + 1, 10), minor=True)
    zoom_ax.grid(which="minor", color="white", alpha=0.35, linewidth=0.5)
    zoom_ax.set_title("Two required masks: bolide (cyan) and clear blue sky (blue)")
    zoom_ax.set_xlabel("original-image pixel x")
    zoom_ax.set_ylabel("original-image pixel y")

    status = fig.text(
        0.5, 0.145,
        "STEP 1 — BOLIDE (cyan): click vertices, then click the first vertex to close.",
        ha="center", va="center",
        weight="bold",
    )
    selected: dict[str, np.ndarray | None] = {"bolide": None, "blue_sky": None}

    def activate(name: str) -> None:
        bolide_selector.set_active(name == "bolide")
        sky_selector.set_active(name == "blue_sky")
        if name == "bolide":
            status.set_text(
                "STEP 1 — BOLIDE (cyan): click vertices, then click the first vertex to close."
            )
        else:
            status.set_text(
                "STEP 2 — CLEAR BLUE SKY (blue): outline a cloud-free sky patch and close it."
            )
        fig.canvas.draw_idle()

    def on_bolide_select(vertices: list[tuple[float, float]]) -> None:
        polygon = np.asarray(vertices, dtype=float)
        if len(polygon) < 3:
            status.set_text("Bolide polygon needs at least three vertices.")
            fig.canvas.draw_idle()
            return
        selected["bolide"] = polygon
        status.set_text(
            f"Bolide saved with {len(polygon)} vertices. Now select clear blue sky."
        )
        fig.canvas.draw_idle()
        activate("blue_sky")

    def on_sky_select(vertices: list[tuple[float, float]]) -> None:
        polygon = np.asarray(vertices, dtype=float)
        if len(polygon) < 3:
            status.set_text("Blue-sky polygon needs at least three vertices.")
            fig.canvas.draw_idle()
            return
        selected["blue_sky"] = polygon
        if selected["bolide"] is None:
            status.set_text(
                f"Blue sky saved with {len(polygon)} vertices. Bolide mask is still missing."
            )
        else:
            status.set_text(
                f"Both masks are ready: bolide {len(selected['bolide'])} vertices; "
                f"blue sky {len(polygon)} vertices. Press Save both masks."
            )
        fig.canvas.draw_idle()

    bolide_selector = PolygonSelector(
        zoom_ax,
        on_bolide_select,
        useblit=True,
        props={"color": "cyan", "linewidth": 2.0, "alpha": 0.9},
        handle_props={"marker": "o", "markersize": 5, "mec": "cyan", "mfc": "white"},
    )
    sky_selector = PolygonSelector(
        zoom_ax,
        on_sky_select,
        useblit=True,
        props={"color": "blue", "linewidth": 2.0, "alpha": 0.9},
        handle_props={"marker": "s", "markersize": 5, "mec": "blue", "mfc": "white"},
    )
    sky_selector.set_active(False)

    bolide_ax = fig.add_axes((0.18, 0.065, 0.18, 0.05))
    sky_ax = fig.add_axes((0.38, 0.065, 0.18, 0.05))
    save_ax = fig.add_axes((0.58, 0.065, 0.18, 0.05))
    reset_ax = fig.add_axes((0.78, 0.065, 0.10, 0.05))
    bolide_button = Button(bolide_ax, "1  Select BOLIDE", color="cyan", hovercolor="0.85")
    sky_button = Button(sky_ax, "2  Select BLUE SKY", color="cornflowerblue", hovercolor="0.85")
    save_button = Button(save_ax, "3  Save BOTH masks")
    reset_button = Button(reset_ax, "Reset all")

    bolide_button.on_clicked(lambda _event: activate("bolide"))
    sky_button.on_clicked(lambda _event: activate("blue_sky"))

    def save(_event) -> None:
        bolide_vertices = selected["bolide"]
        sky_vertices = selected["blue_sky"]
        if bolide_vertices is None or len(bolide_vertices) < 3:
            status.set_text("Bolide mask is missing. Press 1 Select BOLIDE.")
            fig.canvas.draw_idle()
            return
        if sky_vertices is None or len(sky_vertices) < 3:
            status.set_text("Blue-sky mask is missing. Press 2 Select BLUE SKY.")
            fig.canvas.draw_idle()
            return
        write_annotation(
            output_path,
            bolide_vertices,
            sky_vertices,
            image_path,
            image.shape,
            args.assumed_utc,
            args.latitude_deg,
            args.longitude_deg,
            args.location_name,
        )
        write_figure(
            figure_path, image, bolide_vertices, sky_vertices,
            (xmin, xmax, ymin, ymax),
            args.title,
            args.assumed_utc,
        )
        status.set_text(f"Saved {output_path} and {figure_path}")
        print(f"Saved annotation: {output_path}", flush=True)
        print(f"Saved figure: {figure_path}", flush=True)
        fig.canvas.draw_idle()

    def reset(_event) -> None:
        selected["bolide"] = None
        selected["blue_sky"] = None
        bolide_selector.clear()
        sky_selector.clear()
        activate("bolide")
        fig.canvas.draw_idle()

    save_button.on_clicked(save)
    reset_button.on_clicked(reset)
    plt.show()


if __name__ == "__main__":
    main()
