#!/usr/bin/env python3
"""Export the existing article plot's artists without changing its data or models."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import plot_fragment_size_hypotheses as source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("products/fragment_figure/scene.h5"))
    args = parser.parse_args()
    fit = json.loads(source.DEFAULT_TRAJECTORY.read_text())
    components = source.load_components_h5(source.DEFAULT_TRAIL_POSITION_H5, max_height_km=120.)
    paths = source.load_dark_flight_kmz(source.DEFAULT_DARK_FLIGHT_KMZ)
    detections = source.load_dark_flight_radar_detections(source.DEFAULT_DARK_FLIGHT_JSON)
    terminus = source.terminus_marker_summary(fit, 33.61)
    trajectory = source.build_fitted_trajectory(fit, 90., 33.61)
    with patch.object(Figure, "savefig"), patch.object(source.plt, "close"):
        source.make_plot(args.out, [], components, {}, source.parse_utc("2026-05-30T18:06:23Z"),
                         terminus, 18., trajectory, paths, detections)
        fig = source.plt.gcf()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.attrs["description"] = "Exact source-plot coordinates; presentation-only Rustplotlib rendering"
        h.attrs["pressure_convention"] = "rho_air * v_air**2"
        inputs = [source.DEFAULT_TRAJECTORY, *source.DEFAULT_TRAIL_POSITION_H5,
                  source.DEFAULT_DARK_FLIGHT_KMZ, source.DEFAULT_DARK_FLIGHT_JSON]
        provenance = h.create_group("provenance")
        for i, path in enumerate(inputs):
            group = provenance.create_group(str(i))
            group.attrs["path"] = str(path)
            group.attrs["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for i, ax in enumerate(fig.axes):
            group = h.create_group(f"panel_{i}")
            group["xlim"] = ax.get_xlim()
            group["ylim"] = ax.get_ylim()
            lines = group.create_group("lines")
            for j, line in enumerate(ax.lines):
                item = lines.create_group(f"{j:03}")
                item["xy"] = line.get_xydata()
                item.attrs["kind"] = 0 if i == 2 else (1 if line.get_linestyle() == "--" else 2)
            markers = group.create_group("markers")
            for j, collection in enumerate(ax.collections):
                item = markers.create_group(f"{j:03}")
                item["xy"] = np.asarray(collection.get_offsets())
                r, g, b = collection.get_facecolors()[0, :3]
                item.attrs["kind"] = 1 if r > g and r > b else (2 if b > g else 0)
            ellipses = [[p.center[0], p.center[1], p.width, p.height, p.get_alpha()]
                        for p in ax.patches if isinstance(p, Ellipse)]
            group["ellipses"] = np.asarray(ellipses).reshape(-1, 5)
    with h5py.File(args.out) as h:
        for i, ax in enumerate(fig.axes):
            for j, line in enumerate(ax.lines):
                np.testing.assert_array_equal(h[f"panel_{i}/lines/{j:03}/xy"][:], line.get_xydata())
            for j, collection in enumerate(ax.collections):
                np.testing.assert_array_equal(h[f"panel_{i}/markers/{j:03}/xy"][:], collection.get_offsets())
    source.plt.close(fig)
    print(f"Exported and verified all source curves and markers: {args.out}")


if __name__ == "__main__":
    main()
