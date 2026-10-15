"""Archive the saved-fit plot's exact curves, errors and covariance envelopes."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import sys

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import replot_article_range_doppler as source


def main():
    output = Path("products/range_doppler_figure/scene.h5")
    output.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(sys, "argv", ["replot_article_range_doppler"]), \
         patch.object(Figure, "savefig"), patch.object(plt, "close"):
        source.main()
        fig = plt.gcf()
    with h5py.File(output, "w") as h:
        fit_path = source.fit.DEFAULT_OUT_DIR / "ballistic_head_trail_correlated_fit.json"
        h.attrs["fit_path"] = str(fit_path)
        h.attrs["fit_sha256"] = hashlib.sha256(fit_path.read_bytes()).hexdigest()
        for i, ax in enumerate(fig.axes):
            g = h.create_group(f"panel_{i}")
            g["xlim"], g["ylim"] = ax.get_xlim(), ax.get_ylim()
            g.attrs["xlabel"] = ax.get_xlabel()
            for j, line in enumerate(ax.lines):
                item = g.create_group(f"lines/{j:03}")
                item["xy"] = line.get_xydata()
                item["rgb"] = to_rgba(line.get_color())[:3]
                item.attrs["marker"] = str(line.get_marker())
                item.attrs["size"] = line.get_markersize()
                item.attrs["draw_line"] = line.get_linestyle() not in ("None", "none", "", " ")
            for j, coll in enumerate(ax.collections):
                if isinstance(coll, LineCollection):
                    paths = coll.get_segments()
                    color = coll.get_colors()[0]
                    name = "errors"
                elif isinstance(coll, PolyCollection):
                    paths = [p.vertices for p in coll.get_paths()]
                    color = coll.get_facecolors()[0]
                    name = "bands"
                else:
                    raise TypeError(type(coll))
                for k, path in enumerate(paths):
                    item = g.create_group(f"{name}/{j:03}_{k:03}")
                    item["xy"] = path
                    item["rgba"] = color
            # Keep the measured correlation-length annotation; omit version stamps.
            for text in ax.texts:
                if not text.get_text().startswith("v"):
                    g.attrs["annotation"] = text.get_text()
                    g["annotation_xy"] = text.get_position()
    with h5py.File(output) as h:
        for i, ax in enumerate(fig.axes):
            for j, line in enumerate(ax.lines):
                np.testing.assert_array_equal(h[f"panel_{i}/lines/{j:03}/xy"][:], line.get_xydata())
            for j, coll in enumerate(ax.collections):
                name = "errors" if isinstance(coll, LineCollection) else "bands"
                paths = coll.get_segments() if name == "errors" else [p.vertices for p in coll.get_paths()]
                for k, path in enumerate(paths):
                    np.testing.assert_array_equal(h[f"panel_{i}/{name}/{j:03}_{k:03}/xy"][:], path)
    plt.close(fig)
    print(f"Verified exact plot coordinates: {output}")


if __name__ == "__main__":
    main()
