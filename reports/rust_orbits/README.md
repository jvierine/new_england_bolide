# Article orbit figure

The final figure is 115.22 mm wide (70% of the article's 164.6 mm linewidth),
with 10 pt LaTeX labels. Rustplotlib's `ArticleLayout` fixes those dimensions
and exports a checked LaTeX inclusion. Both axes use the original -8 to 8 AU
limits and the same physical scale.

The archived `input.h5` retains all 2,000 v0.3.1 Keplerian samples, the 57
complete catalogue orbits, planetary elements at the original epoch, and input
hashes. Curves use the original `plot_orbit_xy.py` functions and point counts:
500 for bolide/meteorite orbits, 900 for planets. No refitting or resampling of
the orbital-element distribution is performed. The source catalogue is read
from the article repository so the comparison subset matches the manuscript.

```sh
conda run -n base python reports/export_orbit_figure.py --from-archive
HDF5_DIR=/opt/anaconda3 DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib \
  cargo run --release --manifest-path reports/rust_orbits/Cargo.toml -- \
  products/orbit_figure
```

Omit `--from-archive` to refresh the input snapshot from the current original
products and article macros. The exporter validates fit version/sample count,
checks all curve coordinates against the original plotting functions, and
writes `products/orbit_figure/scene.h5` plus the archive. Figure exports are
`products/orbit_figure/orbit_xy_bolide.pdf`, `.png`, and `.include.tex`.

The 57 comparison orbits are pale blue-gray hairlines. All 2,000 bolide curves
form a darker uncertainty ensemble, rasterized together at 450 dpi to keep the
PDF compact. Planetary orbits, catalogue curves, axes and labels remain vector.
The redundant orbital-element inset and provenance stamp are omitted; values
remain in the manuscript table, with provenance in the caption source/comment.

Reusable Rustplotlib features used here are `Line::opacity`, `Legend::columns`,
`Panel::rasterize_lines`, and `ArticleLayout::figure`/`save_for_article`.
