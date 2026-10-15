# Article observing-geometry map

Run from the processing repository:

```sh
conda run -n base python reports/plot_article_observing_geometry.py
```

The Python preparation step writes `products/observing_geometry/observing_geometry.h5`.
It preserves the archived v0.2.9 trajectory, 300 km backward extension, 90 km
start of the observed segment, six Marc Fries weather-radar centroids, and
the station and Mechanicville coordinates used by the earlier figure.
The camera coordinate identifies the town, not a surveyed camera position.
Natural Earth 10 m land, coastlines, and state boundaries supply map context.

The Rust renderer uses the sibling `rustplotlib` checkout, HDF5, and LaTeX
with `newtxtext`/`newtxmath`. The wrapper locates HDF5 in the active conda
environment. It exports vector PDF and a 450 dpi PNG to
`reports/figures/appendix_correlated_fit/article_observing_geometry`.
The map's latitude and longitude scales have equal local distances near
42.225 degrees north. The figure is natively 128.4 mm wide with 10 pt labels,
matching the article body size. Include the PDF at its natural size, without
LaTeX width scaling, to preserve the typography.
Marker sizes and label offsets are specified in physical millimetres.
