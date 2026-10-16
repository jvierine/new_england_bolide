# Article range--Doppler figure

The archived `scene.h5` contains the original saved-fit plot's exact measurement,
error-bar, predicted-observable and covariance-envelope coordinates. No fit is
performed by the renderer. Regenerate the scene from the local fit inputs with:

```sh
conda run -n base python reports/export_range_doppler_figure.py
```

Render the archived scene directly by passing this directory, or render the
fresh export by passing `products/range_doppler_figure`:

```sh
HDF5_DIR=/opt/anaconda3 DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib \
  cargo run --release --manifest-path reports/rust_range_doppler/Cargo.toml -- \
  products/range_doppler_figure
```

Output is a native 164.6 mm-wide PDF with 10 pt text. Station colors identify
both observations and model curves; the article caption defines the symbols.
Muted blue, terracotta and sage identify Beverly, Clinton and Pomfret. The
renderer remaps only station colors, including covariance envelopes and legend
keys; archived coordinates and uncertainty magnitudes are unchanged.
The article replot omits the isolated Clinton Doppler pick at 18:06:21.240 UTC
(source index 323) at the author's request. This is a display-only exclusion;
the saved fit, model curves and covariance are not modified or refitted.
Rustplotlib supplies vector markers, physical-size error-bar caps and the
data-avoiding legend. The archived fit's reference epoch is 18:06:16.26 UTC.
