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
Rustplotlib supplies vector markers, physical-size error-bar caps and the
data-avoiding legend. The archived fit's reference epoch is 18:06:16.26 UTC.
