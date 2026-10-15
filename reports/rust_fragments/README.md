# Article trail and dark-flight figure

This renderer changes presentation only. The archived `scene.h5` contains the
coordinates, component extents, axis bounds, and pressure curve exported from
the existing article figure, together with SHA-256 hashes of its scientific
inputs. No trajectory fitting or dark-flight integration occurs in Rust.

Render the archived scene from the processing repository root:

```sh
HDF5_DIR=/opt/anaconda3 DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib cargo run --release --manifest-path reports/rust_fragments/Cargo.toml -- reports/rust_fragments
```

The HDF5 environment paths above apply to the local macOS Conda installation;
use the corresponding HDF5 installation on other systems. PDF and PNG are
written beside the input scene. The PDF uses vector curves, markers, axes,
and LaTeX labels. Only the translucent component extents are rasterized.

To explicitly export a fresh scene from the Python source plot:

```sh
conda run -n base python reports/export_fragment_figure.py --out products/fragment_figure/scene.h5
```

The exporter verifies exact equality of every exported line and marker with
the source Matplotlib artists. The pressure abscissa is transformed with
`log10` only for plotting, preserving the original logarithmic axis.
