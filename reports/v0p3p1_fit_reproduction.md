# Reproducing trajectory fit v0.3.1

Run from the repository root in the base conda environment:

```bash
conda run -n base python fit_ballistic_head_trail_correlated_appendix.py
```

The defaults in the fitter select the v0.3.1 configuration: a six-parameter
inertial-state fit with the fixed 1.6 m, 5000 kg body model, one coherent code,
three-spectrum incoherent integration, and the full overlap covariance implied
by one-code stepping.

Required versioned inputs:

- `reports/head_echo_python_selection_v0p2p9_rdmf_txsteer_pol_rinterp4_zp16_3incoh_180618_180623/head_echo_python_measurement_selection.h5`
- `reports/interactive_sparse_component_imager/beverly_trail_positions.h5`
- `products/specular_trail_positions/specular_trail_position_summary.json`
- `products/ballistic_head_trail_fit/ballistic_initial_state.json`
- station geometry files under `2026_05_30_bolide/`

Input SHA-256 checksums:

```text
8bee112b9480de464cceb4051dd834fbbe1102bf47cfc32d669ae0b6797c26f6  head_echo_python_measurement_selection.h5
c7b716f32f7346b1cd5315621368d9f3cb5aac9a868358ecd182b3acc6a7a653  beverly_trail_positions.h5
57d46f5a1143009a56dcd16e260c429ea813cd751449e1bfb3158ea971cc0ba4  specular_trail_position_summary.json
7a21e1746c04220d1cd9a55be0a6fb279b969530a2c6a19505f538b8eff2ac83  ballistic_initial_state.json
```

The historical fit JSON records a `combined_fit_json` path, but the v0.3.1
fitter does not read that argument. The numerical initial state comes from
`ballistic_initial_state.json`.

The main output is written to
`products/ballistic_head_trail_correlated_fit_v0p2p9_rinterp4_zp16_3incoh/ballistic_head_trail_correlated_fit.json`.
