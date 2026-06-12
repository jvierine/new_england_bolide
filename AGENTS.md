# Repository Notes

This repository is the processing and analysis workspace for the 2026 May 30
Cape Cod Bay / New England bolide radar analysis. Keep data processing,
scripts, generated products, figures, and regenerated report-support assets
here.

All analysis, processing, plotting, and asset-generation code for the paper
belongs in this repository. Do not add such code to the Overleaf manuscript
repository at `/home/j/src/cape_cod_bay`; copy only generated manuscript-facing
assets there.

The manuscript TeX entry point now lives in the sibling repository:

```text
/home/j/src/cape_cod_bay
```

That repository is linked to Overleaf. Use it for manuscript-level edits and
pushes to Overleaf. Overleaf cannot follow local links or absolute paths, so
the manuscript repository must contain its own copies of the report body,
tables, bibliography, figure PNGs, and TeX support files under its local
`reports/` directory.

When report-support assets are regenerated here, copy the needed files into
`../cape_cod_bay/reports/` before pushing the manuscript to Overleaf.

Data products must be stored as HDF5 (`.h5`) unless there is an explicit,
documented external format requirement. Do not create NumPy `.npz` data
products for analysis or manuscript support assets.

When adding or updating memo figures, cite the script that generated each plot
in the memo text or caption/provenance list before compiling and pushing the
memo repository.

For head-echo RDMF products used in RCS or SNR work, use noise-equalized
SNR-maximizing two-channel polarization matching by default. Only generate
uncombined channel-summed RDMF products when explicitly making a comparison or
diagnostic, and label those products/figures as unpolarized.

## Remote Compute On avaruus

Use `juha@avaruus.uit.no` for expensive matched-filter and RDMF jobs that are
too large for the local workstation. The remote checkout is:

```text
/home/juha/src/new_england_bolide
```

The host has 256 CPU cores and about 1 TiB RAM. Raw data, products, and a local
Python environment are available in the remote checkout. The remote working tree
may contain local edits, so prefer syncing only the needed scripts, metadata,
and generated assets with `rsync` instead of doing destructive git operations
there.

Typical remote command pattern:

```bash
ssh juha@avaruus.uit.no 'cd ~/src/new_england_bolide && \
  source .venv/bin/activate 2>/dev/null || true; \
  python rdmf_peak_batch.py ...'
```

For long jobs, run under `nohup` and log to `products/*.log`. Check progress
with `pgrep -af "rdmf_peak_batch.py"` and `tail -f products/<log>.log`, and
make sure no stale workers remain after completion.

The full 0--600 km, all-station, 4x zero-padded Doppler job is very expensive.
For head-echo trajectory updates, prefer station/range-window-specific products
such as Beverly 210--265 km, Clinton 245--280 km, and Pomfret 370--405 km.
After a remote run, `rsync` generated products and figures back to the local
checkout before committing. HDF5 products under `products/` are normally ignored
and should not be committed unless explicitly needed.

## Web Server Deploys

Use `j@juha.no` as the web server for public static deployments. Web
directories on that server live under:

```text
/var/www/html/
```

The Cape Cod Bay bolide WebGL viewer is deployed as:

```text
/var/www/html/bolide/index.html
```

The local generated source file for that single-page viewer is:

```text
reports/figures/bolide_webgl_volume_viewer.html
```

Before replacing the deployed file, make a timestamped backup on the server,
then copy the regenerated local HTML, for example:

```bash
ssh j@juha.no 'cp /var/www/html/bolide/index.html /var/www/html/bolide/index.html.bak-$(date -u +%Y%m%dT%H%M%SZ)'
rsync -av --checksum reports/figures/bolide_webgl_volume_viewer.html j@juha.no:/var/www/html/bolide/index.html
```

Verify the public page at `https://juha.no/bolide/` after deployment.

## Current Analysis State

As of 2026-06-03, the head-echo trajectory products use the 4x zero-padded
Doppler RDMF products in:

```text
products/rdmf_zeropad4_head_merged/
```

Those products contain the station-specific head-echo windows used after the
remote avaruus run: Beverly, Clinton, and Pomfret. The current head-echo fit is:

```text
products/head_echo_trajectory_roi_fit/head_echo_trajectory.json
```

Important current trajectory state:

- The first Beverly point was removed from the fit as an outlier by trimming the
  first 0.06 s of Beverly retained ROI picks.
- The current head-only trajectory fit has 177 observations.
- The current head-only fit residuals are about 468 m RMS in bistatic range and
  90 Hz RMS in Doppler.
- Figure 5 in the manuscript is generated from
  `reports/generate_trajectory_report_assets.py` as
  `reports/figures/head_echo_range_doppler_fit.png`.
- The transmit AoD ambiguity figures were regenerated after the Beverly trim, so
  they use the updated 82.0 km head-echo reference height.

Recent processing commits that capture this state:

```text
c2a7fb0 Use zero-padded Doppler FFT for head echo fit
626dd93 Remove Beverly head echo outlier from trajectory fit
a8be605 Regenerate AoD ambiguity figure after trajectory trim
eed62a6 Update trajectory uncertainty product image
```

Recent Overleaf commits that synced manuscript state:

```text
4ce8c16 Remove Beverly outlier from head trajectory figure
bd47fbf Clarify ballistic head trail fit equations
57a4bf3 Lower Siena trail height label
```

Do not confuse the old straight-line ECI helper fit with the manuscript's
reported combined head/trail solution. Section 3.4 now describes a ballistic
inertial fit: trail positions are closest-point constraints to the propagated
ballistic curve.

Do not estimate atmospheric drag in future ballistic or orbit-association
updates. Use a fixed physical body model for the bolide: a spherical meteoroid
with 1.6 m diameter and 5 metric tons mass. If an atmospheric drag coefficient
is needed for propagation, derive the ballistic parameter from that fixed
sphere and mass assumption rather than fitting \(C_D A/m\) as a free parameter.

The `dasst` package is available in the `/home/j/fys3002` virtual environment,
installed editable from `/home/j/src/dasst/src`. Use this interpreter for
orbit-association work that needs `dasst`:

```bash
source /home/j/fys3002/bin/activate
python -c "import dasst; print(dasst.__file__)"
```

Do not use the base conda environment for `dasst`; it does not contain the
package.

The independent correlated-covariance ballistic appendix workflow lives in:

```text
fit_ballistic_head_trail_correlated_appendix.py
products/ballistic_head_trail_correlated_appendix_fit/
reports/figures/appendix_correlated_fit/
reports/appendix_correlated_fit/
```

This workflow was created as appendix/debug material only. It uses manually
selected Gaussian-refined head-echo measurements plus retained trail positions,
with block covariance terms for overlapping coherent integrations: 16 code
periods for head echoes and 8 code periods for trail echoes, stepped by one
code period. Trail interferometric-position covariance is additionally gated by
the raw 3 km range gate: measurements in different raw range gates are treated
as uncorrelated even when their coherent integrations overlap in time. Because
atmospheric drag is no longer to be estimated, update this
workflow before using it again: remove the fitted \(C_D A/m\) state and use the
fixed 1.6 m, 5 metric ton spherical-body assumption instead.

The earlier correlated-fit uncertainty bands in
`ballistic_head_trail_correlated_altitude_enu.png` exploded because the
linearized covariance sampled an essentially unconstrained, bounded drag
parameter as an unconstrained Gaussian variable. That diagnostic band should
not be interpreted physically and should not be used in the manuscript without
first removing fitted drag or otherwise replacing the uncertainty model.

The manually selected head-echo measurement workflow is separate from the main
ROI-derived fit and should not overwrite existing products. The current Python
selection workflow is:

```text
reports/select_head_echo_measurements.py
reports/head_echo_python_selection_workflow_ext0p5/
reports/figures/head_echo_python_selection_range_doppler.png
```

When comparing selected head-echo measurements to existing trajectory data, use
the same ROI-derived observations used in `fig:trajectory-observables`, namely
`products/head_echo_trajectory_roi_fit/head_echo_trajectory.json`. Do not
compare against local Gaussian seed picks unless that is explicitly the point
of the diagnostic.

The local working tree may intentionally contain uncommitted trail-ROI and
micro-Doppler work, including regenerated `trail_echo_roi.png`, altitude-label
annotation files, and `reports/generate_head_microdoppler.py`. Preserve those
unless explicitly asked to clean or commit them.

## Head-Echo Micro-Doppler / Gaussian Refinement Notes

The 16-code head-echo micro-Doppler/Gaussian refinement work is exploratory and
should be validated visually before any refined points are used in a trajectory
or ballistic fit. In particular, do not use old common-grid Gaussian caches
unless they were generated with the range prediction described below and then
visually inspected.

Do not overplot scatter-point markers on the head-echo micro-Doppler pcolormesh.
The diagnostic is meant to show weak Doppler substructure; marker overlays hide
the thing being inspected. Use annotations outside the image or subtle reference
lines only.

The range seed for predicted head-echo matched-filter patches must match the
dashed white range track in `reports/figures/trail_echo_roi.png`. That figure is
generated by `trail_echo_roi.py`, which calls
`estimate_head_echo_trajectory.predict_ranges(...)` with the ENU-at-transmitter
fit parameters from:

```text
products/head_echo_trajectory_roi_fit/head_echo_trajectory.json
```

Do not substitute the simpler ECEF helper model for this patch seed. The
predicted-grid Gaussian extractor must use the same `predict_ranges` and
`predict_dopplers` code path as `trail_echo_roi.py`; otherwise the matched
filter patches can be centered on the wrong range and show no actual head echo.

The invalid cache produced during the exploratory common-grid attempt is:

```text
products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_common_grid.h5
```

Treat any figures derived from that cache, especially
`reports/figures/head_echo_gaussian_fits/`, as diagnostic failures unless they
are regenerated after the extractor fix. A good smoke test is the Clinton
retained echo near 0.400 s after 18:06:20 UTC: the Figure-3 predicted seed is
about 263.45 km and the 2-D Gaussian fit should land on the visible echo near
261.28 km and -1957 Hz absolute Doppler.

For expensive extraction, sync only the needed scripts to
`juha@avaruus.uit.no:/home/juha/src/new_england_bolide` and run the retained
head-time extractor there, for example:

```bash
python reports/extract_head_echo_gaussian_patches.py \
  --stations beverly clinton pomfret \
  --retained-head-times \
  --workers 128 \
  --out products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5
```

Then copy the HDF5 back and run the local 2-D FFT notch/Gaussian fitting and
micro-Doppler plotting scripts. Inspect the per-patch plots before using any
corrections in a fit.

Extended head-echo search run completed 2026-06-04:

```text
remote output copied locally: products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_ext0p5.h5
remote log copied locally: products/logs/head_echo_gaussian_retained_ext0p5.log
time grid: each station's retained head-echo window expanded by 0.5 s before
           and 0.5 s after, with 0.01 s spacing
patch counts: Beverly 182, Clinton 134, Pomfret 173
Gaussian fit summary: reports/head_echo_gaussian_fit_summary_retained_ext0p5.csv
micro-Doppler figure: reports/figures/head_echo_microdoppler_ext0p5.png
validation contact sheet: reports/figures/head_echo_gaussian_validation_ext0p5_contact_sheet.png
full validation figures: reports/figures/head_echo_gaussian_fits_retained_ext0p5/
```

Live job launched 2026-06-04 from this workflow:

```text
host: juha@avaruus.uit.no
repo: /home/juha/src/new_england_bolide
main pid observed: 3603881
log: products/logs/head_echo_gaussian_retained_fig3seed.log
remote output: products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5
```

Check the job from any machine with:

```bash
ssh juha@avaruus.uit.no \
  "pgrep -af 'extract_head_echo_gaussian_patches.py.*retained-head-times' || true; \
   tail -40 /home/juha/src/new_england_bolide/products/logs/head_echo_gaussian_retained_fig3seed.log; \
   ls -lh /home/juha/src/new_england_bolide/products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5"
```

When complete, copy the cache back locally:

```bash
mkdir -p products/head_echo_gaussian_patches
scp juha@avaruus.uit.no:/home/juha/src/new_england_bolide/products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5 \
  products/head_echo_gaussian_patches/
```

Then regenerate the review figures and corrected micro-Doppler diagnostic:

```bash
.venv/bin/python reports/fit_head_echo_gaussian_patches.py \
  --cache products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5 \
  --out-dir reports/figures/head_echo_gaussian_fits_retained_fig3seed \
  --summary reports/head_echo_gaussian_fit_summary_retained_fig3seed.csv

.venv/bin/python reports/generate_head_microdoppler_corrected.py \
  --cache products/head_echo_gaussian_patches/head_echo_gaussian_patches_16code_retained_fig3seed.h5 \
  --out-fig reports/figures/head_echo_microdoppler_corrected.png \
  --summary reports/head_echo_microdoppler_corrected_summary.csv
```

Only after the retained Figure-3-seeded per-patch plots visibly show the head
echo should any Gaussian range/Doppler corrections be considered for a
trajectory or ballistic refit.

## Copying Report Assets To Overleaf

Any code in this repository that regenerates manuscript figures, tables,
bibliography files, report body text, or TeX support files should leave the
processing products here and then copy the manuscript-facing assets into the
Overleaf repository:

```text
/home/j/src/cape_cod_bay/reports/
```

Use the same relative layout under `reports/` in both repositories. In
particular, generated manuscript images from this repo's `reports/figures/`
must be copied to:

```text
/home/j/src/cape_cod_bay/reports/figures/
```

The manuscript in `../cape_cod_bay/main.tex` uses only local paths such as
`reports/processing_report_body.tex`, `reports/report_stats.tex`, and
`reports/figures/*.png`; Overleaf cannot see files in this processing
repository.

A safe sync command for manuscript assets is:

```bash
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.aux' \
  --exclude '*.bcf' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.run.xml' \
  --exclude 'texput.log' \
  reports/ ../cape_cod_bay/reports/
```

If the CAS template files are updated, also copy these to the root of the
Overleaf repo:

```bash
cp reports/cas-sc.cls reports/cas-common.sty reports/cas-model2-names.bst ../cape_cod_bay/
```

In short:

- Processing, products, regenerated figure/table sources: this repo.
- Manuscript driver, copied report assets, and Overleaf sync: `../cape_cod_bay/`.
- Do not recreate `reports/processing_report.tex` here as the canonical
  manuscript entry point; update `../cape_cod_bay/main.tex`.

## 2026-06-12 RDMF AoD/Polarization Server Run

Dinner handoff state, written 2026-06-12 20:39 CST.

The routine RDMF processing chain was updated and pushed to Git:

```text
be10bb6 Add trajectory steered RDMF polarization mode
```

Files in the focused commit:

```text
range_doppler_mf_bolide.py
run_rdmf_stations.py
```

What changed:

- `range_doppler_mf_bolide.py` now has standard-chain TX AoD steering via
  `--tx-aod-beamform`, using the best-fit trajectory JSON to compute the
  physical transmit-array steering vector.
- It now supports `--polarization-combine`. The order is: form the
  trajectory-steered TX beam, incoherently average the RX polarizations for peak
  selection, select the peak head-echo pixel, derive one RX polarization vector
  from that peak pixel, and apply that same polarization vector to the full
  range-Doppler window.
- No arbitrary gain is applied to the data. The coherent TX voltage sum is
  normalized by the number of TX codes when forming power, so the noise scale is
  kept comparable while coherent SNR improvement remains in the signal.
- Doppler whitening is now off by default in both the worker and station runner.
- HDF5 outputs include diagnostics: `pol_gamma`, `pol_peak_range_idx`,
  `pol_peak_doppler_idx`, `tx_aod_unit_enu`, `tx_aod_beamform`,
  `tx_aod_phase_sign`, `polarization_combine`, and `antenna_pos_tx`.

Remote host:

```text
juha@avaruus.uit.no
/home/juha/src/new_england_bolide
```

The server checkout is at `be10bb6`. Note: the server did not expose a conda
executable over SSH, but `/home/juha/venv/bin/python` has `numpy`, `h5py`,
`yaml`, and `matplotlib`, so the active run uses that interpreter.

Running server job:

```text
parent PID: 2011338
launcher shell PID: 2011336
station workers at launch: 2011339, 2011341, 2011343, 2011345, 2011347
```

Output directory:

```text
products/rdmf_fft16_txaod_pol_standard_be10bb6/
```

Log:

```text
logs/rdmf_fft16_txaod_pol_standard_be10bb6.log
```

Command that was started on the server:

```bash
/home/juha/venv/bin/python run_rdmf_stations.py \
  --out-dir products/rdmf_fft16_txaod_pol_standard_be10bb6 \
  --stations amherstnh beverly clinton pomfret siena \
  --start-second 75 \
  --seconds 10 \
  --n-coh 1 \
  --codes-per-step 1 \
  --range-interp 1 \
  --doppler-zero-pad-factor 16 \
  --range-resolution-m 3000 \
  --fmin -12000 \
  --fmax 12000 \
  --ignore-freq 150 \
  --head-min-doppler-hz 250 \
  --no-doppler-whiten \
  --tx-aod-beamform \
  --tx-aod-trajectory-json products/ballistic_head_trail_correlated_fit_v0p2p7_zp16_neighbor0p5/ballistic_head_trail_correlated_fit.json \
  --tx-aod-phase-sign -1 \
  --polarization-combine \
  --progress-every 25 \
  --checkpoint-every 100 \
  --plot-every 120 \
  --snr-threshold 3.5
```

At the latest check before this handoff was committed, all five station workers
had reached checkpoint 100/999. Checkpoint HDF5 files and refreshed RTI PNGs
had been written for all five stations. The first estimate was about
13--15 minutes per station, running in parallel.

Useful continuation commands:

```bash
ssh juha@avaruus.uit.no 'pgrep -af "range_doppler_mf_bolide|run_rdmf_stations"'
ssh juha@avaruus.uit.no 'tail -n 120 ~/src/new_england_bolide/logs/rdmf_fft16_txaod_pol_standard_be10bb6.log'
rsync -av juha@avaruus.uit.no:~/src/new_england_bolide/products/rdmf_fft16_txaod_pol_standard_be10bb6/ products/rdmf_fft16_txaod_pol_standard_be10bb6/
```
