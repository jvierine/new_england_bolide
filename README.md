# New England Bolide

Range-Doppler matched-filter and transmit angle-of-departure tools for the
2026-05-30 New England bolide SIMONe/Haystack coded-CW data.

## Inputs

The scripts expect the raw data directory layout under `2026_05_30_bolide/`.
The YAML files in that directory are tracked because they contain code seeds,
site coordinates, TX antenna ENU offsets, radar frequency, code length, and
range resolution. Raw `.h5` files and generated products are ignored by git.

## Data

Raw data will be archived on Zenodo.

Zenodo DOI: TBD

After downloading the archive, place it at the repository root as
`2026_05_30_bolide.tar.gz`.  The reproduction script below will extract it
when needed.  After extraction, the raw data are available under:

```text
2026_05_30_bolide/
```

## Install

This project was set up with Python 3.13.7. To recreate the local environment:

```bash
cd /Users/j/src/new_england_bolide
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the scientific Python dependencies import correctly:

```bash
python -c "import astropy, dasst, h5py, matplotlib, numpy, scipy, yaml; print('ok')"
```

## Reproduce The Report

From a fresh clone plus `2026_05_30_bolide.tar.gz`, rebuild the products,
figures, tables, and PDF with:

```bash
./scripts/reproduce_report.sh --clean 2026_05_30_bolide.tar.gz
```

Set `MAX_PARALLEL` to control the high-resolution matched-filter batch
parallelism:

```bash
MAX_PARALLEL=32 ./scripts/reproduce_report.sh --clean 2026_05_30_bolide.tar.gz
```

The script creates `.venv`, installs `requirements.txt`, extracts the raw
archive, regenerates the high-resolution RDMF products, fits the head-echo
trajectory, recomputes the orbit/source association products, regenerates all
report figures/tables, and rebuilds `reports/processing_report.pdf`.

All non-raw catalog and ephemeris inputs used by the report are tracked under
`data/`, including `NEA.txt`, `streamestablisheddata2026.txt`, and
`de440s.bsp`.  The raw `.h5` radar files and generated `products/` are not
tracked; they are reconstructed from `2026_05_30_bolide.tar.gz`.

## Range-Doppler Products

Full station-parallel matched-filter run:

```bash
python run_rdmf_stations.py \
  --start-second 0 \
  --codes-per-step 8 \
  --range-interp 1 \
  --checkpoint-every 100 \
  --progress-every 50 \
  --out-dir products/full_run
```

High-resolution run around the head echo:

```bash
python run_rdmf_stations.py \
  --start-second 75 \
  --seconds 10 \
  --codes-per-step 1 \
  --range-interp 2 \
  --max-range-km 600 \
  --ignore-freq 0 \
  --fmin -50000 \
  --fmax 50000 \
  --out-dir products/highres_180615_10s
```

## Angle Of Departure

Single pick from an RDMF product:

```bash
python angle_departure_estimator.py \
  --rdmf-file products/full_run/rdmf_clinton_0000s_all.h5 \
  --pick-max \
  --channels 0 \
  --range-interp 1 \
  --out products/aod/aod_clinton_peak.h5 \
  --plot
```

Peak-Doppler AoD bank from 18:06:20 UTC to the end of the local snippet:

```bash
python aod_peak_bank.py parallel \
  --start-second 80 \
  --codes-per-step 1 \
  --range-interp 2 \
  --snr-threshold-db 6 \
  --grid-size 121 \
  --max-zenith-angle 90 \
  --out-dir products/aod_peak_180620_to_end
```

The AoD bank stores both `voltage_mf` and `voltage_lstsq` complex products with
shape `[time, channel, range, tx_code]`, plus peak SNR, peak Doppler, and AoD
estimates per range gate.

For a many-core server, split the 18:06:20-to-end interval into independent
station/time chunks. This avoids shared HDF5 writes and can use many cores:

```bash
python aod_peak_bank.py batch \
  --start-second 80 \
  --total-seconds 100 \
  --chunk-seconds 1 \
  --max-parallel 128 \
  --codes-per-step 1 \
  --range-interp 2 \
  --snr-threshold-db 6 \
  --grid-size 121 \
  --max-zenith-angle 90 \
  --out-dir products/aod_peak_180620_to_end_batch
```

This writes one file per station/chunk. Station range windows are built in:
Beverly 0-300 km, Clinton 0-300 km, Pomfret 300-450 km, Siena 400-500 km, and
Amherst 0-350 km.

The AoD search grid follows the PANSY interferometry style: a regular grid in
horizontal direction cosines over a zenith-angle cap, with elevation and
azimuth derived from the resulting ENU unit vectors.

## Manual Head Echo ROI Labels

Use the Matplotlib GUI to mark which stations contain a head echo and the
range-time box to use for trajectory estimation:

```bash
python label_head_echo_roi.py \
  --merged-dir products/rdmf_exact_highres_180615_10s_merged_20260602_095431 \
  --out annotations/head_echo_rois.json
```

Draw a box on the RTI panel, then click `Save ROI` or press `s`. Click
`No Head Echo` or press `n` when the station has no usable head echo. Use the
arrow keys or the `Previous`/`Next` buttons to move between stations. The JSON
stores the time/range box plus Doppler and SNR summary statistics from the
selected bins.

## Orbit Determination

The event-specific orbit sampler uses the fitted head-echo trajectory JSON,
samples its linearized ENU position/velocity covariance, converts samples to
ECEF/ITRS, and propagates pre-encounter orbits with `dasst`/`rebound`.

The default command expects the trajectory fit at
`products/head_echo_trajectory_roi_fit/head_echo_trajectory.json` and the
tracked JPL ephemeris kernel at `data/de440s.bsp`:

```bash
python orbit_det_sample.py \
  --samples 200 \
  --plot \
  --out products/orbit_determination/head_echo_orbit_samples.h5
```

Use a different ephemeris kernel with:

```bash
python orbit_det_sample.py --kernel /path/to/de440s.bsp
```

The output HDF5 stores sampled ENU states, ECEF states, final propagated HCRS
states, final Keplerian elements in ICRS and heliocentric mean ecliptic frames,
radiant estimates, and sample means/standard deviations. Add
`--save-propagation` to also store full propagation histories.
