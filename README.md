# New England Bolide

Range-Doppler matched-filter and transmit angle-of-departure tools for the
2026-05-30 New England bolide SIMONe/Haystack coded-CW data.

## Inputs

The scripts expect the raw data directory layout under `2026_05_30_bolide/`.
The YAML files in that directory are tracked because they contain code seeds,
site coordinates, TX antenna ENU offsets, radar frequency, code length, and
range resolution. Raw `.h5` files and generated products are ignored by git.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
  --az-step 2 \
  --el-step 2 \
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
  --az-step 2 \
  --el-step 2 \
  --out-dir products/aod_peak_180620_to_end_batch
```

This writes one file per station/chunk. Station range windows are built in:
Beverly 0-300 km, Clinton 0-300 km, Pomfret 300-450 km, Siena 400-500 km, and
Amherst 0-350 km.
