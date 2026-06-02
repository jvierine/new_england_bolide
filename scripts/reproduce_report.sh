#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_ARCHIVE="${1:-2026_05_30_bolide.tar.gz}"
CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
  RAW_ARCHIVE="${2:-2026_05_30_bolide.tar.gz}"
fi

MAX_PARALLEL="${MAX_PARALLEL:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CHUNK_DIR="products/rdmf_exact_highres_180615_10s_chunks"
MERGED_DIR="products/rdmf_exact_highres_180615_10s_merged_20260602_095431"
ORBIT_DIR="products/orbit_determination"

run() {
  echo
  echo "+ $*"
  "$@"
}

if [[ ! -f "$RAW_ARCHIVE" ]]; then
  echo "Missing raw archive: $RAW_ARCHIVE" >&2
  echo "Usage: $0 [--clean] [2026_05_30_bolide.tar.gz]" >&2
  exit 1
fi

if [[ $CLEAN -eq 1 ]]; then
  rm -rf products plots reports/figures reports/*.aux reports/*.bbl reports/*.bcf \
    reports/*.blg reports/*.log reports/*.out reports/*.run.xml
fi

if [[ ! -d "$VENV_DIR" ]]; then
  run "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
run python -m pip install --upgrade pip
run python -m pip install -r requirements.txt

if [[ ! -d 2026_05_30_bolide ]]; then
  run tar -xzf "$RAW_ARCHIVE"
fi

if [[ ! -f data/de440s.bsp ]]; then
  echo "Missing data/de440s.bsp. This kernel is tracked in the GitHub repo and should be present after clone." >&2
  exit 1
fi

mkdir -p products reports/figures "$ORBIT_DIR"

run python reports/generate_report_assets.py

run python rdmf_peak_batch.py batch \
  --data-dir 2026_05_30_bolide \
  --start-second 75 \
  --total-seconds 10 \
  --chunk-seconds 0.1 \
  --max-parallel "$MAX_PARALLEL" \
  --codes-per-step 1 \
  --range-interp 4 \
  --range-min-km 0 \
  --range-max-km 600 \
  --fmin -50000 \
  --fmax 50000 \
  --ignore-freq 0 \
  --doppler-whiten \
  --out-dir "$CHUNK_DIR" \
  --overwrite

run python rdmf_peak_batch.py merge \
  --in-dir "$CHUNK_DIR" \
  --out-dir "$MERGED_DIR" \
  --overwrite

PLOT_DIR="plots/rdmf_exact_highres_180615_10s_20260602_095431"
mkdir -p "$PLOT_DIR"
for station in amherstnh beverly clinton pomfret siena; do
  run python plot_rdmf_bolide.py \
    "$MERGED_DIR/rdmf_peak_${station}_merged.h5" \
    --snr-threshold 5 \
    --out-prefix "$PLOT_DIR/rdmf_peak_${station}_merged"
done

run python reports/generate_station_rti_figures.py

run python estimate_head_echo_trajectory.py \
  --annotations annotations/head_echo_rois.json \
  --data-dir 2026_05_30_bolide \
  --out-dir products/head_echo_trajectory_roi_fit \
  --covariance-samples 800 \
  --no-show

run python reports/generate_trajectory_report_assets.py

run python orbit_det_sample.py \
  --trajectory-json products/head_echo_trajectory_roi_fit/head_echo_trajectory.json \
  --data-dir 2026_05_30_bolide \
  --kernel data/de440s.bsp \
  --samples 200 \
  --plot \
  --out "$ORBIT_DIR/best_fit_orbit.h5"

run python orbit_det_sample.py \
  --trajectory-json products/head_echo_trajectory_roi_fit/head_echo_trajectory.json \
  --data-dir 2026_05_30_bolide \
  --kernel data/de440s.bsp \
  --samples 40 \
  --out "$ORBIT_DIR/orbit_mc_40.h5"

run python nea_jopek_search.py
run python meteor_source_search.py
run python reports/generate_orbit_source_table.py
run python plot_orbit_xy.py --out reports/figures/orbit_xy_bolide.png
run python plot_shower_radiants.py

(
  cd reports
  run pdflatex -interaction=nonstopmode processing_report.tex
  run biber processing_report
  run pdflatex -interaction=nonstopmode processing_report.tex
  run pdflatex -interaction=nonstopmode processing_report.tex
)

echo
echo "Rebuilt reports/processing_report.pdf"
