#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${ROOT_DIR}/models/model_baseline_v2_torch_mps"
INPUT_DIR="${ROOT_DIR}/hr-optical"
OUTPUT_ROOT="${ROOT_DIR}/outputs/optical_corrected_v2"
MODE="balanced"

if [[ $# -gt 0 ]]; then
  MODEL_PATH="$1"
fi
if [[ $# -gt 1 ]]; then
  INPUT_DIR="$2"
fi
if [[ $# -gt 2 ]]; then
  OUTPUT_ROOT="$3"
fi
if [[ $# -gt 3 ]]; then
  MODE="$4"
fi

cd "${ROOT_DIR}"
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p "${OUTPUT_ROOT}"

find "${INPUT_DIR}" -type f \( -name "*.fit" -o -name "*.fit.gz" \) | sort | while read -r file; do
  base="$(basename "${file}")"
  stem="${base%.fit.gz}"
  stem="${stem%.fit}"
  out_dir="${OUTPUT_ROOT}/${stem}"
  mkdir -p "${out_dir}"

  echo "Processing ${base}"

  hf-corrector correct \
    --fit "${file}" \
    --model "${MODEL_PATH}" \
    --mode "${MODE}" \
    --out "${out_dir}/correction.json"

  hf-corrector export \
    --fit "${file}" \
    --correction "${out_dir}/correction.json" \
    --out-dir "${out_dir}" \
    --formats csv fit
done

echo "Done. Outputs in ${OUTPUT_ROOT}"
