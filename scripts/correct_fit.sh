#!/usr/bin/env bash
# Correct HR artifacts in a single FIT file using model_seq_v1_mps_stable.
#
# Usage:
#   ./scripts/correct_fit.sh INPUT.fit [OUTPUT.fit]
#
# If OUTPUT.fit is omitted the corrected file is written next to INPUT as
# <stem>_corrected.fit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${ROOT_DIR}/models/model_seq_v1_mps_stable"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INPUT.fit [OUTPUT.fit]" >&2
  exit 1
fi

INPUT_FIT="$1"

if [[ ! -f "${INPUT_FIT}" ]]; then
  echo "ERROR: input file not found: ${INPUT_FIT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model not found at ${MODEL_PATH}" >&2
  echo "  Train it first with:" >&2
  echo "    ./scripts/train_seq_mps.sh models/model_seq_v1_mps_stable stable" >&2
  exit 1
fi

# Derive default output path next to the input file.
BASE="$(basename "${INPUT_FIT}")"
STEM="${BASE%.fit.gz}"
STEM="${STEM%.fit}"
DEFAULT_OUT="$(dirname "${INPUT_FIT}")/${STEM}_corrected.fit"
OUTPUT_FIT="${2:-${DEFAULT_OUT}}"

cd "${ROOT_DIR}"
source .venv/bin/activate

echo "Input:  ${INPUT_FIT}"
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_FIT}"
echo

python scripts/correct_to_fit.py \
  --fit    "${INPUT_FIT}" \
  --model  "${MODEL_PATH}" \
  --out    "${OUTPUT_FIT}"
