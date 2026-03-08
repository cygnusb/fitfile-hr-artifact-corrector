#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT_DIR}/outputs/train_manifest_clean.json"
OUT_DIR="${ROOT_DIR}/models/model_seq_v1_mps"
PRESET="default"

if [[ $# -gt 0 ]]; then
  OUT_DIR="$1"
fi
if [[ $# -gt 1 ]]; then
  PRESET="$2"
fi
if [[ $# -gt 2 ]]; then
  MANIFEST="$3"
fi

cd "${ROOT_DIR}"
source .venv/bin/activate
export PYTHONPATH=src

SEQ_LEN=120
STRIDE=10
EPOCHS=12
BATCH_SIZE=64
LR=0.001
LAMBDA_DYN=0.35
MAX_DROP=1.2
MAX_WINDOWS=40000

if [[ "${PRESET}" == "stable" ]]; then
  # More smoothing / stronger dynamics constraint.
  SEQ_LEN=180
  STRIDE=8
  EPOCHS=16
  BATCH_SIZE=64
  LR=0.0008
  LAMBDA_DYN=0.55
  MAX_DROP=0.9
  MAX_WINDOWS=50000
fi

echo "Training preset: ${PRESET}"
echo "Output model: ${OUT_DIR}"
echo "Manifest: ${MANIFEST}"

python -m hf_corrector.cli train \
  --manifest "${MANIFEST}" \
  --out "${OUT_DIR}" \
  --device mps \
  --seq-len "${SEQ_LEN}" \
  --stride "${STRIDE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lambda-dynamics "${LAMBDA_DYN}" \
  --max-drop-bpm-per-step "${MAX_DROP}" \
  --max-windows "${MAX_WINDOWS}"
