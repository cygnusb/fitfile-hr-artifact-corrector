#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHEST_DIR="${ROOT_DIR}/hr-chest"
TOURS_DIR="${ROOT_DIR}/touren"
OUT_DIR="${ROOT_DIR}/models/model_seq_combined_mps"
PRESET="default"

if [[ $# -gt 0 ]]; then
  CHEST_DIR="$1"
fi
if [[ $# -gt 1 ]]; then
  TOURS_DIR="$2"
fi
if [[ $# -gt 2 ]]; then
  OUT_DIR="$3"
fi
if [[ $# -gt 3 ]]; then
  PRESET="$4"
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
PAIR_MATCH_MAX_SECONDS=2.0
PAIRED_WEIGHT=3
MIN_PAIRED_POINTS=600
PATIENCE=5
VAL_FRACTION=0.15

if [[ "${PRESET}" == "stable" ]]; then
  SEQ_LEN=180
  STRIDE=8
  EPOCHS=16
  BATCH_SIZE=64
  LR=0.0008
  LAMBDA_DYN=0.55
  MAX_DROP=0.9
  MAX_WINDOWS=50000
  PAIR_MATCH_MAX_SECONDS=2.0
  PAIRED_WEIGHT=4
  MIN_PAIRED_POINTS=900
fi

echo "Training preset: ${PRESET}"
echo "Chest dir: ${CHEST_DIR}"
echo "Tours dir: ${TOURS_DIR}"
echo "Output model: ${OUT_DIR}"

python -m hf_corrector.cli train-combined \
  --chest-dir "${CHEST_DIR}" \
  --tours-dir "${TOURS_DIR}" \
  --out "${OUT_DIR}" \
  --device mps \
  --seq-len "${SEQ_LEN}" \
  --stride "${STRIDE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lambda-dynamics "${LAMBDA_DYN}" \
  --max-drop-bpm-per-step "${MAX_DROP}" \
  --max-windows "${MAX_WINDOWS}" \
  --patience "${PATIENCE}" \
  --val-fraction "${VAL_FRACTION}" \
  --pair-match-max-seconds "${PAIR_MATCH_MAX_SECONDS}" \
  --paired-weight "${PAIRED_WEIGHT}" \
  --min-paired-points "${MIN_PAIRED_POINTS}"
