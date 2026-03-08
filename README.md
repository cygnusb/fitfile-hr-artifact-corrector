# fitfile-hr-artifact-corrector

FIT heart-rate artifact correction toolkit with offline ML training, a CLI, and an MCP Server for tool-based integration.

## What this project does

- Parses `.fit` / `.fit.gz` activity files.
- Detects likely optical HR artifacts.
- Replaces artifact points with model predictions.
- Exports correction audit data (`original_hr`, `corrected_hr`, `confidence`, `artifact_flag`).
- Provides an MCP Server exposing correction/export tools for agent workflows.
- Plots HR before/after correction (optionally with power overlay and corrected-region highlighting).

The current pipeline is optimized for cycling FIT data.

## Current model approach

The project currently uses a **sequence model (GRU)** trained on chest-strap rides:

- Inputs per timestep: `power`, `cadence`, `speed`, `altitude`, `grade`, plus rolling features.
- Sequence training with dynamics-aware loss to discourage unrealistic rapid HR drops.
- Inference combines:
  - model prediction
  - artifact detector
  - per-ride calibration
  - post-filters for residual downward spikes

## Repository layout

- `src/hf_corrector/` core package
- `scripts/` helper scripts (training, batch correction, plotting)
- `outputs/` generated reports and correction files (ignored)
- `models/` trained model artifacts (ignored)
- `hr-chest/`, `hr-optical/` local datasets (ignored)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1) Build clean chest training manifest

Run quality-gating for chest-strap rides:

```bash
hf-corrector qa-chest \
  --chest-dir /abs/path/hr-chest \
  --out-report /abs/path/outputs/chest_qa_report.json \
  --out-manifest /abs/path/outputs/train_manifest_clean.json
```

## 2) Train ML model

### Recommended (Apple Silicon MPS)

```bash
./scripts/train_seq_mps.sh
```

Stable preset (stronger smoothing / dynamics constraints):

```bash
./scripts/train_seq_mps.sh \
  /abs/path/models/model_seq_v1_mps_stable \
  stable \
  /abs/path/outputs/train_manifest_clean.json
```

### Manual training command

```bash
PYTHONPATH=src python -m hf_corrector.cli train \
  --manifest /abs/path/outputs/train_manifest_clean.json \
  --out /abs/path/models/model_seq_v1_mps \
  --device mps \
  --seq-len 120 \
  --stride 10 \
  --epochs 12 \
  --batch-size 64 \
  --lr 0.001 \
  --lambda-dynamics 0.35 \
  --max-drop-bpm-per-step 1.2 \
  --max-windows 40000
```

If you pass `--device mps` and MPS is unavailable, training fails intentionally.

## 3) Correct a FIT file

```bash
hf-corrector correct \
  --fit /abs/path/activity.fit.gz \
  --model /abs/path/models/model_seq_v1_mps_stable \
  --mode balanced \
  --hr-bias 8 \
  --out /abs/path/outputs/run/correction.json
```

Notes:
- `--hr-bias` is applied only to replaced points.
- `--mode`: `safe | balanced | aggressive`.

## 4) Export correction results

```bash
hf-corrector export \
  --fit /abs/path/activity.fit.gz \
  --correction /abs/path/outputs/run/correction.json \
  --out-dir /abs/path/outputs/run \
  --formats csv fit
```

- `audit.csv` is always produced.
- FIT rewrite can fail on some files due to strict parser constraints in `fit-tool`; export returns `fit_error` and continues.

## 5) Plot HR before/after

Full activity with power + corrected-region shading:

```bash
PYTHONPATH=src python scripts/plot_hr_comparison.py \
  --correction-json /abs/path/outputs/run/correction.json \
  --fit /abs/path/activity.fit.gz \
  --out /abs/path/outputs/run/hr_power_full.png \
  --title "Full ride: corrected HR + power"
```

Middle 30-minute window:

```bash
PYTHONPATH=src python scripts/plot_hr_comparison.py \
  --correction-json /abs/path/outputs/run/correction.json \
  --fit /abs/path/activity.fit.gz \
  --window-minutes 30 \
  --window-center middle \
  --out /abs/path/outputs/run/hr_power_middle30.png \
  --title "Middle 30 min: corrected HR + power"
```

## Batch correction script

```bash
./scripts/batch_correct_optical.sh \
  /abs/path/models/model_seq_v1_mps_stable \
  /abs/path/hr-optical \
  /abs/path/outputs/optical_corrected_v2 \
  balanced
```

## MCP server

```bash
PYTHONPATH=src python -m hf_corrector.mcp_server --model-dir /abs/path/models/model_seq_v1_mps_stable
```

## Known limitations

- FIT rewrite (`fit-tool`) may fail for specific vendor FIT variants with malformed/edge-case field definitions.
- In those cases, use `audit.csv` and `correction.json` as the reliable outputs.
