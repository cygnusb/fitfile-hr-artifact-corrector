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

The project uses a **sequence model (GRU)** with two supported training modes:

- **Chest-only baseline**: train on clean chest-strap rides (`hr-chest/`) to learn general HR dynamics.
- **Combined training**: mix chest-only rides with paired tours under `touren/`, where each `touren/<name>/` contains one `*_chest.fit.gz` and one `*_optical.fit.gz`. In paired tours, the model uses **optical features as input** and the matched **chest HR as target**.

Common model behavior:

- Inputs per timestep (16 features): `power`, `cadence`, `speed`, `altitude`, `grade`, rolling means (30s/60s/120s windows), and temporal derivatives (`d_power`, `d_speed`, `d_altitude`).
- Sequence training with dynamics-aware loss to discourage unrealistic rapid HR drops.
- **OneCycleLR** learning-rate scheduler for better convergence.
- **Validation split** (15%) with **early stopping** (patience 5) to prevent overfitting.
- **MC-Dropout** uncertainty estimation for model-based confidence scores.
- **Overlapping-window inference** to stay close to training sequence length on long rides.
- **Adaptive artifact detector** derives per-ride thresholds from the ride's own HR/power distribution.
- Inference combines:
  - model prediction (overlapping windows)
  - adaptive artifact detector
  - per-ride calibration
  - MC-Dropout confidence
  - post-filters for residual downward spikes

## Repository layout

- `src/hf_corrector/` core package
- `scripts/` helper scripts (training, batch correction, plotting)
- `outputs/` generated reports and correction files (ignored)
- `models/` trained model artifacts (ignored)
- `hr-chest/`, `hr-optical/`, `touren/` local datasets (ignored)

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
  --chest-dir hr-chest \
  --out-report outputs/chest_qa_report.json \
  --out-manifest outputs/train_manifest_clean.json
```

## 2) Train ML model

### Option A: Chest-only baseline

### Recommended (Apple Silicon MPS)

```bash
./scripts/train_seq_mps.sh
```

Stable preset (stronger smoothing / dynamics constraints):

```bash
./scripts/train_seq_mps.sh \
  models/model_seq_v1_mps_stable \
  stable \
  outputs/train_manifest_clean.json
```

### Manual training command

```bash
PYTHONPATH=src python -m hf_corrector.cli train \
  --manifest outputs/train_manifest_clean.json \
  --out models/model_seq_v1_mps \
  --device mps \
  --seq-len 120 \
  --stride 10 \
  --epochs 12 \
  --batch-size 64 \
  --lr 0.001 \
  --lambda-dynamics 0.35 \
  --max-drop-bpm-per-step 1.2 \
  --max-windows 40000 \
  --patience 5 \
  --val-fraction 0.15
```

If you pass `--device mps` and MPS is unavailable, training fails intentionally.

### Option B: Combined training from `hr-chest/` and paired `touren/`

Expected paired-tour layout:

```text
touren/
  tour1/
    476019853281493296_chest.fit.gz
    476019863744184622_optical.fit.gz
  tour2/
    ..._chest.fit.gz
    ..._optical.fit.gz
```

The combined pipeline:

- loads all chest-only FITs from `hr-chest/`
- discovers all paired tours under `touren/`
- aligns optical and chest records by nearest timestamp within a configurable tolerance
- trains on chest-only groups plus paired `optical -> chest` groups
- keeps repeated paired groups on the same side of the validation split

Recommended (Apple Silicon MPS):

```bash
./scripts/train_combined_mps.sh \
  hr-chest \
  touren \
  models/model_seq_combined_mps \
  stable
```

Manual command:

```bash
PYTHONPATH=src python -m hf_corrector.cli train-combined \
  --chest-dir hr-chest \
  --tours-dir touren \
  --out models/model_seq_combined_mps \
  --device mps \
  --seq-len 180 \
  --stride 8 \
  --epochs 16 \
  --batch-size 64 \
  --lr 0.0008 \
  --lambda-dynamics 0.55 \
  --max-drop-bpm-per-step 0.9 \
  --max-windows 50000 \
  --patience 5 \
  --val-fraction 0.15 \
  --pair-match-max-seconds 2.0 \
  --paired-weight 4 \
  --min-paired-points 900
```

Combined training writes `dataset_report.json` into the model directory with pairing stats and accepted/rejected tours.

## 3) Correct a FIT file

```bash
hf-corrector correct \
  --fit activity.fit.gz \
  --model models/model_seq_v1_mps_stable \
  --mode balanced \
  --hr-bias 8 \
  --out outputs/run/correction.json
```

Notes:
- `--hr-bias` is applied only to replaced points.
- `--mode`: `safe | balanced | aggressive`.

## 4) Export correction results

```bash
hf-corrector export \
  --fit activity.fit.gz \
  --correction outputs/run/correction.json \
  --out-dir outputs/run \
  --formats csv fit
```

- `audit.csv` is always produced.
- FIT rewrite can fail on some files due to strict parser constraints in `fit-tool`; export returns `fit_error` and continues.

## 5) Plot HR before/after

Full activity with power + corrected-region shading:

```bash
PYTHONPATH=src python scripts/plot_hr_comparison.py \
  --correction-json outputs/run/correction.json \
  --fit activity.fit.gz \
  --out outputs/run/hr_power_full.png \
  --title "Full ride: corrected HR + power"
```

Middle 30-minute window:

```bash
PYTHONPATH=src python scripts/plot_hr_comparison.py \
  --correction-json outputs/run/correction.json \
  --fit activity.fit.gz \
  --window-minutes 30 \
  --window-center middle \
  --out outputs/run/hr_power_middle30.png \
  --title "Middle 30 min: corrected HR + power"
```

## Batch correction script

```bash
./scripts/batch_correct_optical.sh \
  models/model_seq_v1_mps_stable \
  hr-optical \
  outputs/optical_corrected_v2 \
  balanced
```

## MCP server

```bash
PYTHONPATH=src python -m hf_corrector.mcp_server --model-dir models/model_seq_v1_mps_stable
```

## Known limitations

- FIT rewrite (`fit-tool`) may fail for specific vendor FIT variants with malformed/edge-case field definitions.
- In those cases, use `audit.csv` and `correction.json` as the reliable outputs.
