# DEV_STATE

Session checkpoint for coding agents continuing work on `fitfile-hr-artifact-corrector`.

## Current status

- Core package implemented under `src/hf_corrector/`.
- Sequence-model training path is active (`torch_gru`) with dynamics-aware loss.
- CLI supports end-to-end flow: `qa-chest`, `train`, `correct`, `export`, `analyze`, `self-check`.
- Plotting script supports:
  - HR original vs corrected
  - optional power overlay from FIT
  - optional middle-window slicing
  - corrected-region shading on x-axis

## Important files

- `src/hf_corrector/model.py`
  - `HRModel.fit_from_groups(...)` (GRU training)
  - strict device behavior: `--device mps` fails if unavailable.
- `src/hf_corrector/training.py`
  - reads clean chest manifest and trains sequence model.
- `src/hf_corrector/corrector.py`
  - artifact-driven replacement, per-ride calibration, spike suppression,
    physiological dynamics constraints.
- `src/hf_corrector/detector.py`
  - artifact scoring rules.
- `scripts/train_seq_mps.sh`
  - default and `stable` training presets.
- `scripts/batch_correct_optical.sh`
  - batch correction/export loop.
- `scripts/plot_hr_comparison.py`
  - rendering utility used for current analysis plots.

## Data and artifacts (local)

- `hr-chest/` chest-strap rides (training source).
- `hr-optical/` optical rides (correction targets).
- `outputs/train_manifest_clean.json` created via chest QA.
- Trained models currently present (local):
  - `models/model_seq_v1_mps`
  - `models/model_seq_v1_mps_stable`

These folders are ignored in git and should not be committed.

## Last validated workflow

1. Train (outside sandbox):
   - `./scripts/train_seq_mps.sh ... stable ...`
2. Correct target file:
   - `hf-corrector correct --fit ... --model ... --mode balanced --hr-bias 8 --out correction_seq_stable.json`
3. Plot middle 30 min with power and corrected regions:
   - `scripts/plot_hr_comparison.py --correction-json ... --fit ... --window-minutes 30 --window-center middle ...`

## Known issues / caveats

- FIT rewrite (`fit-tool`) fails for some FIT files with strict field-size parser errors.
  - Current behavior: export still writes CSV and returns `fit_error` instead of crashing.
- Bias correction (`--hr-bias`) is currently manual and global for replaced points only.

## Suggested next steps

1. Add objective evaluation command (before/after metrics) for labeled chest+optical overlap sessions.
2. Add automatic per-ride bias estimation from stable non-artifact anchor regions.
3. Consider replacing FIT rewrite backend for higher compatibility with vendor FIT variants.
4. Add unit tests for dynamics constraints and corrected-region plotting.
