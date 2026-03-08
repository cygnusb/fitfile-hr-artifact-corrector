from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import build_feature_matrix, extract_target_hr
from .io_fit import load_fit_records
from .model import HRModel


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML manifest needs PyYAML. Install with: pip install -e ."
            ) from exc
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data or {}

    return json.loads(p.read_text(encoding="utf-8"))


def train_from_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    device: str = "auto",
    seq_len: int = 120,
    stride: int = 10,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_dynamics: float = 0.35,
    max_drop_bpm_per_step: float = 1.2,
    max_windows: int = 40000,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    chest_files = manifest.get("chest_strap_files", [])
    if not chest_files:
        raise ValueError("manifest requires non-empty 'chest_strap_files'")

    records = []
    x_groups = []
    y_groups = []
    for fp in chest_files:
        recs = load_fit_records(fp)
        if not recs:
            continue
        records.extend(recs)
        x_groups.append(build_feature_matrix(recs))
        y_groups.append(extract_target_hr(recs))

    model = HRModel.fit_from_groups(
        x_groups,
        y_groups,
        device=device,
        seq_len=seq_len,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_dynamics=lambda_dynamics,
        max_drop_bpm_per_step=max_drop_bpm_per_step,
        max_windows=max_windows,
    )

    metadata = {
        "train_files": len(chest_files),
        "train_rows": int(len(records)),
        "device": model.train_device,
        "model_type": model.backend,
        "seq_len": seq_len,
        "stride": stride,
        "note": "Unpaired setup: model trained on chest-strap norm-HR behavior",
    }
    model.save(out_dir, metadata=metadata)

    return {
        "model_dir": str(Path(out_dir)),
        "train_files": len(chest_files),
        "train_rows": len(records),
        "device": metadata["device"],
    }
