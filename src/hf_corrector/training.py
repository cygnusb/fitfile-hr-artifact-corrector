from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import build_feature_matrix, extract_target_hr
from .io_fit import load_fit_records
from .model import HRModel
from .types import FitRecord


@dataclass(slots=True)
class PreparedTrainingGroup:
    x: np.ndarray
    y: np.ndarray
    group_id: str
    source: str


@dataclass(slots=True)
class PairedTourReport:
    tour_dir: str
    optical_file: str
    chest_file: str
    optical_records: int
    chest_records: int
    matched_records: int
    match_ratio: float
    accepted: bool
    reason: str | None = None


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
    patience: int = 5,
    val_fraction: float = 0.15,
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
        patience=patience,
        val_fraction=val_fraction,
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


def train_from_combined_directories(
    chest_dir: str | Path,
    tours_dir: str | Path,
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
    patience: int = 5,
    val_fraction: float = 0.15,
    pair_match_max_seconds: float = 2.0,
    paired_weight: int = 3,
    min_paired_points: int = 600,
) -> dict[str, Any]:
    groups, report = prepare_combined_training_groups(
        chest_dir=chest_dir,
        tours_dir=tours_dir,
        pair_match_max_seconds=pair_match_max_seconds,
        paired_weight=paired_weight,
        min_paired_points=min_paired_points,
    )
    if not groups:
        raise ValueError("No training groups available from chest_dir/tours_dir inputs")

    x_groups = [group.x for group in groups]
    y_groups = [group.y for group in groups]
    group_ids = [group.group_id for group in groups]

    model = HRModel.fit_from_groups(
        x_groups,
        y_groups,
        group_ids=group_ids,
        device=device,
        seq_len=seq_len,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_dynamics=lambda_dynamics,
        max_drop_bpm_per_step=max_drop_bpm_per_step,
        max_windows=max_windows,
        patience=patience,
        val_fraction=val_fraction,
    )

    metadata = {
        "train_files": report["dataset"]["chest_files_total"] + report["dataset"]["paired_tours_accepted"],
        "train_rows": report["dataset"]["chest_rows"] + report["dataset"]["paired_rows"],
        "device": model.train_device,
        "model_type": model.backend,
        "seq_len": seq_len,
        "stride": stride,
        "paired_weight": paired_weight,
        "pair_match_max_seconds": pair_match_max_seconds,
        "min_paired_points": min_paired_points,
        "dataset_sources": {
            "chest_only_files": report["dataset"]["chest_files_total"],
            "paired_tours_accepted": report["dataset"]["paired_tours_accepted"],
            "paired_tours_total": report["dataset"]["paired_tours_total"],
            "paired_rows": report["dataset"]["paired_rows"],
        },
        "note": "Combined setup: chest-only baseline plus paired optical->chest supervision",
    }
    model.save(out_dir, metadata=metadata)

    model_dir = Path(out_dir)
    (model_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "model_dir": str(model_dir),
        "train_groups": len(groups),
        "train_rows": metadata["train_rows"],
        "device": metadata["device"],
        "paired_tours_accepted": report["dataset"]["paired_tours_accepted"],
        "paired_rows": report["dataset"]["paired_rows"],
        "chest_files_total": report["dataset"]["chest_files_total"],
    }


def prepare_combined_training_groups(
    *,
    chest_dir: str | Path,
    tours_dir: str | Path,
    pair_match_max_seconds: float,
    paired_weight: int,
    min_paired_points: int,
) -> tuple[list[PreparedTrainingGroup], dict[str, Any]]:
    if paired_weight < 1:
        raise ValueError("paired_weight must be >= 1")
    if pair_match_max_seconds <= 0:
        raise ValueError("pair_match_max_seconds must be > 0")
    if min_paired_points < 1:
        raise ValueError("min_paired_points must be >= 1")

    groups: list[PreparedTrainingGroup] = []
    chest_files = sorted(_iter_fit_files(chest_dir))
    chest_rows = 0
    for fp in chest_files:
        recs = load_fit_records(fp)
        if not recs:
            continue
        chest_rows += len(recs)
        groups.append(
            PreparedTrainingGroup(
                x=build_feature_matrix(recs),
                y=extract_target_hr(recs),
                group_id=f"chest:{fp}",
                source="chest_only",
            )
        )

    paired_reports: list[dict[str, Any]] = []
    paired_rows = 0
    paired_tours_total = 0
    paired_tours_accepted = 0
    for tour_dir, optical_file, chest_file in _discover_paired_tours(tours_dir):
        paired_tours_total += 1
        optical_records = load_fit_records(optical_file)
        chest_records = load_fit_records(chest_file)

        aligned_optical, aligned_chest = _align_paired_records(
            optical_records,
            chest_records,
            max_gap_seconds=pair_match_max_seconds,
        )
        matched_records = len(aligned_optical)
        match_ratio = matched_records / max(1, len(optical_records))
        report = PairedTourReport(
            tour_dir=str(tour_dir),
            optical_file=str(optical_file),
            chest_file=str(chest_file),
            optical_records=len(optical_records),
            chest_records=len(chest_records),
            matched_records=matched_records,
            match_ratio=round(match_ratio, 4),
            accepted=matched_records >= min_paired_points,
            reason=None if matched_records >= min_paired_points else "too_few_matched_points",
        )
        paired_reports.append(_paired_report_to_dict(report))
        if matched_records < min_paired_points:
            continue

        x_pair = build_feature_matrix(aligned_optical)
        y_pair = extract_target_hr(aligned_chest)
        if len(x_pair) == 0:
            continue

        paired_tours_accepted += 1
        paired_rows += len(x_pair)
        group_id = f"paired:{tour_dir}"
        for _ in range(paired_weight):
            groups.append(
                PreparedTrainingGroup(
                    x=x_pair,
                    y=y_pair,
                    group_id=group_id,
                    source="paired_optical_to_chest",
                )
            )

    report = {
        "inputs": {
            "chest_dir": str(chest_dir),
            "tours_dir": str(tours_dir),
            "pair_match_max_seconds": pair_match_max_seconds,
            "paired_weight": paired_weight,
            "min_paired_points": min_paired_points,
        },
        "dataset": {
            "chest_files_total": len(chest_files),
            "chest_rows": chest_rows,
            "paired_tours_total": paired_tours_total,
            "paired_tours_accepted": paired_tours_accepted,
            "paired_rows": paired_rows,
            "training_groups_total": len(groups),
        },
        "paired_tours": paired_reports,
    }
    return groups, report


def _iter_fit_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return sorted([p for p in path.rglob("*") if p.name.endswith((".fit", ".fit.gz"))])


def _discover_paired_tours(root: str | Path) -> list[tuple[Path, Path, Path]]:
    tours_root = Path(root)
    if not tours_root.exists():
        raise FileNotFoundError(f"Directory does not exist: {tours_root}")

    pairs: list[tuple[Path, Path, Path]] = []
    for tour_dir in sorted([p for p in tours_root.iterdir() if p.is_dir()]):
        chest_candidates = sorted(_match_suffix_files(tour_dir, "_chest.fit", "_chest.fit.gz"))
        optical_candidates = sorted(_match_suffix_files(tour_dir, "_optical.fit", "_optical.fit.gz"))
        if len(chest_candidates) != 1 or len(optical_candidates) != 1:
            raise ValueError(
                f"Expected exactly one *_chest and one *_optical FIT file in {tour_dir}, "
                f"found chest={len(chest_candidates)} optical={len(optical_candidates)}"
            )
        pairs.append((tour_dir, optical_candidates[0], chest_candidates[0]))
    return pairs


def _match_suffix_files(tour_dir: Path, *suffixes: str) -> list[Path]:
    return [p for p in tour_dir.iterdir() if p.is_file() and p.name.endswith(suffixes)]


def _align_paired_records(
    optical_records: list[FitRecord],
    chest_records: list[FitRecord],
    *,
    max_gap_seconds: float,
) -> tuple[list[FitRecord], list[FitRecord]]:
    if not optical_records or not chest_records:
        return [], []

    chest_seconds = [record.timestamp.timestamp() for record in chest_records]
    aligned_optical: list[FitRecord] = []
    aligned_chest: list[FitRecord] = []
    last_chest_idx = -1

    for optical_record in optical_records:
        target_second = optical_record.timestamp.timestamp()
        idx = bisect_left(chest_seconds, target_second, lo=last_chest_idx + 1)
        candidates: list[int] = []
        if idx < len(chest_seconds):
            candidates.append(idx)
        if idx - 1 > last_chest_idx:
            candidates.append(idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda candidate: (abs(chest_seconds[candidate] - target_second), candidate))
        gap_seconds = abs(chest_seconds[best_idx] - target_second)
        if gap_seconds > max_gap_seconds:
            continue

        aligned_optical.append(optical_record)
        aligned_chest.append(chest_records[best_idx])
        last_chest_idx = best_idx

    return aligned_optical, aligned_chest


def _paired_report_to_dict(report: PairedTourReport) -> dict[str, Any]:
    return {
        "tour_dir": report.tour_dir,
        "optical_file": report.optical_file,
        "chest_file": report.chest_file,
        "optical_records": report.optical_records,
        "chest_records": report.chest_records,
        "matched_records": report.matched_records,
        "match_ratio": report.match_ratio,
        "accepted": report.accepted,
        "reason": report.reason,
    }
