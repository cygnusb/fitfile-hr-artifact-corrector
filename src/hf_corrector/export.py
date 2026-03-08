from __future__ import annotations

import json
from pathlib import Path

from .io_fit import rewrite_fit_heart_rate, write_audit_csv
from .types import CorrectionPoint


def points_to_rows(points: list[CorrectionPoint]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in points:
        rows.append(
            {
                "timestamp": p.timestamp.isoformat(),
                "original_hr": p.original_hr,
                "corrected_hr": p.corrected_hr,
                "confidence": round(p.confidence, 4),
                "artifact_flag": p.artifact_flag,
                "artifact_probability": round(p.artifact_probability, 4),
            }
        )
    return rows


def save_correction_json(path: str | Path, points: list[CorrectionPoint], summary: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "points": points_to_rows(points),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_correction_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_outputs(
    input_fit: str | Path,
    out_dir: str | Path,
    points: list[CorrectionPoint],
    formats: list[str],
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = points_to_rows(points)
    produced: dict[str, str] = {}

    if "csv" in formats:
        csv_path = out / "audit.csv"
        write_audit_csv(csv_path, rows)
        produced["csv"] = str(csv_path)

    if "fit" in formats:
        fit_path = out / "corrected.fit"
        corrections_by_ts = {
            row["timestamp"]: float(row["corrected_hr"])
            for row in rows
            if row["corrected_hr"] is not None
        }
        try:
            rewrite_fit_heart_rate(input_fit, fit_path, corrections_by_ts)
            produced["fit"] = str(fit_path)
        except Exception as exc:
            produced["fit_error"] = str(exc)

    return produced
