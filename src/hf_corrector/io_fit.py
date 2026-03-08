from __future__ import annotations

import csv
import gzip
import shutil
from datetime import datetime
from pathlib import Path

import fitdecode

from .types import FitRecord


def _resolve_fit_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".gz":
        out = Path("/tmp") / p.with_suffix("").name
        with gzip.open(p, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out
    return p


def load_fit_records(path: str | Path) -> list[FitRecord]:
    fit_path = _resolve_fit_path(path)
    records: list[FitRecord] = []
    with fitdecode.FitReader(str(fit_path)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name != "record":
                continue
            data = {field.name: field.value for field in frame.fields}
            timestamp = data.get("timestamp")
            if not isinstance(timestamp, datetime):
                continue
            speed = data.get("enhanced_speed", data.get("speed"))
            altitude = data.get("enhanced_altitude", data.get("altitude"))
            grade = data.get("grade")
            records.append(
                FitRecord(
                    timestamp=timestamp,
                    heart_rate=_safe_float(data.get("heart_rate")),
                    power=_safe_float(data.get("power")),
                    cadence=_safe_float(data.get("cadence")),
                    speed=_safe_float(speed),
                    altitude=_safe_float(altitude),
                    grade=_safe_float(grade),
                    raw=data,
                )
            )
    return records


def write_audit_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rewrite_fit_heart_rate(
    input_fit: str | Path,
    output_fit: str | Path,
    corrections_by_ts: dict[str, float],
) -> None:
    """Rewrite FIT heart_rate values if optional fit-tool is available."""
    try:
        from fit_tool.fit_file import FitFile
        from fit_tool.profile.messages.record_message import RecordMessage
    except ImportError as exc:
        raise RuntimeError(
            "FIT rewrite requires optional dependency 'fit-tool'. "
            "Install with: pip install -e .[rewrite]"
        ) from exc

    input_path = _resolve_fit_path(input_fit)
    fitfile = FitFile.from_file(str(input_path))
    for msg in fitfile.messages:
        if not isinstance(msg, RecordMessage):
            continue
        ts = msg.get_value("timestamp")
        if ts is None:
            continue
        key = ts.isoformat()
        corrected = corrections_by_ts.get(key)
        if corrected is None:
            continue
        msg.set_value("heart_rate", int(round(corrected)))

    out = Path(output_fit)
    out.parent.mkdir(parents=True, exist_ok=True)
    fitfile.to_file(str(out))


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
