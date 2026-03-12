from __future__ import annotations

import csv
import gzip
import shutil
import struct
from datetime import datetime
from pathlib import Path

import fitdecode
from fitdecode.utils import compute_crc

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
    """Rewrite FIT heart_rate values using in-place binary patching via fitdecode."""
    input_path = _resolve_fit_path(input_fit)
    data = bytearray(input_path.read_bytes())

    # local_mesg_num → byte offset of heart_rate field within data body (after 1-byte header)
    record_hr_offsets: dict[int, int] = {}
    patches: list[tuple[int, int]] = []

    with fitdecode.FitReader(str(input_path), keep_raw_chunks=True) as reader:
        for frame in reader:
            if isinstance(frame, fitdecode.FitDefinitionMessage):
                if frame.name != "record":
                    continue
                byte_offset = 0
                hr_offset = None
                for field_def in frame.field_defs:
                    if field_def.name == "heart_rate":
                        hr_offset = byte_offset
                        break
                    byte_offset += field_def.size
                if hr_offset is not None:
                    record_hr_offsets[frame.local_mesg_num] = hr_offset
                else:
                    record_hr_offsets.pop(frame.local_mesg_num, None)

            elif isinstance(frame, fitdecode.FitDataMessage):
                if frame.name != "record":
                    continue
                if frame.local_mesg_num not in record_hr_offsets:
                    continue
                try:
                    ts = frame.get_value("timestamp")
                except KeyError:
                    continue
                if not isinstance(ts, datetime):
                    continue
                corrected = corrections_by_ts.get(ts.isoformat())
                if corrected is None:
                    continue
                new_hr = max(0, min(255, int(round(corrected))))
                abs_offset = frame.chunk.offset + 1 + record_hr_offsets[frame.local_mesg_num]
                patches.append((abs_offset, new_hr))

    for abs_offset, new_hr in patches:
        data[abs_offset] = new_hr

    new_crc = compute_crc(data, start=0, end=len(data) - 2)
    struct.pack_into("<H", data, len(data) - 2, new_crc)

    out = Path(output_fit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
