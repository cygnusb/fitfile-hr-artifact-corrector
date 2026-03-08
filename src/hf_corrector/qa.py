from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .io_fit import load_fit_records


@dataclass(slots=True)
class QAThresholds:
    min_records: int = 800
    min_duration_min: float = 20.0
    min_distance_km: float = 10.0
    min_gps_ratio: float = 0.85
    min_hr_points_ratio: float = 0.9
    max_missing_hr_ratio: float = 0.1
    max_unrealistic_hr_ratio: float = 0.01
    max_zero_power_ratio: float = 0.7
    max_hr_jump_ratio: float = 0.02


def analyze_chest_directory(
    chest_dir: str | Path,
    out_report: str | Path,
    out_manifest: str | Path,
    thresholds: QAThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or QAThresholds()
    root = Path(chest_dir)
    files = sorted([p for p in root.rglob("*") if p.name.endswith((".fit", ".fit.gz"))])

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    parse_errors: list[dict[str, object]] = []

    for fp in files:
        try:
            records = load_fit_records(fp)
        except Exception as exc:
            parse_errors.append({"file": str(fp), "error": str(exc)})
            continue

        metrics = _compute_metrics(records)
        reasons = _reject_reasons(metrics, thresholds)
        row = {"file": str(fp), "metrics": metrics, "reasons": reasons}
        if reasons:
            rejected.append(row)
        else:
            accepted.append(row)

    manifest = {
        "chest_strap_files": [r["file"] for r in accepted],
        "optical_files": [],
    }

    summary = {
        "input_dir": str(root),
        "files_total": len(files),
        "files_parsed": len(files) - len(parse_errors),
        "parse_errors": len(parse_errors),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accept_ratio": round(len(accepted) / len(files), 4) if files else 0.0,
    }
    report = {
        "summary": summary,
        "thresholds": asdict(thresholds),
        "accepted": accepted,
        "rejected": rejected,
        "parse_errors": parse_errors,
    }

    out_report_path = Path(out_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_manifest_path = Path(out_manifest)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def _compute_metrics(records) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {"records": 0}

    t0 = records[0].timestamp
    t1 = records[-1].timestamp
    duration_min = ((t1 - t0).total_seconds() + 1.0) / 60.0

    hr_vals = [r.heart_rate for r in records if r.heart_rate is not None]
    hr_points = len(hr_vals)
    missing_hr_ratio = 1.0 - (hr_points / n)

    unrealistic = [h for h in hr_vals if h < 40.0 or h > 220.0]
    unrealistic_hr_ratio = len(unrealistic) / max(1, hr_points)

    gps_points = sum(1 for r in records if r.raw.get("position_lat") is not None and r.raw.get("position_long") is not None)
    gps_ratio = gps_points / n

    dist_vals = [r.raw.get("distance") for r in records if isinstance(r.raw.get("distance"), (int, float))]
    distance_km = (max(dist_vals) / 1000.0) if dist_vals else 0.0

    power_vals = [r.power for r in records if r.power is not None]
    zero_power_ratio = 0.0
    if power_vals:
        zero_power_ratio = sum(1 for p in power_vals if p == 0) / len(power_vals)

    jumps = 0
    jump_den = 0
    for i in range(1, n):
        a = records[i - 1].heart_rate
        b = records[i].heart_rate
        if a is None or b is None:
            continue
        jump_den += 1
        if abs(b - a) >= 18:
            jumps += 1
    hr_jump_ratio = jumps / max(1, jump_den)

    return {
        "records": n,
        "duration_min": round(duration_min, 2),
        "distance_km": round(distance_km, 2),
        "gps_ratio": round(gps_ratio, 4),
        "hr_points_ratio": round(hr_points / n, 4),
        "missing_hr_ratio": round(missing_hr_ratio, 4),
        "unrealistic_hr_ratio": round(unrealistic_hr_ratio, 4),
        "zero_power_ratio": round(zero_power_ratio, 4),
        "hr_jump_ratio": round(hr_jump_ratio, 4),
    }


def _reject_reasons(metrics: dict[str, float | int], th: QAThresholds) -> list[str]:
    reasons: list[str] = []
    if int(metrics.get("records", 0)) < th.min_records:
        reasons.append("too_few_records")
    if float(metrics.get("duration_min", 0.0)) < th.min_duration_min:
        reasons.append("too_short_duration")
    if float(metrics.get("distance_km", 0.0)) < th.min_distance_km:
        reasons.append("too_short_distance")
    if float(metrics.get("gps_ratio", 0.0)) < th.min_gps_ratio:
        reasons.append("insufficient_gps_track")
    if float(metrics.get("hr_points_ratio", 0.0)) < th.min_hr_points_ratio:
        reasons.append("insufficient_hr_coverage")
    if float(metrics.get("missing_hr_ratio", 1.0)) > th.max_missing_hr_ratio:
        reasons.append("too_many_missing_hr")
    if float(metrics.get("unrealistic_hr_ratio", 1.0)) > th.max_unrealistic_hr_ratio:
        reasons.append("unrealistic_hr_values")
    if float(metrics.get("zero_power_ratio", 0.0)) > th.max_zero_power_ratio:
        reasons.append("too_much_zero_power")
    if float(metrics.get("hr_jump_ratio", 1.0)) > th.max_hr_jump_ratio:
        reasons.append("too_many_hr_jumps")
    return reasons
