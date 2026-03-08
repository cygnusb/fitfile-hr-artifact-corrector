from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class FitRecord:
    timestamp: datetime
    heart_rate: float | None
    power: float | None
    cadence: float | None
    speed: float | None
    altitude: float | None
    grade: float | None
    raw: dict[str, Any]


@dataclass(slots=True)
class CorrectionPoint:
    timestamp: datetime
    original_hr: float | None
    corrected_hr: float | None
    confidence: float
    artifact_flag: bool
    artifact_probability: float
