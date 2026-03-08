from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import FitRecord


@dataclass(slots=True)
class DetectorConfig:
    low_hr_threshold: float = 90.0
    low_hr_power_threshold: float = 130.0
    high_hr_threshold: float = 150.0
    high_hr_power_threshold: float = 120.0
    jump_bpm_threshold: float = 18.0
    jump_power_delta_max: float = 35.0
    max_drop_bpm_per_s: float = 3.0
    abrupt_drop_bpm: float = 10.0
    high_hr_low_power_score: float = 0.0


def artifact_probability(records: list[FitRecord], cfg: DetectorConfig | None = None) -> np.ndarray:
    cfg = cfg or DetectorConfig()
    probs = np.zeros(len(records), dtype=float)

    for i, r in enumerate(records):
        hr = r.heart_rate
        p = r.power if r.power is not None else 0.0
        if hr is None:
            probs[i] = 1.0
            continue

        score = 0.0
        if hr < cfg.low_hr_threshold and p >= cfg.low_hr_power_threshold:
            score += 0.65
        # High HR during low power (coasting/recovery) is often physiologically valid.
        if hr > cfg.high_hr_threshold and p < cfg.high_hr_power_threshold:
            score += cfg.high_hr_low_power_score
        if i > 0:
            prev = records[i - 1]
            if prev.heart_rate is not None:
                dhr = abs(hr - prev.heart_rate)
                pp = prev.power if prev.power is not None else 0.0
                dp = abs(p - pp)
                if dhr >= cfg.jump_bpm_threshold and dp <= cfg.jump_power_delta_max:
                    score += 0.35
                dt = (r.timestamp - prev.timestamp).total_seconds()
                if dt > 0:
                    drop_rate = (prev.heart_rate - hr) / dt
                    if drop_rate > cfg.max_drop_bpm_per_s and (prev.heart_rate - hr) >= cfg.abrupt_drop_bpm:
                        score += 0.7

        probs[i] = min(score, 1.0)

    return probs


def artifact_flags(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return probs >= threshold
