from __future__ import annotations

from collections import deque

import numpy as np

from .types import FitRecord


FEATURE_NAMES = [
    "power",
    "cadence",
    "speed",
    "altitude",
    "grade",
    "power_roll_30",
    "cadence_roll_30",
    "speed_roll_30",
]


def build_feature_matrix(records: list[FitRecord]) -> np.ndarray:
    power_win: deque[float] = deque(maxlen=30)
    cadence_win: deque[float] = deque(maxlen=30)
    speed_win: deque[float] = deque(maxlen=30)
    rows: list[list[float]] = []

    for r in records:
        p = r.power if r.power is not None else 0.0
        c = r.cadence if r.cadence is not None else 0.0
        s = r.speed if r.speed is not None else 0.0
        a = r.altitude if r.altitude is not None else 0.0
        g = r.grade if r.grade is not None else 0.0

        power_win.append(p)
        cadence_win.append(c)
        speed_win.append(s)

        rows.append(
            [
                p,
                c,
                s,
                a,
                g,
                float(np.mean(power_win)),
                float(np.mean(cadence_win)),
                float(np.mean(speed_win)),
            ]
        )

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float)


def extract_target_hr(records: list[FitRecord]) -> np.ndarray:
    vals = [r.heart_rate if r.heart_rate is not None else np.nan for r in records]
    return np.asarray(vals, dtype=float)
