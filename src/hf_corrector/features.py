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
    "power_roll_60",
    "cadence_roll_60",
    "speed_roll_60",
    "power_roll_120",
    "speed_roll_120",
    "d_power",
    "d_speed",
    "d_altitude",
]


def build_feature_matrix(records: list[FitRecord]) -> np.ndarray:
    power_win_30: deque[float] = deque(maxlen=30)
    cadence_win_30: deque[float] = deque(maxlen=30)
    speed_win_30: deque[float] = deque(maxlen=30)
    power_win_60: deque[float] = deque(maxlen=60)
    cadence_win_60: deque[float] = deque(maxlen=60)
    speed_win_60: deque[float] = deque(maxlen=60)
    power_win_120: deque[float] = deque(maxlen=120)
    speed_win_120: deque[float] = deque(maxlen=120)
    rows: list[list[float]] = []

    prev_p = 0.0
    prev_s = 0.0
    prev_a = 0.0

    for i, r in enumerate(records):
        p = r.power if r.power is not None else 0.0
        c = r.cadence if r.cadence is not None else 0.0
        s = r.speed if r.speed is not None else 0.0
        a = r.altitude if r.altitude is not None else 0.0
        g = r.grade if r.grade is not None else 0.0

        power_win_30.append(p)
        cadence_win_30.append(c)
        speed_win_30.append(s)
        power_win_60.append(p)
        cadence_win_60.append(c)
        speed_win_60.append(s)
        power_win_120.append(p)
        speed_win_120.append(s)

        d_power = p - prev_p if i > 0 else 0.0
        d_speed = s - prev_s if i > 0 else 0.0
        d_altitude = a - prev_a if i > 0 else 0.0

        rows.append(
            [
                p,
                c,
                s,
                a,
                g,
                float(np.mean(power_win_30)),
                float(np.mean(cadence_win_30)),
                float(np.mean(speed_win_30)),
                float(np.mean(power_win_60)),
                float(np.mean(cadence_win_60)),
                float(np.mean(speed_win_60)),
                float(np.mean(power_win_120)),
                float(np.mean(speed_win_120)),
                d_power,
                d_speed,
                d_altitude,
            ]
        )

        prev_p = p
        prev_s = s
        prev_a = a

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float)


def extract_target_hr(records: list[FitRecord]) -> np.ndarray:
    vals = [r.heart_rate if r.heart_rate is not None else np.nan for r in records]
    return np.asarray(vals, dtype=float)
