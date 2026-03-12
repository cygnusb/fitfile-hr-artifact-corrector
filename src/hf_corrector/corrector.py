from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import DetectorConfig, artifact_flags, artifact_probability
from .features import build_feature_matrix
from .model import HRModel
from .types import CorrectionPoint, FitRecord


@dataclass(slots=True)
class CorrectionConfig:
    threshold: float = 0.5
    mode: str = "balanced"
    hr_bias: float = 0.0
    use_mc_uncertainty: bool = True


def _threshold_for_mode(mode: str) -> float:
    if mode == "safe":
        return 0.7
    if mode == "aggressive":
        return 0.35
    return 0.5


def correct_records(
    records: list[FitRecord],
    model: HRModel,
    detector_cfg: DetectorConfig | None = None,
    config: CorrectionConfig | None = None,
) -> list[CorrectionPoint]:
    config = config or CorrectionConfig()
    config.threshold = _threshold_for_mode(config.mode)

    x = build_feature_matrix(records)

    # Use MC-Dropout uncertainty when available
    if config.use_mc_uncertainty and hasattr(model, "predict_with_uncertainty"):
        pred, uncertainty = model.predict_with_uncertainty(x)
    else:
        pred = model.predict(x)
        uncertainty = np.zeros(len(pred), dtype=float)

    probs = artifact_probability(records, detector_cfg)
    pred = _calibrate_predictions(records, pred, probs)
    flags = artifact_flags(probs, threshold=config.threshold)

    corrected_vals: list[float | None] = []
    out: list[CorrectionPoint] = []
    for i, r in enumerate(records):
        orig = r.heart_rate
        replaced = bool(flags[i] or orig is None)
        corrected = pred[i] if replaced else orig
        if replaced and corrected is not None and config.hr_bias != 0.0:
            corrected = float(corrected + config.hr_bias)

        # Confidence from MC-Dropout uncertainty (lower std = higher confidence)
        if uncertainty[i] > 0:
            mc_conf = float(np.clip(1.0 - uncertainty[i] / 10.0, 0.0, 1.0))
        else:
            mc_conf = 1.0 - abs(float(probs[i]) - config.threshold)
        conf = float(np.clip(mc_conf, 0.0, 1.0))

        if corrected is not None:
            corrected = float(np.clip(corrected, 45.0, 210.0))
        corrected_vals.append(corrected)
        out.append(
            CorrectionPoint(
                timestamp=r.timestamp,
                original_hr=orig,
                corrected_hr=corrected,
                confidence=conf,
                artifact_flag=bool(flags[i]),
                artifact_probability=float(probs[i]),
            )
        )
    _suppress_downward_spikes(records, out, corrected_vals)
    _apply_hr_dynamics_constraints(records, out, corrected_vals)
    return out


def _calibrate_predictions(records: list[FitRecord], pred: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Calibrate model output per ride using likely-valid optical segments."""
    if len(pred) == 0:
        return pred

    idx = []
    y_true = []
    y_pred = []
    for i, r in enumerate(records):
        hr = r.heart_rate
        p = r.power if r.power is not None else 0.0
        if hr is None:
            continue
        # Anchors: low artifact probability + some load + plausible HR band.
        if probs[i] <= 0.2 and p >= 80.0 and 60.0 <= hr <= 190.0:
            idx.append(i)
            y_true.append(float(hr))
            y_pred.append(float(pred[i]))

    if len(y_true) < 200:
        return pred

    xp = np.asarray(y_pred, dtype=float)
    yp = np.asarray(y_true, dtype=float)
    x_mean = xp.mean()
    y_mean = yp.mean()
    var = float(np.mean((xp - x_mean) ** 2))
    if var < 1e-6:
        return pred

    cov = float(np.mean((xp - x_mean) * (yp - y_mean)))
    slope = cov / var
    intercept = y_mean - slope * x_mean

    # Keep calibration conservative.
    slope = float(np.clip(slope, 0.85, 1.2))
    intercept = float(np.clip(intercept, -20.0, 20.0))
    return slope * pred + intercept


def _suppress_downward_spikes(
    records: list[FitRecord], points: list[CorrectionPoint], corrected_vals: list[float | None]
) -> None:
    """Remove short downward HR spikes that survive detection at segment borders."""
    n = len(points)
    if n < 3:
        return
    for i in range(1, n - 1):
        c = corrected_vals[i]
        p = records[i].power if records[i].power is not None else 0.0
        l = corrected_vals[i - 1]
        r = corrected_vals[i + 1]
        if not (isinstance(c, (int, float)) and isinstance(l, (int, float)) and isinstance(r, (int, float))):
            continue
        if p < 130.0:
            continue
        local_ref = max(l, r)
        # Typical artifact remnant: single-sample collapse to ~90 while neighbors stay high.
        if c <= 95.0 and (local_ref - c) >= 12.0:
            c_new = float(np.clip((l + r) / 2.0, 45.0, 210.0))
            corrected_vals[i] = c_new
            points[i].corrected_hr = c_new
            points[i].artifact_flag = True


def _apply_hr_dynamics_constraints(
    records: list[FitRecord], points: list[CorrectionPoint], corrected_vals: list[float | None]
) -> None:
    """Constrain instantaneous HR dynamics to physiologically plausible rates."""
    n = len(points)
    if n < 2:
        return
    for i in range(1, n):
        prev = corrected_vals[i - 1]
        curr = corrected_vals[i]
        if not (isinstance(prev, (int, float)) and isinstance(curr, (int, float))):
            continue
        dt = (records[i].timestamp - records[i - 1].timestamp).total_seconds()
        if dt <= 0:
            continue

        p = records[i].power if records[i].power is not None else 0.0
        # At low power/coasting HR should usually decay slowly.
        max_drop_per_s = 1.2 if p < 60.0 else 2.5
        max_rise_per_s = 4.0

        min_allowed = prev - max_drop_per_s * dt
        max_allowed = prev + max_rise_per_s * dt
        new_val = curr
        if curr < min_allowed:
            new_val = float(min_allowed)
        elif curr > max_allowed:
            new_val = float(max_allowed)

        if new_val != curr:
            new_val = float(np.clip(new_val, 45.0, 210.0))
            corrected_vals[i] = new_val
            points[i].corrected_hr = new_val
            points[i].artifact_flag = True


def summarize(records: list[FitRecord], points: list[CorrectionPoint]) -> dict[str, float | int]:
    hr_vals = [r.heart_rate for r in records if r.heart_rate is not None]
    corrected = [p.corrected_hr for p in points if p.corrected_hr is not None]
    flagged = sum(1 for p in points if p.artifact_flag)

    return {
        "records": len(records),
        "hr_points": len(hr_vals),
        "flagged_points": flagged,
        "flagged_ratio": round(flagged / len(points), 4) if points else 0.0,
        "hr_avg_raw": round(float(np.mean(hr_vals)), 2) if hr_vals else 0.0,
        "hr_avg_corrected": round(float(np.mean(corrected)), 2) if corrected else 0.0,
    }
