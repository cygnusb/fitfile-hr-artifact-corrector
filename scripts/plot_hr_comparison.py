#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from hf_corrector.io_fit import load_fit_records


def _load_points(correction_json: Path):
    payload = json.loads(correction_json.read_text(encoding="utf-8"))
    points = payload["points"]
    ts = []
    orig = []
    corr = []
    flags = []
    for row in points:
        ts.append(datetime.fromisoformat(row["timestamp"]))
        orig.append(row.get("original_hr"))
        corr.append(row.get("corrected_hr"))
        flags.append(bool(row.get("artifact_flag")))
    return ts, orig, corr, flags


def _load_power_by_timestamp(fit_path: Path) -> dict[str, float]:
    recs = load_fit_records(fit_path)
    out: dict[str, float] = {}
    for r in recs:
        if r.power is None:
            continue
        out[r.timestamp.isoformat()] = float(r.power)
    return out


def _line_points(x, y, x0, y0, w, h, x_min, x_max, y_min, y_max):
    pts = []
    dx = max(1e-9, x_max - x_min)
    dy = max(1e-9, y_max - y_min)
    for xv, yv in zip(x, y):
        if yv is None:
            continue
        px = x0 + int((xv - x_min) / dx * w)
        py = y0 + h - int((yv - y_min) / dy * h)
        pts.append((px, py))
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot original vs corrected HR")
    parser.add_argument("--correction-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fit", help="Optional FIT file to overlay power")
    parser.add_argument("--window-minutes", type=float, default=0.0, help="0 means full activity")
    parser.add_argument("--window-center", choices=["middle"], default="middle")
    parser.add_argument("--title", default="Heart Rate: Original vs Corrected")
    args = parser.parse_args()

    correction_json = Path(args.correction_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts, orig, corr, flags = _load_points(correction_json)
    if not ts:
        raise SystemExit("No points found in correction JSON")

    power = None
    if args.fit:
        power_map = _load_power_by_timestamp(Path(args.fit))
        payload = json.loads(correction_json.read_text(encoding="utf-8"))
        power = [power_map.get(row["timestamp"]) for row in payload["points"]]

    if args.window_minutes > 0 and len(ts) > 2:
        if args.window_center == "middle":
            mid_t = ts[len(ts) // 2]
            half_s = args.window_minutes * 60.0 / 2.0
            idx = [i for i, t in enumerate(ts) if abs((t - mid_t).total_seconds()) <= half_s]
            if idx:
                i0, i1 = idx[0], idx[-1] + 1
                ts = ts[i0:i1]
                orig = orig[i0:i1]
                corr = corr[i0:i1]
                flags = flags[i0:i1]
                if power is not None:
                    power = power[i0:i1]

    t0 = ts[0]
    x = [(t - t0).total_seconds() / 60.0 for t in ts]
    y_vals = [v for v in orig if v is not None] + [v for v in corr if v is not None]
    if not y_vals:
        raise SystemExit("No HR values found in correction JSON")

    x_min, x_max = 0.0, max(x)
    y_min = max(40.0, min(y_vals) - 5.0)
    y_max = min(220.0, max(y_vals) + 5.0)

    width, height = 1800, 900
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # Grid + axis tick labels
    tick_color = (90, 90, 90)
    for i in range(0, 11):
        y = margin_t + int(i * plot_h / 10)
        d.line([(margin_l, y), (margin_l + plot_w, y)], fill=(230, 230, 230), width=1)
        y_val = y_max - (y_max - y_min) * (i / 10.0)
        d.text((10, y - 7), f"{y_val:.0f}", fill=tick_color)
    for i in range(0, 11):
        xx = margin_l + int(i * plot_w / 10)
        d.line([(xx, margin_t), (xx, margin_t + plot_h)], fill=(235, 235, 235), width=1)
        x_val = x_min + (x_max - x_min) * (i / 10.0)
        d.text((xx - 10, margin_t + plot_h + 8), f"{x_val:.0f}", fill=tick_color)

    # Axis
    d.rectangle([(margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h)], outline=(80, 80, 80), width=2)

    # Shade corrected regions along x-axis.
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            x0 = x[i]
            x1 = x[j - 1] if j - 1 < len(x) else x[i]
            px0 = margin_l + int((x0 - x_min) / max(1e-9, x_max - x_min) * plot_w)
            px1 = margin_l + int((x1 - x_min) / max(1e-9, x_max - x_min) * plot_w)
            if px1 <= px0:
                px1 = px0 + 1
            d.rectangle([(px0, margin_t), (px1, margin_t + plot_h)], fill=(250, 242, 220))
            i = j
        i += 1

    orig_pts = _line_points(x, orig, margin_l, margin_t, plot_w, plot_h, x_min, x_max, y_min, y_max)
    corr_pts = _line_points(x, corr, margin_l, margin_t, plot_w, plot_h, x_min, x_max, y_min, y_max)

    if len(orig_pts) > 1:
        d.line(orig_pts, fill=(214, 39, 40), width=2)  # red
    if len(corr_pts) > 1:
        d.line(corr_pts, fill=(31, 119, 180), width=2)  # blue

    if power is not None:
        p_vals = [v for v in power if isinstance(v, (int, float))]
        if p_vals:
            p_min = 0.0
            p_max = max(250.0, max(p_vals) * 1.05)
            p_pts = _line_points(x, power, margin_l, margin_t, plot_w, plot_h, x_min, x_max, p_min, p_max)
            if len(p_pts) > 1:
                d.line(p_pts, fill=(44, 160, 44), width=2)  # green
            d.text((width - 520, 18), f"Power axis: 0-{p_max:.0f} W", fill=(20, 20, 20))
            for i in range(0, 11):
                y = margin_t + int(i * plot_h / 10)
                p_val = p_max - (p_max - p_min) * (i / 10.0)
                d.text((margin_l + plot_w + 8, y - 7), f"{p_val:.0f}", fill=(44, 120, 44))

    # Legend and simple labels (no custom font to keep runtime robust)
    d.text((margin_l, 15), args.title, fill=(20, 20, 20))
    d.line([(width - 340, 25), (width - 300, 25)], fill=(214, 39, 40), width=3)
    d.text((width - 295, 18), "Original HR", fill=(20, 20, 20))
    d.line([(width - 200, 25), (width - 160, 25)], fill=(31, 119, 180), width=3)
    d.text((width - 155, 18), "Corrected HR", fill=(20, 20, 20))
    d.rectangle([(width - 790, 18), (width - 750, 31)], fill=(250, 242, 220), outline=(160, 160, 160))
    d.text((width - 745, 18), "Corrected region", fill=(20, 20, 20))
    if power is not None:
        d.line([(width - 640, 25), (width - 600, 25)], fill=(44, 160, 44), width=3)
        d.text((width - 595, 18), "Power", fill=(20, 20, 20))
    d.text((margin_l, height - 30), f"Duration: {x_max:.1f} min", fill=(40, 40, 40))
    d.text((margin_l + 220, height - 30), f"Y range: {y_min:.0f}-{y_max:.0f} bpm", fill=(40, 40, 40))
    d.text((margin_l + plot_w // 2 - 50, height - 50), "Time (minutes)", fill=(30, 30, 30))
    d.text((20, margin_t - 30), "HR (bpm)", fill=(30, 30, 30))
    if power is not None:
        d.text((margin_l + plot_w + 8, margin_t - 30), "Power (W)", fill=(44, 120, 44))

    img.save(out_path, format="PNG")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
