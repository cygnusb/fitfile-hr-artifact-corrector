#!/usr/bin/env python3
"""One-shot FIT correction: reads a FIT file, applies HR artifact correction,
writes the corrected HR values back into a new FIT file via fit-tool.

Usage:
    python scripts/correct_to_fit.py \
        --fit activity.fit \
        --model models/model_seq_v1 \
        --out activity_corrected.fit

Optional flags mirror the CLI `correct` subcommand:
    --mode   safe | balanced | aggressive  (default: balanced)
    --hr-bias FLOAT                        (default: 0.0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_corrector.corrector import CorrectionConfig, correct_records, summarize
from hf_corrector.io_fit import load_fit_records, rewrite_fit_heart_rate
from hf_corrector.model import HRModel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct HR artifacts in a FIT file and write a new FIT file."
    )
    parser.add_argument("--fit", required=True, help="Input FIT (or .fit.gz) file")
    parser.add_argument("--model", required=True, help="Trained model directory")
    parser.add_argument(
        "--out",
        help="Output FIT path (default: <input>_corrected.fit next to input)",
    )
    parser.add_argument(
        "--mode",
        choices=["safe", "balanced", "aggressive"],
        default="balanced",
    )
    parser.add_argument("--hr-bias", type=float, default=0.0)
    args = parser.parse_args()

    fit_path = Path(args.fit)
    if args.out:
        out_path = Path(args.out)
    else:
        stem = fit_path.name.removesuffix(".gz").removesuffix(".fit")
        out_path = fit_path.parent / f"{stem}_corrected.fit"

    print(f"Loading  {fit_path}")
    records = load_fit_records(fit_path)
    if not records:
        print("ERROR: no records found in FIT file", file=sys.stderr)
        return 1

    print(f"Loading model from {args.model}")
    model = HRModel.load(args.model)

    print(f"Correcting ({args.mode} mode, hr_bias={args.hr_bias:+.1f}) …")
    points = correct_records(
        records,
        model,
        config=CorrectionConfig(mode=args.mode, hr_bias=args.hr_bias),
    )

    summary = summarize(records, points)
    flagged = summary["flagged_points"]
    total = summary["records"]
    ratio = summary["flagged_ratio"] * 100
    print(
        f"  {flagged}/{total} points corrected ({ratio:.1f}%)  "
        f"avg HR: {summary['hr_avg_raw']:.0f} → {summary['hr_avg_corrected']:.0f} bpm"
    )

    corrections_by_ts = {
        p.timestamp.isoformat(): float(p.corrected_hr)
        for p in points
        if p.corrected_hr is not None
    }

    print(f"Writing  {out_path}")
    rewrite_fit_heart_rate(fit_path, out_path, corrections_by_ts)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
