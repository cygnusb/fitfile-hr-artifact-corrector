from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

from .corrector import CorrectionConfig, correct_records, summarize
from .export import export_outputs, load_correction_json, save_correction_json
from .io_fit import load_fit_records
from .model import HRModel, pick_compute_device
from .qa import analyze_chest_directory
from .training import train_from_combined_directories, train_from_manifest


def cmd_train(args: argparse.Namespace) -> int:
    result = train_from_manifest(
        args.manifest,
        args.out,
        device=args.device,
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_dynamics=args.lambda_dynamics,
        max_drop_bpm_per_step=args.max_drop_bpm_per_step,
        max_windows=args.max_windows,
        patience=args.patience,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_train_combined(args: argparse.Namespace) -> int:
    result = train_from_combined_directories(
        chest_dir=args.chest_dir,
        tours_dir=args.tours_dir,
        out_dir=args.out,
        device=args.device,
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_dynamics=args.lambda_dynamics,
        max_drop_bpm_per_step=args.max_drop_bpm_per_step,
        max_windows=args.max_windows,
        patience=args.patience,
        val_fraction=args.val_fraction,
        pair_match_max_seconds=args.pair_match_max_seconds,
        paired_weight=args.paired_weight,
        min_paired_points=args.min_paired_points,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    records = load_fit_records(args.fit)
    model = HRModel.load(args.model) if args.model else _identity_model(records)
    points = correct_records(
        records,
        model,
        config=CorrectionConfig(mode=args.mode, hr_bias=args.hr_bias),
    )
    print(json.dumps(summarize(records, points), indent=2))
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    records = load_fit_records(args.fit)
    model = HRModel.load(args.model)
    points = correct_records(
        records,
        model,
        config=CorrectionConfig(mode=args.mode, hr_bias=args.hr_bias),
    )
    summary = summarize(records, points)
    save_correction_json(args.out, points, summary)
    print(json.dumps({"out": args.out, "summary": summary}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    data = load_correction_json(args.correction)
    points = _points_from_json(data["points"])
    produced = export_outputs(args.fit, args.out_dir, points, args.formats)
    print(json.dumps(produced, indent=2))
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    checks = []
    required_modules = [
        "fitdecode",
        "numpy",
        "torch",
        "yaml",
        "mcp",
    ]

    for mod_name in required_modules:
        try:
            module = importlib.import_module(mod_name)
            version = getattr(module, "__version__", "unknown")
            checks.append({"name": mod_name, "ok": True, "version": str(version)})
        except Exception as exc:  # pragma: no cover - startup guard
            checks.append({"name": mod_name, "ok": False, "error": str(exc)})

    fit_probe = {"name": "fit_probe", "ok": True}
    if args.fit:
        try:
            records = load_fit_records(args.fit)
            fit_probe["records"] = len(records)
        except Exception as exc:  # pragma: no cover - startup guard
            fit_probe = {"name": "fit_probe", "ok": False, "error": str(exc)}
    checks.append(fit_probe)

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "compute_device": pick_compute_device(),
        "checks": checks,
    }
    ok = all(item.get("ok") for item in checks)
    report["status"] = "ok" if ok else "fail"
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def cmd_qa_chest(args: argparse.Namespace) -> int:
    report = analyze_chest_directory(
        chest_dir=args.chest_dir,
        out_report=args.out_report,
        out_manifest=args.out_manifest,
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hf-corrector")
    sp = p.add_subparsers(dest="cmd", required=True)

    train = sp.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    train.add_argument("--seq-len", type=int, default=120)
    train.add_argument("--stride", type=int, default=10)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--lambda-dynamics", type=float, default=0.35)
    train.add_argument("--max-drop-bpm-per-step", type=float, default=1.2)
    train.add_argument("--max-windows", type=int, default=40000)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--val-fraction", type=float, default=0.15)
    train.set_defaults(func=cmd_train)

    train_combined = sp.add_parser("train-combined")
    train_combined.add_argument("--chest-dir", required=True)
    train_combined.add_argument("--tours-dir", required=True)
    train_combined.add_argument("--out", required=True)
    train_combined.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    train_combined.add_argument("--seq-len", type=int, default=120)
    train_combined.add_argument("--stride", type=int, default=10)
    train_combined.add_argument("--epochs", type=int, default=12)
    train_combined.add_argument("--batch-size", type=int, default=64)
    train_combined.add_argument("--lr", type=float, default=1e-3)
    train_combined.add_argument("--lambda-dynamics", type=float, default=0.35)
    train_combined.add_argument("--max-drop-bpm-per-step", type=float, default=1.2)
    train_combined.add_argument("--max-windows", type=int, default=40000)
    train_combined.add_argument("--patience", type=int, default=5)
    train_combined.add_argument("--val-fraction", type=float, default=0.15)
    train_combined.add_argument("--pair-match-max-seconds", type=float, default=2.0)
    train_combined.add_argument("--paired-weight", type=int, default=3)
    train_combined.add_argument("--min-paired-points", type=int, default=600)
    train_combined.set_defaults(func=cmd_train_combined)

    analyze = sp.add_parser("analyze")
    analyze.add_argument("--fit", required=True)
    analyze.add_argument("--model")
    analyze.add_argument("--mode", choices=["safe", "balanced", "aggressive"], default="balanced")
    analyze.add_argument("--hr-bias", type=float, default=0.0)
    analyze.set_defaults(func=cmd_analyze)

    correct = sp.add_parser("correct")
    correct.add_argument("--fit", required=True)
    correct.add_argument("--model", required=True)
    correct.add_argument("--mode", choices=["safe", "balanced", "aggressive"], default="balanced")
    correct.add_argument("--hr-bias", type=float, default=0.0)
    correct.add_argument("--out", required=True)
    correct.set_defaults(func=cmd_correct)

    export = sp.add_parser("export")
    export.add_argument("--fit", required=True)
    export.add_argument("--correction", required=True)
    export.add_argument("--out-dir", required=True)
    export.add_argument("--formats", nargs="+", default=["csv"], choices=["csv", "fit"])
    export.set_defaults(func=cmd_export)

    self_check = sp.add_parser("self-check")
    self_check.add_argument("--fit", help="Optional FIT file path for read probe")
    self_check.set_defaults(func=cmd_self_check)

    qa_chest = sp.add_parser("qa-chest")
    qa_chest.add_argument("--chest-dir", required=True)
    qa_chest.add_argument("--out-report", required=True)
    qa_chest.add_argument("--out-manifest", required=True)
    qa_chest.set_defaults(func=cmd_qa_chest)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


class _IdentityModel:
    def __init__(self, default: float):
        self.default = default

    def predict(self, x):
        import numpy as np

        return np.full((len(x),), self.default, dtype=float)


def _identity_model(records):
    vals = [r.heart_rate for r in records if r.heart_rate is not None]
    mean_hr = sum(vals) / len(vals) if vals else 120.0
    return _IdentityModel(mean_hr)


def _points_from_json(points_json):
    from datetime import datetime

    from .types import CorrectionPoint

    out = []
    for row in points_json:
        out.append(
            CorrectionPoint(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                original_hr=row.get("original_hr"),
                corrected_hr=row.get("corrected_hr"),
                confidence=float(row["confidence"]),
                artifact_flag=bool(row["artifact_flag"]),
                artifact_probability=float(row["artifact_probability"]),
            )
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
