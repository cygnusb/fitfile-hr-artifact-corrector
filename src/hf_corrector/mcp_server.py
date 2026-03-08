from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corrector import CorrectionConfig, correct_records, summarize
from .export import export_outputs, save_correction_json
from .io_fit import load_fit_records
from .model import HRModel


def _register_tools(mcp, model_dir: str):
    @mcp.tool()
    def fit_analyze(file_path: str, mode: str = "balanced") -> dict:
        records = load_fit_records(file_path)
        model = HRModel.load(model_dir)
        points = correct_records(records, model, config=CorrectionConfig(mode=mode))
        return summarize(records, points)

    @mcp.tool()
    def fit_correct(file_path: str, out_json: str, mode: str = "balanced") -> dict:
        records = load_fit_records(file_path)
        model = HRModel.load(model_dir)
        points = correct_records(records, model, config=CorrectionConfig(mode=mode))
        summary = summarize(records, points)
        save_correction_json(out_json, points, summary)
        return {"out_json": out_json, "summary": summary}

    @mcp.tool()
    def fit_export(file_path: str, correction_json: str, out_dir: str, formats: list[str]) -> dict:
        from .cli import _points_from_json
        from .export import load_correction_json

        payload = load_correction_json(correction_json)
        points = _points_from_json(payload["points"])
        produced = export_outputs(file_path, out_dir, points, formats)
        return produced

    @mcp.tool()
    def fit_model_info() -> dict:
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            return {"model_dir": model_dir, "status": "missing_config"}
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return {"model_dir": model_dir, "config": cfg}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--name", default="fitfile-hr-artifact-corrector")
    args = parser.parse_args()

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP server requires optional dependency 'mcp'. "
            "Install with: pip install -e .[mcp]"
        ) from exc

    mcp = FastMCP(args.name)
    _register_tools(mcp, model_dir=args.model_dir)
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
