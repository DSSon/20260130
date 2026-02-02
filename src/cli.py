"""CLI entry point for backtesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backtest import run_backtest
from src.backtest.run import BacktestConfig
from src.config import load_config


def _load_config(path: str) -> BacktestConfig:
    raw = load_config(path)

    data = raw.get("data", {})
    model = raw.get("model", {})
    backtest = raw.get("backtest", {})

    return BacktestConfig(
        data_path=data["path"],
        time_col=data.get("time_col", "date"),
        target_col=data["target_col"],
        feature_cols=data["feature_cols"],
        return_col=data.get("return_col"),
        task=raw.get("task", "regression"),
        model_name=model.get("name", "linear"),
        model_params=model.get("params", {}) or {},
        train_ratio=backtest.get("train_ratio", 0.7),
        top_n=backtest.get("top_n", 10),
        output_dir=backtest.get("output_dir", "outputs"),
        save_predictions=backtest.get("save_predictions", True),
    )


def _cmd_backtest(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    summary = run_backtest(config)

    print("Backtest summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    output_path = Path(config.output_dir) / "summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline backtesting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--config", required=True)
    backtest_parser.set_defaults(func=_cmd_backtest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
