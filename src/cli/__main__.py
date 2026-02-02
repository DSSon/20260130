"""Command-line entrypoints for news_impact_alert."""

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
    parser = argparse.ArgumentParser(
        prog="news_impact_alert",
        description="News impact alert pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("prepare-data", help="Ingest and clean news/market data.")
    subparsers.add_parser("train", help="Train impact estimation models.")

    backtest_parser = subparsers.add_parser("backtest", help="Run backtests and evaluation.")
    backtest_parser.add_argument("--config", required=True)
    backtest_parser.set_defaults(func=_cmd_backtest)

    subparsers.add_parser("alert-demo", help="Run a demo alerting workflow.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
