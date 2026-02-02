"""Command-line entrypoints for news_impact_alert."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news_impact_alert",
        description="News impact alert pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("prepare-data", help="Ingest and clean news/market data.")
    subparsers.add_parser("train", help="Train impact estimation models.")
    subparsers.add_parser("backtest", help="Run backtests and evaluation.")
    subparsers.add_parser("alert-demo", help="Run a demo alerting workflow.")

    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
