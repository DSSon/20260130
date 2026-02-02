from __future__ import annotations

import argparse
from pathlib import Path

from news_impact.alerts import PrintAlertSink
from news_impact.config import load_settings
from news_impact.data_sources.news_csv import CsvNewsCollector
from news_impact.data_sources.price_csv import CsvPriceCollector
from news_impact.backtest import run_backtest
from news_impact.model import ImpactModel
from news_impact.realtime import monitor_news


def main() -> None:
    parser = argparse.ArgumentParser(description="News impact modeling CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("backtest", help="Run backtest and generate report")
    subparsers.add_parser("monitor", help="Score latest news and emit alerts")

    args = parser.parse_args()
    settings = load_settings()

    news_collector = CsvNewsCollector(settings.news_csv_path)
    price_collector = CsvPriceCollector(settings.price_csv_path)

    if args.command == "backtest":
        news_items = list(news_collector.fetch())
        price_bars = list(price_collector.fetch())
        result = run_backtest(news_items, price_bars, settings.report_dir)
        result.model.save(settings.model_path)
        print(f"Saved model to {settings.model_path}")
        print(f"Report saved to {settings.report_dir}")
    elif args.command == "monitor":
        if not settings.model_path.exists():
            raise SystemExit("Model not found. Run backtest first.")
        model = ImpactModel.load(settings.model_path)
        alerts = monitor_news(
            collector=news_collector,
            model=model,
            alert_threshold=settings.alert_threshold,
            sink=PrintAlertSink(),
        )
        print(f"Generated {len(alerts)} alerts")


if __name__ == "__main__":
    main()
