from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    news_csv_path: Path
    price_csv_path: Path
    model_path: Path
    report_dir: Path
    alert_threshold: float


def load_settings(env_path: Optional[Path] = None) -> Settings:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    base_dir = Path(".")
    news_csv = Path(_get_env("NEWS_CSV", "data/news.csv"))
    price_csv = Path(_get_env("PRICE_CSV", "data/prices.csv"))
    model_path = Path(_get_env("MODEL_PATH", "artifacts/model.pkl"))
    report_dir = Path(_get_env("REPORT_DIR", "reports"))
    alert_threshold = float(_get_env("ALERT_THRESHOLD", "0.65"))

    return Settings(
        news_csv_path=base_dir / news_csv,
        price_csv_path=base_dir / price_csv,
        model_path=base_dir / model_path,
        report_dir=base_dir / report_dir,
        alert_threshold=alert_threshold,
    )


def _get_env(key: str, default: str) -> str:
    import os

    return os.getenv(key, default)
