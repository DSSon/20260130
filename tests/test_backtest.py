import json
from pathlib import Path

from src.backtest.run import BacktestConfig, run_backtest
from src.config import load_config


def test_run_backtest_creates_outputs(tmp_path: Path) -> None:
    raw = load_config("configs/example.yaml")
    raw["backtest"]["output_dir"] = str(tmp_path)

    config = BacktestConfig(
        data_path=raw["data"]["path"],
        time_col=raw["data"]["time_col"],
        target_col=raw["data"]["target_col"],
        feature_cols=raw["data"]["feature_cols"],
        return_col=raw["data"].get("return_col"),
        task=raw["task"],
        model_name=raw["model"]["name"],
        model_params=raw["model"].get("params", {}),
        train_ratio=raw["backtest"]["train_ratio"],
        top_n=raw["backtest"]["top_n"],
        output_dir=raw["backtest"]["output_dir"],
        save_predictions=raw["backtest"]["save_predictions"],
    )

    summary = run_backtest(config)

    metrics_path = tmp_path / "metrics.json"
    preds_path = tmp_path / "predictions.csv"

    assert metrics_path.exists()
    assert preds_path.exists()
    assert summary["task"] == "regression"

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "mae" in metrics
    assert "rmse" in metrics
