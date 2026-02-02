"""Run time-based backtests and output metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models import build_model
from src.models.baseline import ModelConfig


@dataclass
class BacktestConfig:
    data_path: str
    time_col: str
    target_col: str
    feature_cols: List[str]
    return_col: str | None
    task: str
    model_name: str
    model_params: Dict[str, Any]
    train_ratio: float
    top_n: int
    output_dir: str
    save_predictions: bool


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_rows(
    rows: List[Dict[str, Any]],
    time_col: str,
    feature_cols: List[str],
    target_col: str,
    return_col: str | None,
) -> Tuple[List[datetime], List[List[float]], List[float], List[float] | None]:
    times: List[datetime] = []
    features: List[List[float]] = []
    targets: List[float] = []
    returns: List[float] = []

    for row in rows:
        times.append(datetime.fromisoformat(row[time_col]))
        features.append([float(row[col]) for col in feature_cols])
        targets.append(float(row[target_col]))
        if return_col:
            returns.append(float(row[return_col]))

    return (
        times,
        features,
        targets,
        returns if return_col else None,
    )


def _split_by_time(
    times: List[datetime],
    features: List[List[float]],
    targets: List[float],
    returns: List[float] | None,
    train_ratio: float,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float], List[float] | None, List[float] | None]:
    order = sorted(range(len(times)), key=lambda idx: times[idx])
    features = [features[idx] for idx in order]
    targets = [targets[idx] for idx in order]
    returns = [returns[idx] for idx in order] if returns is not None else None

    split_idx = max(1, int(len(times) * train_ratio))
    x_train, x_test = features[:split_idx], features[split_idx:]
    y_train, y_test = targets[:split_idx], targets[split_idx:]
    r_train = returns[:split_idx] if returns is not None else None
    r_test = returns[split_idx:] if returns is not None else None

    return x_train, x_test, y_train, y_test, r_train, r_test


def _top_n_return(
    returns: List[float] | None,
    scores: List[float],
    top_n: int,
) -> float | None:
    if returns is None or not returns:
        return None
    top_n = min(top_n, len(returns))
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])[-top_n:]
    return sum(returns[idx] for idx in order) / top_n


def _regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    errors = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))
    return {"mae": mae, "rmse": rmse}


def _auc_score(y_true: List[int], y_score: List[float]) -> float:
    paired = sorted(zip(y_score, y_true))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0.0
    for idx, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += idx
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _f1_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _classification_metrics(y_true: List[int], y_score: List[float]) -> Dict[str, float]:
    auc_score = _auc_score(y_true, y_score)
    preds = [1 if score >= 0.5 else 0 for score in y_score]
    f1 = _f1_score(y_true, preds)
    return {"auc": auc_score, "f1": f1}


def _roc_curve(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float]]:
    thresholds = sorted(set(y_score), reverse=True)
    tpr_list = []
    fpr_list = []
    positives = sum(y_true)
    negatives = len(y_true) - positives

    for threshold in thresholds:
        tp = sum(1 for yt, ys in zip(y_true, y_score) if ys >= threshold and yt == 1)
        fp = sum(1 for yt, ys in zip(y_true, y_score) if ys >= threshold and yt == 0)
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list


def _save_plot(
    output_dir: Path,
    task: str,
    y_true: List[float],
    y_score: List[float],
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_path = output_dir / "metrics.png"
    plt.figure(figsize=(6, 4))
    if task == "regression":
        plt.scatter(y_true, y_score, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
    else:
        fpr, tpr = _roc_curve([int(v) for v in y_true], y_score)
        roc_auc = _auc_score([int(v) for v in y_true], y_score)
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)


def run_backtest(config: BacktestConfig) -> Dict[str, Any]:
    rows = _load_rows(config.data_path)
    times, features, targets, returns = _parse_rows(
        rows,
        config.time_col,
        config.feature_cols,
        config.target_col,
        config.return_col,
    )

    x_train, x_test, y_train, y_test, _, r_test = _split_by_time(
        times,
        features,
        targets,
        returns,
        config.train_ratio,
    )

    model = build_model(config.task, ModelConfig(config.model_name, config.model_params))
    model.fit(x_train, y_train)

    if config.task == "classification":
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)
        else:
            scores = model.predict(x_test)
        metrics = _classification_metrics([int(v) for v in y_test], scores)
    else:
        scores = model.predict(x_test)
        metrics = _regression_metrics(y_test, scores)

    top_return = _top_n_return(r_test, scores, config.top_n)
    if top_return is not None:
        metrics["top_n_return"] = top_return

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if config.save_predictions:
        preds_path = output_dir / "predictions.csv"
        with preds_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = [config.time_col, "y_true", "y_pred"]
            if config.return_col:
                header.append("return")
            writer.writerow(header)
            for idx, value in enumerate(scores):
                row = [times[len(y_train) + idx].isoformat(), y_test[idx], value]
                if r_test is not None:
                    row.append(r_test[idx])
                writer.writerow(row)

    plot_path = _save_plot(output_dir, config.task, y_test, scores)
    if plot_path:
        metrics["plot_path"] = plot_path

    summary = {
        "task": config.task,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "metrics": metrics,
    }
    return summary
