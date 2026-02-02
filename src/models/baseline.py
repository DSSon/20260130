"""Baseline model builders implemented without heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 500, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.coef_: List[float] = []
        self.intercept_: float = 0.0

    def fit(self, x: List[List[float]], y: List[float]) -> "LinearRegressionGD":
        if not x:
            raise ValueError("Empty training data")
        n_samples = len(x)
        n_features = len(x[0])
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for row, target in zip(x, y):
                pred = _dot(self.coef_, row) + self.intercept_
                error = pred - target
                for j in range(n_features):
                    grad_w[j] += error * row[j]
                grad_b += error
            for j in range(n_features):
                grad_w[j] = (grad_w[j] / n_samples) + self.l2 * self.coef_[j]
                self.coef_[j] -= self.learning_rate * grad_w[j]
            self.intercept_ -= self.learning_rate * grad_b / n_samples
        return self

    def predict(self, x: List[List[float]]) -> List[float]:
        return [_dot(self.coef_, row) + self.intercept_ for row in x]


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 500, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.coef_: List[float] = []
        self.intercept_: float = 0.0

    def fit(self, x: List[List[float]], y: List[int]) -> "LogisticRegressionGD":
        if not x:
            raise ValueError("Empty training data")
        n_samples = len(x)
        n_features = len(x[0])
        self.coef_ = [0.0] * n_features
        self.intercept_ = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for row, target in zip(x, y):
                linear = _dot(self.coef_, row) + self.intercept_
                pred = _sigmoid(linear)
                error = pred - target
                for j in range(n_features):
                    grad_w[j] += error * row[j]
                grad_b += error
            for j in range(n_features):
                grad_w[j] = (grad_w[j] / n_samples) + self.l2 * self.coef_[j]
                self.coef_[j] -= self.learning_rate * grad_w[j]
            self.intercept_ -= self.learning_rate * grad_b / n_samples
        return self

    def predict_proba(self, x: List[List[float]]) -> List[float]:
        return [_sigmoid(_dot(self.coef_, row) + self.intercept_) for row in x]

    def predict(self, x: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(x)]


class RegressionStump:
    def __init__(self, feature_index: int, threshold: float, left_value: float, right_value: float):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value

    def predict_row(self, row: List[float]) -> float:
        return self.left_value if row[self.feature_index] <= self.threshold else self.right_value


def _fit_regression_stump(
    x: List[List[float]],
    residuals: List[float],
) -> RegressionStump:
    n_features = len(x[0])
    best_error = float("inf")
    best_params: Tuple[int, float, float, float] | None = None

    for j in range(n_features):
        values = sorted(set(row[j] for row in x))
        for threshold in values:
            left = [residual for row, residual in zip(x, residuals) if row[j] <= threshold]
            right = [residual for row, residual in zip(x, residuals) if row[j] > threshold]
            if not left or not right:
                continue
            left_value = sum(left) / len(left)
            right_value = sum(right) / len(right)
            error = 0.0
            for row, residual in zip(x, residuals):
                pred = left_value if row[j] <= threshold else right_value
                error += (residual - pred) ** 2
            if error < best_error:
                best_error = error
                best_params = (j, threshold, left_value, right_value)

    if best_params is None:
        return RegressionStump(0, 0.0, 0.0, 0.0)
    return RegressionStump(*best_params)


class GradientBoostingRegressorSimple:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_value = 0.0
        self.stumps: List[RegressionStump] = []

    def fit(self, x: List[List[float]], y: List[float]) -> "GradientBoostingRegressorSimple":
        if not x:
            raise ValueError("Empty training data")
        self.base_value = sum(y) / len(y)
        current = [self.base_value] * len(y)
        self.stumps = []

        for _ in range(self.n_estimators):
            residuals = [target - pred for target, pred in zip(y, current)]
            stump = _fit_regression_stump(x, residuals)
            self.stumps.append(stump)
            for i, row in enumerate(x):
                current[i] += self.learning_rate * stump.predict_row(row)
        return self

    def predict(self, x: List[List[float]]) -> List[float]:
        preds = [self.base_value] * len(x)
        for stump in self.stumps:
            for i, row in enumerate(x):
                preds[i] += self.learning_rate * stump.predict_row(row)
        return preds


class GradientBoostingClassifierSimple:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_value = 0.0
        self.stumps: List[RegressionStump] = []

    def fit(self, x: List[List[float]], y: List[int]) -> "GradientBoostingClassifierSimple":
        if not x:
            raise ValueError("Empty training data")
        positive_rate = max(min(sum(y) / len(y), 0.99), 0.01)
        self.base_value = math.log(positive_rate / (1 - positive_rate))
        current = [self.base_value] * len(y)
        self.stumps = []

        for _ in range(self.n_estimators):
            probs = [_sigmoid(score) for score in current]
            residuals = [target - prob for target, prob in zip(y, probs)]
            stump = _fit_regression_stump(x, residuals)
            self.stumps.append(stump)
            for i, row in enumerate(x):
                current[i] += self.learning_rate * stump.predict_row(row)
        return self

    def predict_proba(self, x: List[List[float]]) -> List[float]:
        scores = [self.base_value] * len(x)
        for stump in self.stumps:
            for i, row in enumerate(x):
                scores[i] += self.learning_rate * stump.predict_row(row)
        return [_sigmoid(score) for score in scores]

    def predict(self, x: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(x)]


def build_model(task: str, config: ModelConfig):
    name = config.name.lower()
    params = config.params or {}

    if task == "regression":
        if name in {"linear", "ridge"}:
            return LinearRegressionGD(
                learning_rate=params.get("learning_rate", 0.05),
                epochs=params.get("epochs", 500),
                l2=params.get("l2", 0.0),
            )
        if name in {"gb", "gbr", "gradient_boosting"}:
            return GradientBoostingRegressorSimple(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            )
    elif task == "classification":
        if name in {"logistic", "logreg"}:
            return LogisticRegressionGD(
                learning_rate=params.get("learning_rate", 0.1),
                epochs=params.get("epochs", 500),
                l2=params.get("l2", 0.0),
            )
        if name in {"gb", "gbc", "gradient_boosting"}:
            return GradientBoostingClassifierSimple(
                n_estimators=params.get("n_estimators", 50),
                learning_rate=params.get("learning_rate", 0.1),
            )

    raise ValueError(f"Unsupported model name '{config.name}' for task '{task}'.")
