from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from news_impact.data_sources.base import NewsItem
from news_impact.features import TextFeatures, build_vectorizer, vectorize_news


@dataclass
class ImpactModel:
    classifier: LogisticRegression
    features: TextFeatures

    def predict_proba(self, news: Sequence[NewsItem]) -> np.ndarray:
        vectors = vectorize_news(self.features, news)
        return self.classifier.predict_proba(vectors)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ImpactModel":
        return joblib.load(path)


def train_model(news: Sequence[NewsItem], labels: Sequence[int]) -> Tuple[ImpactModel, dict[str, float]]:
    features = build_vectorizer(news)
    vectors = vectorize_news(features, news)
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(vectors, labels)

    preds = classifier.predict(vectors)
    probas = classifier.predict_proba(vectors)[:, 1]
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probas),
    }
    return ImpactModel(classifier=classifier, features=features), metrics


def build_labels(returns: pd.Series) -> Sequence[int]:
    return (returns > 0).astype(int).tolist()
