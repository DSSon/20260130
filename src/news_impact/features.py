from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from news_impact.data_sources.base import NewsItem


@dataclass
class TextFeatures:
    vectorizer: TfidfVectorizer

    def transform(self, texts: Sequence[str]) -> pd.DataFrame:
        matrix = self.vectorizer.transform(texts)
        return pd.DataFrame(matrix.toarray(), columns=self.vectorizer.get_feature_names_out())


def build_vectorizer(news: Sequence[NewsItem]) -> TextFeatures:
    corpus = _build_corpus(news)
    if not corpus:
        raise ValueError("Cannot build vectorizer with no news items.")
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return TextFeatures(vectorizer=vectorizer)


def vectorize_news(features: TextFeatures, news: Sequence[NewsItem]) -> pd.DataFrame:
    texts = _build_corpus(news)
    if not texts:
        return pd.DataFrame()
    return features.transform(texts)


def _build_corpus(news: Sequence[NewsItem]) -> list[str]:
    corpus: list[str] = []
    for item in news:
        headline = item.headline or ""
        body = item.body or ""
        corpus.append(f"{headline} {body}".strip())
    return corpus
