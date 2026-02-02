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
    corpus = [f"{item.headline} {item.body}" for item in news]
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return TextFeatures(vectorizer=vectorizer)


def vectorize_news(features: TextFeatures, news: Sequence[NewsItem]) -> pd.DataFrame:
    texts = [f"{item.headline} {item.body}" for item in news]
    return features.transform(texts)
