from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Protocol

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


class EmbeddingProvider(Protocol):
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: Iterable[str], max_features: int) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_features)
    return {token: idx for idx, (token, _) in enumerate(most_common)}


def tfidf_features(
    texts: List[str],
    vocab: Optional[Dict[str, int]] = None,
    max_features: int = 500,
) -> List[Dict[str, float]]:
    if vocab is None:
        vocab = build_vocab(texts, max_features)

    doc_freq = [0] * len(vocab)
    tokenized = []
    for text in texts:
        tokens = tokenize(text)
        tokenized.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in vocab:
                doc_freq[vocab[token]] += 1

    num_docs = max(len(texts), 1)
    idf = [math.log((1 + num_docs) / (1 + df)) + 1 for df in doc_freq]

    features: List[Dict[str, float]] = []
    for tokens in tokenized:
        counts: Counter[str] = Counter(token for token in tokens if token in vocab)
        doc_len = sum(counts.values()) or 1
        row: Dict[str, float] = {}
        for token, count in counts.items():
            idx = vocab[token]
            tf = count / doc_len
            row[f"tfidf_{token}"] = tf * idf[idx]
        features.append(row)

    return features


def embedding_features(
    texts: List[str], embedder: Optional[EmbeddingProvider] = None
) -> List[Dict[str, float]]:
    if embedder is None:
        return [dict() for _ in texts]

    embeddings = embedder.encode(texts)
    features = []
    for vector in embeddings:
        row = {f"emb_{idx}": float(value) for idx, value in enumerate(vector)}
        features.append(row)
    return features
