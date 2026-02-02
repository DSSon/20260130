from __future__ import annotations

import random
from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def train_val_test_split(
    items: Sequence[T],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[List[T], List[T], List[T]]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios must be positive")

    normalized = [ratio / total for ratio in ratios]
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    train_end = int(len(items) * normalized[0])
    val_end = train_end + int(len(items) * normalized[1])

    train = [items[i] for i in indices[:train_end]]
    val = [items[i] for i in indices[train_end:val_end]]
    test = [items[i] for i in indices[val_end:]]
    return train, val, test
