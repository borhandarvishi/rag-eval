"""Retrieval metrics: Recall@K and MRR."""

from __future__ import annotations

from typing import Sequence, Set


def recall_at_k(
    retrieved: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = set(retrieved[:k])
    rel: Set[int] = set(relevant)
    return len(top & rel) / float(len(rel))


def reciprocal_rank(
    retrieved: Sequence[int],
    relevant: Sequence[int],
) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0
