"""Six retrieval methodologies over a multi-vector Milvus collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker

from config import SPARSE_SEARCH_PARAMS
from src.embeddings import EmbeddingEngine
from src.milvus_manager import MilvusManager


@dataclass(frozen=True, slots=True)
class SearchResult:
    ids: List[int]
    latency_ms: float


class SearchStrategies:
    """
    Implements:
      1. Pure Dense
      2. Pure Sparse (BM25)
      3. Hybrid + RRFRanker
      4. Hybrid + WeightedRanker
      5. Dense → Cross-Encoder Rerank
      6. Hybrid → Cross-Encoder Rerank
    """

    def __init__(
        self,
        collection: Collection,
        engine: EmbeddingEngine,
        dense_search_params: dict,
        dense_metric: str = "IP",
        weighted_dense: float = 0.7,
        weighted_sparse: float = 0.3,
        rerank_candidates: int = 20,
    ) -> None:
        self.collection = collection
        self.engine = engine
        self.dense_search_params = dense_search_params
        self.dense_metric = dense_metric
        self.weighted_dense = weighted_dense
        self.weighted_sparse = weighted_sparse
        self.rerank_candidates = rerank_candidates

    def method_map(self) -> Dict[str, Callable[[str, int], SearchResult]]:
        return {
            "1_pure_dense": self.pure_dense,
            "2_pure_sparse": self.pure_sparse,
            "3_hybrid_rrf": self.hybrid_rrf,
            "4_hybrid_weighted": self.hybrid_weighted,
            "5_dense_rerank": self.dense_then_rerank,
            "6_hybrid_rerank": self.hybrid_then_rerank,
        }

    # ---------------------------------------------------------------- helpers
    def _time_call(self, fn: Callable[[], List[int]]) -> SearchResult:
        t0 = time.perf_counter()
        ids = fn()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SearchResult(ids=ids, latency_ms=latency_ms)

    def _dense_search(self, query: str, limit: int) -> List[int]:
        qvec = self.engine.encode_dense_query(query)
        hits = self.collection.search(
            data=[qvec],
            anns_field=MilvusManager.DENSE_FIELD,
            param={
                "metric_type": self.dense_metric,
                "params": self.dense_search_params,
            },
            limit=limit,
            output_fields=[],
        )
        return [int(h.id) for h in hits[0]]

    def _sparse_search(self, query: str, limit: int) -> List[int]:
        qvec = self.engine.encode_sparse_query(query)
        if not qvec:
            return []
        hits = self.collection.search(
            data=[qvec],
            anns_field=MilvusManager.SPARSE_FIELD,
            param={
                "metric_type": "IP",
                "params": SPARSE_SEARCH_PARAMS,
            },
            limit=limit,
            output_fields=[],
        )
        return [int(h.id) for h in hits[0]]

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        rerank,
    ) -> List[int]:
        dense_vec = self.engine.encode_dense_query(query)
        sparse_vec = self.engine.encode_sparse_query(query)

        # Empty BM25 query → fall back to dense-only (avoids Milvus sparse errors)
        if not sparse_vec:
            return self._dense_search(query, limit)

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field=MilvusManager.DENSE_FIELD,
            param={
                "metric_type": self.dense_metric,
                "params": self.dense_search_params,
            },
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field=MilvusManager.SPARSE_FIELD,
            param={
                "metric_type": "IP",
                "params": SPARSE_SEARCH_PARAMS,
            },
            limit=limit,
        )
        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=limit,
            output_fields=[],
        )
        return [int(h.id) for h in results[0]]

    def _rerank_ids(
        self, query: str, candidate_ids: Sequence[int], top_k: int
    ) -> List[int]:
        text_map = MilvusManager.fetch_texts(self.collection, candidate_ids)
        # Preserve candidate order for stable pairing
        docs: List[str] = []
        ids: List[int] = []
        for cid in candidate_ids:
            if cid in text_map:
                ids.append(cid)
                docs.append(text_map[cid])
        ranked = self.engine.rerank(query, docs, ids, top_k=top_k)
        return [i for i, _ in ranked]

    # ------------------------------------------------------------- strategies
    def pure_dense(self, query: str, top_k: int) -> SearchResult:
        return self._time_call(lambda: self._dense_search(query, top_k))

    def pure_sparse(self, query: str, top_k: int) -> SearchResult:
        return self._time_call(lambda: self._sparse_search(query, top_k))

    def hybrid_rrf(self, query: str, top_k: int) -> SearchResult:
        return self._time_call(
            lambda: self._hybrid_search(query, top_k, RRFRanker())
        )

    def hybrid_weighted(self, query: str, top_k: int) -> SearchResult:
        return self._time_call(
            lambda: self._hybrid_search(
                query,
                top_k,
                WeightedRanker(self.weighted_dense, self.weighted_sparse),
            )
        )

    def dense_then_rerank(self, query: str, top_k: int) -> SearchResult:
        def _run() -> List[int]:
            candidates = self._dense_search(
                query, max(self.rerank_candidates, top_k)
            )
            return self._rerank_ids(query, candidates, top_k)

        return self._time_call(_run)

    def hybrid_then_rerank(self, query: str, top_k: int) -> SearchResult:
        def _run() -> List[int]:
            candidates = self._hybrid_search(
                query,
                max(self.rerank_candidates, top_k),
                RRFRanker(),
            )
            return self._rerank_ids(query, candidates, top_k)

        return self._time_call(_run)
