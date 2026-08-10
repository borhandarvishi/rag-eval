"""Batched dense / sparse embedding + cross-encoder reranker."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


class EmbeddingEngine:
    """
    Dense (OpenAI or local ST), sparse (BM25), and cross-encoder reranker.

    Dense embeddings are cached on disk to avoid recompute across index runs.
    BM25 is fit on the corpus and the fitted function is pickled for query-time use.
    """

    def __init__(
        self,
        dense_model_name: str,
        reranker_model_name: str,
        cache_dir: Path,
        batch_size: int = 64,
        rerank_batch_size: int = 16,
        device: Optional[str] = None,
        *,
        provider: str = "openai",
        api_key: Optional[str] = None,
        dense_dim: int = 1536,
    ) -> None:
        self.dense_model_name = dense_model_name
        self.reranker_model_name = reranker_model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.rerank_batch_size = rerank_batch_size
        self.device = device or _pick_device()
        self.provider = provider.strip().lower()
        self.api_key = api_key or None
        self.dense_dim = dense_dim

        self._openai = None
        self._dense_local = None
        self._reranker = None
        self._bm25 = None
        # In-memory query cache: same question is embedded by multiple strategies
        self._query_dense_cache: dict[str, List[float]] = {}

    # ------------------------------------------------------------------ dense
    def _require_openai(self):
        if self._openai is None:
            if not self.api_key:
                raise RuntimeError(
                    "EMBEDDING_MODEL_API_KEY is required for OpenAI dense embeddings"
                )
            from openai import OpenAI

            # Longer timeout + SDK retries for flaky networks
            self._openai = OpenAI(
                api_key=self.api_key,
                timeout=60.0,
                max_retries=5,
            )
        return self._openai

    def _openai_embeddings_create(self, **kwargs):
        """Call embeddings API with extra backoff on timeout / rate limits."""
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        client = self._require_openai()
        last_err: Optional[Exception] = None
        for attempt in range(6):
            try:
                return client.embeddings.create(**kwargs)
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_err = exc
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
        assert last_err is not None
        raise last_err

    def _load_dense_local(self):
        if self._dense_local is None:
            from sentence_transformers import SentenceTransformer

            self._dense_local = SentenceTransformer(
                self.dense_model_name, device=self.device
            )
        return self._dense_local

    def _encode_dense_openai(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        vectors: List[List[float]] = []
        iterator: Iterable[int] = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="OpenAI dense encode", unit="batch")

        for start in iterator:
            chunk = list(texts[start : start + self.batch_size])
            # OpenAI rejects empty strings; keep index alignment with a space
            safe = [t if t and t.strip() else " " for t in chunk]
            kwargs = {
                "model": self.dense_model_name,
                "input": safe,
            }
            # text-embedding-3-* supports explicit dimensions
            if self.dense_model_name.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dense_dim
            response = self._openai_embeddings_create(**kwargs)
            # API may return out of order — sort by index
            ordered = sorted(response.data, key=lambda row: row.index)
            vectors.extend(row.embedding for row in ordered)

        arr = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(arr)

    def _encode_dense_local(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        model = self._load_dense_local()
        return model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_dense(
        self,
        texts: Sequence[str],
        *,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        cache_path = (
            self.cache_dir / f"dense_{cache_key}.npy" if cache_key else None
        )
        if use_cache and cache_path and cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts) and cached.shape[1] == self.dense_dim:
                return cached

        if self.provider == "openai":
            vectors = self._encode_dense_openai(texts, show_progress=show_progress)
        elif self.provider in {"local", "sentence-transformers", "st"}:
            vectors = self._encode_dense_local(texts, show_progress=show_progress)
        else:
            raise ValueError(
                f"Unsupported EMBEDDING_MODEL_PROVIDER: {self.provider!r} "
                "(expected 'openai' or 'local')"
            )

        if vectors.shape[1] != self.dense_dim:
            raise ValueError(
                f"Dense dim mismatch: got {vectors.shape[1]}, expected {self.dense_dim}"
            )

        if use_cache and cache_path is not None:
            np.save(cache_path, vectors)
        return vectors

    def encode_dense_query(self, query: str) -> List[float]:
        cached = self._query_dense_cache.get(query)
        if cached is not None:
            return cached
        vec = self.encode_dense(
            [query], cache_key=None, use_cache=False, show_progress=False
        )[0]
        out = vec.tolist()
        self._query_dense_cache[query] = out
        return out

    # ----------------------------------------------------------------- sparse
    def fit_bm25(self, corpus: Sequence[str], *, use_cache: bool = True) -> None:
        cache_path = self.cache_dir / "bm25_model.pkl"
        if use_cache and cache_path.exists():
            with cache_path.open("rb") as fh:
                self._bm25 = pickle.load(fh)
            return

        try:
            from pymilvus.model.sparse import BM25EmbeddingFunction
            from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pymilvus.model is missing. Install with: "
                'pip install "pymilvus.model"   # or: pip install "pymilvus[model]"'
            ) from exc

        analyzer = build_default_analyzer(language="en")
        bm25 = BM25EmbeddingFunction(analyzer)
        bm25.fit(list(corpus))
        self._bm25 = bm25

        if use_cache:
            with cache_path.open("wb") as fh:
                pickle.dump(bm25, fh)

    def _require_bm25(self):
        if self._bm25 is None:
            raise RuntimeError("BM25 not fitted. Call fit_bm25() first.")
        return self._bm25

    @staticmethod
    def _sparse_to_dict(sparse_row) -> dict:
        """Convert scipy / pymilvus sparse row to {index: weight} dict."""
        if isinstance(sparse_row, dict):
            return {int(k): float(v) for k, v in sparse_row.items()}

        # scipy.sparse vector / csr row
        if hasattr(sparse_row, "tocoo"):
            coo = sparse_row.tocoo()
            return {int(i): float(v) for i, v in zip(coo.col, coo.data)}

        if hasattr(sparse_row, "indices") and hasattr(sparse_row, "data"):
            return {
                int(i): float(v)
                for i, v in zip(sparse_row.indices, sparse_row.data)
            }

        raise TypeError(f"Unsupported sparse row type: {type(sparse_row)}")

    def encode_sparse_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> List[dict]:
        bm25 = self._require_bm25()
        out: List[dict] = []
        iterator: Iterable[int] = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Sparse encode (docs)", unit="batch")
        for start in iterator:
            chunk = list(texts[start : start + batch_size])
            encoded = bm25.encode_documents(chunk)
            # encode_documents may return a sparse matrix
            if hasattr(encoded, "getrow"):
                for i in range(encoded.shape[0]):
                    out.append(self._sparse_to_dict(encoded.getrow(i)))
            else:
                for row in encoded:
                    out.append(self._sparse_to_dict(row))
        return out

    def encode_sparse_query(self, query: str) -> dict:
        bm25 = self._require_bm25()
        encoded = bm25.encode_queries([query])
        if hasattr(encoded, "getrow"):
            return self._sparse_to_dict(encoded.getrow(0))
        return self._sparse_to_dict(encoded[0])

    # -------------------------------------------------------------- reranker
    def _load_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "sentence_transformers is missing. Install with: "
                    "pip install sentence-transformers"
                ) from exc

            self._reranker = CrossEncoder(
                self.reranker_model_name, device=self.device
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        doc_ids: Sequence[int],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Return (doc_id, score) sorted descending, truncated to top_k."""
        if not documents:
            return []
        model = self._load_reranker()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(doc_ids, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [(int(i), float(s)) for i, s in ranked[:top_k]]

    def unload(self) -> None:
        """Free GPU/CPU model memory between stages if needed."""
        self._dense_local = None
        self._reranker = None
        self._openai = None
        self._query_dense_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
