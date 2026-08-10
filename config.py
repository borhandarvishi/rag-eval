"""Central configuration for the Milvus RAG evaluation project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"

PRODUCTS_CSV = DATA_DIR / "ecommerce_products.csv"
QA_CSV = DATA_DIR / "ecommerce_products_qa.csv"

# Load `.env` from project root (does not override already-exported env vars)
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Dense embeddings (from `.env`)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBEDDING_MODEL_PROVIDER = os.getenv(
    "EMBEDDING_MODEL_PROVIDER", "openai"
).strip().lower()
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "").strip()

# Backward-compatible aliases used by the CLI / engine
DENSE_MODEL_NAME = EMBEDDING_MODEL
# text-embedding-3-small default output size
DENSE_DIM = int(os.getenv("DENSE_DIM", "1536"))

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Embedding batch size — OpenAI allows large batches; keep modest for RAM
EMBED_BATCH_SIZE = 64
RERANK_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Milvus (prefer values from `.env`)
# ---------------------------------------------------------------------------
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost").strip()
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530").strip()
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "").strip()  # optional Zilliz / auth
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "ecom_rag_eval").strip()

# Insert in chunks to bound memory / RPC payload size
INSERT_BATCH_SIZE = 64

# Hybrid WeightedRanker weights: (dense, sparse)
WEIGHTED_DENSE = 0.7
WEIGHTED_SPARSE = 0.3

# Candidate pool before cross-encoder rerank
RERANK_CANDIDATES = 20
EVAL_TOP_K = 10

# Dense index configurations to benchmark
# name -> (index_type, index_params, search_params)
DENSE_INDEX_CONFIGS: Dict[str, Tuple[str, dict, dict]] = {
    "HNSW_ef64": (
        "HNSW",
        {"M": 16, "efConstruction": 200},
        {"ef": 64},
    ),
    "HNSW_ef128": (
        "HNSW",
        {"M": 16, "efConstruction": 200},
        {"ef": 128},
    ),
    "IVF_FLAT": (
        "IVF_FLAT",
        {"nlist": 128},
        {"nprobe": 16},
    ),
}

SPARSE_INDEX_PARAMS = {"drop_ratio_build": 0.2}
SPARSE_SEARCH_PARAMS = {"drop_ratio_search": 0.2}


@dataclass
class RunConfig:
    """Runtime knobs overridable via CLI."""

    host: str = MILVUS_HOST
    port: str = MILVUS_PORT
    token: str = MILVUS_TOKEN
    collection_prefix: str = COLLECTION_PREFIX
    embed_batch_size: int = EMBED_BATCH_SIZE
    insert_batch_size: int = INSERT_BATCH_SIZE
    top_k: int = EVAL_TOP_K
    rerank_candidates: int = RERANK_CANDIDATES
    weighted_dense: float = WEIGHTED_DENSE
    weighted_sparse: float = WEIGHTED_SPARSE
    index_names: List[str] = field(
        default_factory=lambda: list(DENSE_INDEX_CONFIGS.keys())
    )
    use_cache: bool = True
    drop_existing: bool = True
