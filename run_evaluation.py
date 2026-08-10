#!/usr/bin/env python3
"""
Milvus retrieval strategy evaluation entrypoint.

Milvus host/port are read from `.env` (MILVUS_HOST / MILVUS_PORT).

Usage:
  cp .env.example .env   # set MILVUS_HOST / MILVUS_PORT
  python run_evaluation.py
  python run_evaluation.py --indexes HNSW_ef64 IVF_FLAT --no-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CACHE_DIR,
    COLLECTION_PREFIX,
    DENSE_DIM,
    DENSE_INDEX_CONFIGS,
    DENSE_MODEL_NAME,
    EMBEDDING_MODEL_API_KEY,
    EMBEDDING_MODEL_PROVIDER,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_TOKEN,
    PRODUCTS_CSV,
    QA_CSV,
    RERANKER_MODEL_NAME,
    RESULTS_DIR,
    RERANK_BATCH_SIZE,
    RunConfig,
)
from src.data_processor import DataProcessor
from src.embeddings import EmbeddingEngine
from src.evaluator import Evaluator
from src.milvus_manager import MilvusManager
from src.search_strategies import SearchStrategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Milvus dense / sparse / hybrid / rerank retrieval."
    )
    parser.add_argument(
        "--host",
        default=MILVUS_HOST,
        help=f"Milvus host (default from .env: {MILVUS_HOST})",
    )
    parser.add_argument(
        "--port",
        default=MILVUS_PORT,
        help=f"Milvus port (default from .env: {MILVUS_PORT})",
    )
    parser.add_argument(
        "--token",
        default=MILVUS_TOKEN,
        help="Optional Milvus / Zilliz auth token (MILVUS_TOKEN)",
    )
    parser.add_argument(
        "--collection-prefix",
        default=COLLECTION_PREFIX,
        help="Prefix for temporary eval collections",
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=list(DENSE_INDEX_CONFIGS.keys()),
        default=list(DENSE_INDEX_CONFIGS.keys()),
        help="Dense index configs to benchmark",
    )
    parser.add_argument("--top-k", type=int, default=RunConfig.top_k)
    parser.add_argument(
        "--rerank-candidates", type=int, default=RunConfig.rerank_candidates
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=RunConfig.embed_batch_size
    )
    parser.add_argument(
        "--insert-batch-size", type=int, default=RunConfig.insert_batch_size
    )
    parser.add_argument(
        "--weighted-dense", type=float, default=RunConfig.weighted_dense
    )
    parser.add_argument(
        "--weighted-sparse", type=float, default=RunConfig.weighted_sparse
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute embeddings / BM25 instead of loading cache",
    )
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="Do not drop Milvus collections after each index run",
    )
    return parser.parse_args()


def build_and_ingest(
    milvus: MilvusManager,
    collection_name: str,
    index_name: str,
    products,
    dense_vectors,
    sparse_vectors,
    cfg: RunConfig,
):
    index_type, index_params, search_params = DENSE_INDEX_CONFIGS[index_name]
    print(f"\n=== Building collection '{collection_name}' [{index_name}] ===")
    collection = milvus.create_collection(
        collection_name,
        dense_index_type=index_type,
        dense_index_params=index_params,
        drop_existing=cfg.drop_existing,
    )
    milvus.insert_batch(
        collection,
        ids=[p.id for p in products],
        texts=[p.text for p in products],
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        batch_size=cfg.insert_batch_size,
    )
    milvus.load(collection)
    print(
        f"Loaded {collection.num_entities} entities | dense index={index_type} "
        f"| search_params={search_params}"
    )
    return collection, search_params


def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        host=args.host,
        port=args.port,
        token=args.token,
        collection_prefix=args.collection_prefix,
        embed_batch_size=args.embed_batch_size,
        insert_batch_size=args.insert_batch_size,
        top_k=args.top_k,
        rerank_candidates=args.rerank_candidates,
        weighted_dense=args.weighted_dense,
        weighted_sparse=args.weighted_sparse,
        index_names=list(args.indexes),
        use_cache=not args.no_cache,
        drop_existing=True,
    )

    print("Loading datasets...")
    processor = DataProcessor(PRODUCTS_CSV, QA_CSV)
    products = processor.load_products()
    qa_records = processor.load_qa()
    corpus = DataProcessor.corpus_texts(products)
    print(f"Products: {len(products)} | QA pairs: {len(qa_records)}")

    print(
        f"\nInitializing embedding engine "
        f"(provider={EMBEDDING_MODEL_PROVIDER}, model={DENSE_MODEL_NAME}, dim={DENSE_DIM})..."
    )
    engine = EmbeddingEngine(
        dense_model_name=DENSE_MODEL_NAME,
        reranker_model_name=RERANKER_MODEL_NAME,
        cache_dir=CACHE_DIR,
        batch_size=cfg.embed_batch_size,
        rerank_batch_size=RERANK_BATCH_SIZE,
        provider=EMBEDDING_MODEL_PROVIDER,
        api_key=EMBEDDING_MODEL_API_KEY,
        dense_dim=DENSE_DIM,
    )

    cache_key = f"products_{DENSE_MODEL_NAME.replace('/', '_')}_{DENSE_DIM}"
    print(f"Encoding dense vectors with {DENSE_MODEL_NAME}...")
    dense_vectors = engine.encode_dense(
        corpus,
        cache_key=cache_key,
        use_cache=cfg.use_cache,
    )

    # Batch-embed all unique questions once (cuts OpenAI round-trips during eval)
    unique_questions = list(dict.fromkeys(q.question for q in qa_records))
    print(f"Pre-encoding {len(unique_questions)} unique questions...")
    q_cache_key = f"questions_{DENSE_MODEL_NAME.replace('/', '_')}_{DENSE_DIM}"
    q_vectors = engine.encode_dense(
        unique_questions,
        cache_key=q_cache_key,
        use_cache=cfg.use_cache,
    )
    for question, vec in zip(unique_questions, q_vectors, strict=True):
        engine._query_dense_cache[question] = vec.tolist()

    print("Fitting BM25 + encoding sparse vectors...")
    engine.fit_bm25(corpus, use_cache=cfg.use_cache)
    sparse_vectors = engine.encode_sparse_documents(corpus)

    milvus = MilvusManager(host=cfg.host, port=cfg.port, token=cfg.token or None)
    print(f"Connecting to Milvus at {cfg.host}:{cfg.port}...")
    milvus.connect()

    evaluator = Evaluator(top_k=cfg.top_k)
    all_rows = []

    try:
        for index_name in cfg.index_names:
            collection_name = f"{cfg.collection_prefix}_{index_name.lower()}"
            collection, search_params = build_and_ingest(
                milvus,
                collection_name,
                index_name,
                products,
                dense_vectors,
                sparse_vectors,
                cfg,
            )

            strategies = SearchStrategies(
                collection=collection,
                engine=engine,
                dense_search_params=search_params,
                weighted_dense=cfg.weighted_dense,
                weighted_sparse=cfg.weighted_sparse,
                rerank_candidates=cfg.rerank_candidates,
            )

            # Warmup (exclude from metrics)
            _ = strategies.pure_dense(qa_records[0].question, cfg.top_k)

            rows = evaluator.evaluate_index(strategies, qa_records, index_name)
            all_rows.extend(rows)

            if not args.keep_collections:
                milvus.release_and_drop(collection_name)
    finally:
        milvus.disconnect()
        engine.unload()

    report_df = Evaluator.to_dataframe(all_rows)
    md_path = Evaluator.save_report(report_df, RESULTS_DIR)

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    try:
        print(report_df.to_markdown(index=False))
    except ImportError:
        print(report_df.to_string(index=False))
    print(f"\nSaved: {md_path}")
    print(f"Saved: {RESULTS_DIR / 'evaluation_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
