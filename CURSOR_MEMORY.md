# Cursor Memory — rag-eval

> **Cursor: Always read this file first before starting a new task to understand the context and past decisions.**

## Project Goal

Benchmark Milvus hybrid search vs pure dense / pure sparse / cross-encoder reranking on an e-commerce product QA dataset. Optimize for clean modular code, batched embedding/ingestion, and clear Recall@K / MRR / latency reporting.

## Architecture Decisions

| Concern | Choice | Notes |
|---------|--------|-------|
| Dense model | **`text-embedding-3-small` (OpenAI)** | dim=**1536**, L2-normalized, metric=`IP`; from `.env` (`EMBEDDING_MODEL`, `EMBEDDING_MODEL_PROVIDER=openai`, `EMBEDDING_MODEL_API_KEY`) |
| Sparse model | BM25 via `pymilvus.model.sparse.BM25EmbeddingFunction` | Stored as `SPARSE_FLOAT_VECTOR` |
| Reranker | `BAAI/bge-reranker-base` (`CrossEncoder`) | Applied on top-20 candidates by default |
| Vector DB | **Remote Milvus from `.env`** (`MILVUS_HOST` / `MILVUS_PORT`) | Verified server version **3.0.0** at `ailab.techstrata.com:19530`; local Docker Compose still ships **v2.4.15** as optional fallback |
| Client | `pymilvus` 3.x (ORM Collection API) | Works against Milvus 3.0; migrate to `MilvusClient` before pymilvus 3.1 |
| Dense indexes | `HNSW` (ef 64/128) + `IVF_FLAT` | Configured in `config.DENSE_INDEX_CONFIGS` |
| Sparse index | `SPARSE_INVERTED_INDEX` + IP | `drop_ratio_build/search=0.2` |
| Hybrid fusion | `RRFRanker` and `WeightedRanker(0.7, 0.3)` | Order: dense req, then sparse req; empty BM25 → dense fallback |
| Document text | `title \| category \| description` | Built vectorized in `DataProcessor` |
| Caching | `cache/dense_*.npy`, `cache/bm25_model.pkl` | Skip re-embed on re-runs; cache key includes model+dim |
| QA schema | cols: `question ids`, `question`, `id` | `id` holds comma-separated relevant product IDs |
| Config | `python-dotenv` loads project-root `.env` | CLI `--host`/`--port` override env |

**Superseded:** Local `BAAI/bge-large-en-v1.5` dense embeddings — replaced by OpenAI `text-embedding-3-small` per product requirement.

## Current State

- [x] Project layout (`config.py`, `src/*`, `run_evaluation.py`)
- [x] `docker-compose.yml` for optional local Milvus 2.4.15
- [x] `.env`-driven Milvus connection (`MILVUS_HOST` / `MILVUS_PORT` / optional `MILVUS_TOKEN`)
- [x] `.env`-driven OpenAI dense embeddings (`text-embedding-3-small`, dim=1536)
- [x] Smoke-tested dense + sparse + hybrid (RRF / Weighted) against remote Milvus 3.0.0
- [x] `DataProcessor` (products + QA)
- [x] `EmbeddingEngine` (OpenAI dense batch, BM25 sparse, reranker, disk cache)
- [x] `MilvusManager` (multi-vector schema, indexes, batch insert)
- [x] Six search strategies
- [x] Evaluator (Recall@5, Recall@10, MRR, avg latency)
- [x] CLI entrypoint + markdown/CSV report
- [x] `README.md` + `requirements.txt`
- [ ] Full OpenAI eval: last run aborted mid-`IVF_FLAT` (~46%); HNSW_ef64/ef128 finished; report not refreshed. Restarted via nohup.

## Next Steps / TODOs

- [ ] Finish OpenAI eval and refresh `results/evaluation_report.*` (check `/tmp/rag-eval-openai.log`)
- [ ] Migrate ORM (`Collection`) → `MilvusClient` before pymilvus 3.1 removal
- [ ] Try BGE-M3 learned sparse as an alternative to BM25
- [ ] Sweep `WeightedRanker` weights and `rerank_candidates`
- [x] In-memory query dense cache (reuse OpenAI vectors across strategies per question)
- [ ] Add IVF_SQ8 / HNSW `M` sensitivity study for larger corpora
- [ ] Chunking experiments if documents grow beyond single-product blurbs
- [ ] CI job: lint + unit tests for metrics / QA ID parsing (no Milvus required)

## Crucial Instructions

1. **Always read this file first** before starting a new task.
2. Prefer extending existing modules over adding new frameworks (keep pymilvus-native).
3. **Always use Milvus settings from `.env`** (`MILVUS_HOST`, `MILVUS_PORT`); do not hardcode localhost unless the user asks for local Docker.
4. **Dense embeddings must use `.env` OpenAI settings** (`EMBEDDING_MODEL=text-embedding-3-small`) unless the user explicitly requests otherwise.
5. Never commit `.env`, API keys, `cache/`, or `milvus/volumes/`.
6. After major architectural changes or finished milestones, **update this file** (decisions + checklist).
7. Keep files concise; put tunables in `config.py`, not scattered magic numbers.
