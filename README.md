# Milvus RAG Retrieval Evaluation

Benchmark **dense**, **sparse (BM25)**, **hybrid**, and **cross-encoder rerank** retrieval strategies on an e-commerce product corpus using **Milvus 2.4+ / 3.0**.

## Architecture

```
data/*.csv
    ↓
DataProcessor          → ProductRecord / QARecord
    ↓
EmbeddingEngine        → Dense (OpenAI text-embedding-3-small) + Sparse (BM25) + Reranker
    ↓
MilvusManager          → multi-vector schema, HNSW / IVF_FLAT + sparse index
    ↓
SearchStrategies       → 6 retrieval methods
    ↓
Evaluator + metrics    → Recall@5/10, MRR, avg latency → results/
```

| Module | Role |
|--------|------|
| `config.py` | Models, batch sizes, index params; loads `.env` |
| `src/data_processor.py` | CSV → typed records |
| `src/embeddings.py` | Batched encode + disk cache |
| `src/milvus_manager.py` | Schema, indexes, batch insert |
| `src/search_strategies.py` | Six search methodologies |
| `src/metrics.py` / `src/evaluator.py` | Metrics + report |
| `run_evaluation.py` | CLI entrypoint |

## Dependencies

See [`requirements.txt`](requirements.txt). Core stack:

- `pymilvus[model]` (BM25 + Milvus client)
- `openai` (`text-embedding-3-small` dense vectors)
- `sentence-transformers` / `torch` (cross-encoder reranker only)
- `pandas`, `numpy`, `python-dotenv`, `tqdm`

## Setup

**1. Configure `.env`**

```bash
cp .env.example .env
```

Required keys:

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL_PROVIDER=openai
EMBEDDING_MODEL_API_KEY=sk-...

MILVUS_HOST=ailab.techstrata.com
MILVUS_PORT=19530
```

The evaluation script reads Milvus + embedding settings from `.env` (CLI `--host` / `--port` override Milvus if needed).

Optional local Milvus via Docker:

```bash
docker compose up -d
```

**2. Python env**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If you see: No module named 'pymilvus.model'
pip install "pymilvus.model"
# equivalent: pip install "pymilvus[model]"
```

Dense vectors are fetched from OpenAI (batched). First rerank run downloads `BAAI/bge-reranker-base`. Dense/BM25 artifacts are cached under `cache/`.

## Run evaluation

```bash
python run_evaluation.py
```

Useful flags:

```bash
# Subset of dense indexes
python run_evaluation.py --indexes HNSW_ef64 IVF_FLAT

# Force re-embed
python run_evaluation.py --no-cache

# Keep collections in Milvus after the run
python run_evaluation.py --keep-collections

# Override .env host
python run_evaluation.py --host localhost --port 19530
```

Reports land in:

- `results/evaluation_report.md`
- `results/evaluation_report.csv`

## Six methodologies

| # | Method | What it does |
|---|--------|----------------|
| 1 | **Pure Dense** | ANN over `FLOAT_VECTOR` (`text-embedding-3-small`, dim=1536, IP) |
| 2 | **Pure Sparse** | ANN over `SPARSE_FLOAT_VECTOR` (BM25) |
| 3 | **Hybrid RRF** | Dense + sparse fused with `RRFRanker` |
| 4 | **Hybrid Weighted** | Dense + sparse fused with `WeightedRanker(0.7, 0.3)` |
| 5 | **Dense → Reranker** | Top-N dense candidates → `bge-reranker-base` |
| 6 | **Hybrid → Reranker** | Hybrid (RRF) candidates → cross-encoder |

## Index / latency benchmarking

Each methodology is measured on:

- **HNSW** with `ef=64` and `ef=128` (same build: `M=16`, `efConstruction=200`)
- **IVF_FLAT** (`nlist=128`, search `nprobe=16`)

Average end-to-end latency (encode query + Milvus search [+ rerank]) is logged in milliseconds.

## Data

| File | Purpose |
|------|---------|
| `data/ecommerce_products.csv` | Knowledge base (`id`, `category`, `title`, `description`) |
| `data/ecommerce_products_qa.csv` | Ground truth (`question ids`, `question`, `id` = relevant product IDs) |

## Project memory

See [`CURSOR_MEMORY.md`](CURSOR_MEMORY.md) for architectural decisions and current status (agents should read it first).
