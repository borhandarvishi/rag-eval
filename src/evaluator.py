"""End-to-end evaluation loop and report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from tqdm import tqdm

from src.data_processor import QARecord
from src.metrics import recall_at_k, reciprocal_rank
from src.search_strategies import SearchStrategies


@dataclass
class MethodStats:
    method: str
    index_type: str
    recalls_at_5: List[float] = field(default_factory=list)
    recalls_at_10: List[float] = field(default_factory=list)
    mrrs: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)

    def summarize(self) -> dict:
        n = len(self.mrrs) or 1
        return {
            "Method": self.method,
            "Index Type": self.index_type,
            "Recall@5": round(sum(self.recalls_at_5) / n, 4),
            "Recall@10": round(sum(self.recalls_at_10) / n, 4),
            "MRR": round(sum(self.mrrs) / n, 4),
            "Avg Latency (ms)": round(sum(self.latencies_ms) / n, 2),
        }


class Evaluator:
    """Iterate QA pairs across strategies and aggregate metrics."""

    METHOD_DISPLAY = {
        "1_pure_dense": "1. Pure Dense",
        "2_pure_sparse": "2. Pure Sparse (BM25)",
        "3_hybrid_rrf": "3. Hybrid RRF",
        "4_hybrid_weighted": "4. Hybrid Weighted",
        "5_dense_rerank": "5. Dense → Reranker",
        "6_hybrid_rerank": "6. Hybrid → Reranker",
    }

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def evaluate_index(
        self,
        strategies: SearchStrategies,
        qa_records: Sequence[QARecord],
        index_name: str,
    ) -> List[dict]:
        method_fns = strategies.method_map()
        stats: Dict[str, MethodStats] = {
            key: MethodStats(
                method=self.METHOD_DISPLAY.get(key, key),
                index_type=index_name,
            )
            for key in method_fns
        }

        for qa in tqdm(qa_records, desc=f"Eval[{index_name}]", unit="q"):
            for key, fn in method_fns.items():
                result = fn(qa.question, self.top_k)
                st = stats[key]
                st.recalls_at_5.append(
                    recall_at_k(result.ids, qa.relevant_ids, 5)
                )
                st.recalls_at_10.append(
                    recall_at_k(result.ids, qa.relevant_ids, 10)
                )
                st.mrrs.append(reciprocal_rank(result.ids, qa.relevant_ids))
                st.latencies_ms.append(result.latency_ms)

        return [s.summarize() for s in stats.values()]

    @staticmethod
    def to_dataframe(rows: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        col_order = [
            "Method",
            "Index Type",
            "Recall@5",
            "Recall@10",
            "MRR",
            "Avg Latency (ms)",
        ]
        return df[col_order]

    @staticmethod
    def save_report(df: pd.DataFrame, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "evaluation_report.csv"
        md_path = results_dir / "evaluation_report.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(
            "# Milvus Retrieval Evaluation Report\n\n"
            + df.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )
        return md_path
