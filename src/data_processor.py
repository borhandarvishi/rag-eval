"""Efficient CSV loading and corpus preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class ProductRecord:
    id: int
    text: str
    title: str
    category: str


@dataclass(frozen=True, slots=True)
class QARecord:
    qid: str
    question: str
    relevant_ids: tuple[int, ...]


class DataProcessor:
    """Load product corpus and ground-truth QA with minimal copies."""

    def __init__(self, products_csv: Path, qa_csv: Path) -> None:
        self.products_csv = Path(products_csv)
        self.qa_csv = Path(qa_csv)

    def load_products(self) -> List[ProductRecord]:
        df = pd.read_csv(
            self.products_csv,
            encoding="utf-8-sig",
            dtype={
                "id": "int64",
                "category": "string",
                "title": "string",
                "description": "string",
            },
        )
        required = {"id", "category", "title", "description"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Products CSV missing columns: {sorted(missing)}")

        texts = (
            df["title"].fillna("")
            + " | "
            + df["category"].fillna("")
            + " | "
            + df["description"].fillna("")
        )
        return [
            ProductRecord(
                id=int(pid),
                text=text,
                title=str(title),
                category=str(category),
            )
            for pid, text, title, category in zip(
                df["id"].to_numpy(),
                texts.to_numpy(),
                df["title"].to_numpy(),
                df["category"].to_numpy(),
                strict=True,
            )
        ]

    def load_qa(self) -> List[QARecord]:
        df = pd.read_csv(self.qa_csv, encoding="utf-8-sig")
        qid_col = "question ids" if "question ids" in df.columns else "qid"
        gt_col = "ids" if "ids" in df.columns else "id"
        if "question" not in df.columns:
            raise ValueError("QA CSV must contain a 'question' column")
        if qid_col not in df.columns or gt_col not in df.columns:
            raise ValueError(
                f"QA CSV must contain '{qid_col}' and ground-truth column "
                f"('{gt_col}' expected)"
            )

        records: List[QARecord] = []
        for row in df.to_dict(orient="records"):
            relevant = self._parse_id_list(row[gt_col])
            if not relevant:
                continue
            records.append(
                QARecord(
                    qid=str(row[qid_col]),
                    question=str(row["question"]),
                    relevant_ids=relevant,
                )
            )
        return records

    @staticmethod
    def _parse_id_list(value: object) -> tuple[int, ...]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return tuple()
        if isinstance(value, int):
            return (int(value),)
        text = str(value).strip().strip('"').strip("'")
        if not text:
            return tuple()
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        return tuple(int(p) for p in parts)

    @staticmethod
    def corpus_texts(products: Sequence[ProductRecord]) -> List[str]:
        return [p.text for p in products]
