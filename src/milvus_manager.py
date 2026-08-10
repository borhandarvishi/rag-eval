"""Milvus schema, indexing, and batched ingestion."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from tqdm import tqdm

from config import (
    DENSE_DIM,
    SPARSE_INDEX_PARAMS,
)


class MilvusManager:
    """Lifecycle helper for multi-vector product collections."""

    DENSE_FIELD = "dense_vector"
    SPARSE_FIELD = "sparse_vector"
    TEXT_FIELD = "text"
    ID_FIELD = "id"

    def __init__(
        self,
        host: str,
        port: str,
        alias: str = "default",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = str(port)
        self.alias = alias
        self.token = token or None
        self.timeout = timeout
        self._connected = False

    def connect(self) -> None:
        kwargs = {
            "alias": self.alias,
            "host": self.host,
            "port": self.port,
            "timeout": self.timeout,
        }
        if self.token:
            kwargs["token"] = self.token
        connections.connect(**kwargs)
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False

    def drop_collection(self, name: str) -> None:
        if utility.has_collection(name, using=self.alias):
            utility.drop_collection(name, using=self.alias)

    def create_collection(
        self,
        name: str,
        *,
        dense_index_type: str,
        dense_index_params: dict,
        drop_existing: bool = True,
        dense_metric: str = "IP",
    ) -> Collection:
        if drop_existing:
            self.drop_collection(name)

        fields = [
            FieldSchema(
                name=self.ID_FIELD,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=self.TEXT_FIELD,
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name=self.DENSE_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=DENSE_DIM,
            ),
            FieldSchema(
                name=self.SPARSE_FIELD,
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="E-commerce products multi-vector RAG eval",
            enable_dynamic_field=False,
        )
        collection = Collection(name=name, schema=schema, using=self.alias)

        collection.create_index(
            field_name=self.DENSE_FIELD,
            index_params={
                "index_type": dense_index_type,
                "metric_type": dense_metric,
                "params": dense_index_params,
            },
        )
        collection.create_index(
            field_name=self.SPARSE_FIELD,
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": SPARSE_INDEX_PARAMS,
            },
        )
        return collection

    def insert_batch(
        self,
        collection: Collection,
        ids: Sequence[int],
        texts: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[dict],
        batch_size: int = 64,
    ) -> int:
        total = len(ids)
        if not (total == len(texts) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("Insert arrays must have equal length")

        inserted = 0
        for start in tqdm(
            range(0, total, batch_size), desc=f"Insert→{collection.name}", unit="batch"
        ):
            end = min(start + batch_size, total)
            # Column-oriented entities for pymilvus
            entities = [
                list(ids[start:end]),
                list(texts[start:end]),
                [list(map(float, v)) for v in dense_vectors[start:end]],
                list(sparse_vectors[start:end]),
            ]
            collection.insert(entities)
            inserted += end - start

        collection.flush()
        return inserted

    def load(self, collection: Collection) -> None:
        collection.load()

    def get_collection(self, name: str) -> Collection:
        return Collection(name=name, using=self.alias)

    def release_and_drop(self, name: str) -> None:
        if utility.has_collection(name, using=self.alias):
            col = Collection(name=name, using=self.alias)
            try:
                col.release()
            except Exception:
                pass
            utility.drop_collection(name, using=self.alias)

    @staticmethod
    def fetch_texts(collection: Collection, ids: Sequence[int]) -> Dict[int, str]:
        if not ids:
            return {}
        # Milvus expr IN clause
        id_list = ", ".join(str(i) for i in ids)
        rows = collection.query(
            expr=f"id in [{id_list}]",
            output_fields=["id", "text"],
        )
        return {int(r["id"]): str(r["text"]) for r in rows}
