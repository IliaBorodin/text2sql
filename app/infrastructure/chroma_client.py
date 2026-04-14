"""Synchronous ChromaDB client for schema similarity search."""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.core.config import Settings
from app.core.exceptions import SchemaRetrievalError
from app.core.models import SearchResult

logger = logging.getLogger(__name__)


class ChromaClient:
    """Read-only client for the indexed table schemas collection."""

    def __init__(self, settings: Settings):
        self._collection_name = settings.chroma_collection

        try:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
            self._collection = self._client.get_collection(self._collection_name)
        except ValueError as exc:
            raise SchemaRetrievalError(
                f"Коллекция {self._collection_name} не найдена. "
                "Запустите scripts/index_schema.py"
            ) from exc
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Ошибка инициализации ChromaDB: {exc}"
            ) from exc

        table_count = self._collection.count()
        logger.info(
            "ChromaDB collection loaded name=%s tables=%s",
            self._collection_name,
            table_count,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
    ) -> list[SearchResult]:
        """Search nearest schema tables for the given embedding."""

        logger.debug("Searching ChromaDB n_results=%s", n_results)

        try:
            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise SchemaRetrievalError(f"Ошибка поиска в ChromaDB: {exc}") from exc

        ids = self._first_result_list(raw.get("ids"))
        documents = self._first_result_list(raw.get("documents"))
        metadatas = self._first_result_list(raw.get("metadatas"))
        distances = self._first_result_list(raw.get("distances"))

        results: list[SearchResult] = []
        for table_id, document, meta, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            metadata = dict(meta or {})
            table_name = (
                metadata.get("table_name")
                or metadata.get("name")
                or str(table_id)
            )
            distance_value = float(distance)
            score = 1.0 / (1.0 + distance_value)

            logger.debug(
                "ChromaDB result table_name=%s distance=%s score=%s",
                table_name,
                distance_value,
                score,
            )

            results.append(
                SearchResult(
                    table_name=table_name,
                    score=score,
                    metadata={**metadata, "document": document},
                )
            )

        if not results:
            logger.warning("ChromaDB search returned no results")

        return sorted(results, key=lambda item: item.score, reverse=True)

    def get_table_count(self) -> int:
        """Return the number of indexed tables in the collection."""

        return self._collection.count()

    def get_all_table_names(self) -> list[str]:
        """Return all table names stored in the ChromaDB collection."""

        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise SchemaRetrievalError(
                f"Ошибка чтения таблиц из ChromaDB: {exc}"
            ) from exc

        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []

        table_names = {
            (
                (metadata or {}).get("table_name")
                or (metadata or {}).get("name")
                or str(table_id)
            )
            for table_id, metadata in zip(ids, metadatas)
        }

        return sorted(table_names)

    @staticmethod
    def _first_result_list(value: Any) -> list[Any]:
        """Unwrap ChromaDB query response shaped as a list per query."""

        if not value:
            return []
        first = value[0]
        return first if isinstance(first, list) else value
