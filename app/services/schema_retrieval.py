from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import SchemaRetrievalError
from app.core.interfaces import EmbeddingClient, VectorStore
from app.core.models import SearchResult, TableContext
from app.infrastructure.xdic.parser import XdicParser

logger = logging.getLogger(__name__)


class SchemaRetrievalService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        xdic_parser: XdicParser,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._xdic_parser = xdic_parser

    async def retrieve(self, question: str, max_tables: int = 10) -> list[TableContext]:
        try:
            embedding = await self._embedding_client.embed(question)
        except Exception as exc:  # pragma: no cover - depends on external client
            detail = str(exc) or exc.__class__.__name__
            raise SchemaRetrievalError(
                f"Не удалось получить эмбеддинг: {detail}"
            ) from exc

        search_results = self._vector_store.search(embedding, n_results=max_tables)
        self._log_search_results(search_results)
        if not search_results:
            raise SchemaRetrievalError("Не найдено релевантных таблиц")

        primary_tables: dict[str, dict[str, Any]] = {}
        for search_result in search_results:
            table_name = search_result.table_name
            table_payload = self._load_table_payload(table_name, score=search_result.score)
            if table_payload is None:
                continue
            primary_tables[table_name] = table_payload

        if not primary_tables:
            raise SchemaRetrievalError("Не удалось получить контекст ни для одной таблицы")

        related_candidates: dict[str, int] = {}
        for table_name in primary_tables:
            for relation in self._get_table_relations(table_name):
                related_table = self._get_related_table_name(relation, table_name)
                if not related_table or related_table in primary_tables:
                    continue
                related_candidates[related_table] = related_candidates.get(related_table, 0) + 1

        expanded_tables: dict[str, dict[str, Any]] = {}
        remaining_slots = max(0, max_tables - len(primary_tables))
        sorted_related = sorted(
            related_candidates.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
        for table_name, _ in sorted_related[:remaining_slots]:
            table_payload = self._load_table_payload(table_name, score=0.0)
            if table_payload is None:
                continue
            expanded_tables[table_name] = table_payload

        selected_tables = {**primary_tables, **expanded_tables}
        if not selected_tables:
            raise SchemaRetrievalError("Не удалось получить контекст ни для одной таблицы")

        all_selected = set(selected_tables)
        relations_by_table: dict[str, list[str]] = {table_name: [] for table_name in selected_tables}
        for table_name in selected_tables:
            for relation in self._get_table_relations(table_name):
                formatted = self._format_relation(relation, table_name, all_selected)
                if formatted and formatted not in relations_by_table[table_name]:
                    relations_by_table[table_name].append(formatted)

        contexts = [
            TableContext(
                name=table_name,
                ddl=payload["ddl"],
                description=payload["description"],
                relevance_score=payload["score"],
                relations=relations_by_table.get(table_name, []),
            )
            for table_name, payload in selected_tables.items()
        ]
        contexts.sort(key=lambda ctx: ctx.relevance_score, reverse=True)

        logger.info(
            "Найдено %s таблиц для вопроса: %s",
            len(contexts),
            [context.name for context in contexts],
        )
        for context in contexts:
            logger.debug(
                "Table %s: score=%s, relations=%s, ddl_length=%s",
                context.name,
                context.relevance_score,
                len(context.relations),
                len(context.ddl),
            )

        return contexts

    def _log_search_results(self, search_results: list[SearchResult]) -> None:
        logger.debug(
            "Vector search returned %s results: %s",
            len(search_results),
            [result.table_name for result in search_results],
        )

    def _load_table_payload(self, table_name: str, score: float) -> dict[str, Any] | None:
        table = self._xdic_parser.tables.get(table_name)
        if table is None:
            logger.warning("Таблица %s найдена, но отсутствует в XdicParser", table_name)
            return None

        ddl = self._xdic_parser.get_create_table_sql(table_name)
        if not ddl or not ddl.strip():
            logger.warning("XdicParser не содержит DDL для таблицы %s", table_name)
            return None

        try:
            context = self._xdic_parser.get_table_context(table_name)
        except Exception:
            logger.exception("Не удалось получить контекст таблицы %s", table_name)
            return None

        description = self._extract_description(context, table)
        return {
            "ddl": ddl,
            "description": description,
            "score": score,
        }

    def _extract_description(self, context: Any, table: Any) -> str:
        if isinstance(context, dict):
            description = context.get("description")
            if isinstance(description, str):
                return description
        if hasattr(context, "description") and isinstance(context.description, str):
            return context.description
        table_description = getattr(table, "description", "")
        return table_description if isinstance(table_description, str) else ""

    def _get_table_relations(self, table_name: str) -> list[dict[str, str]]:
        table = self._xdic_parser.tables.get(table_name)
        if table is None:
            return []

        relations: list[dict[str, str]] = []
        fields = getattr(table, "fields", {})
        for field in getattr(fields, "values", lambda: [])():
            if not getattr(field, "is_foreign_key", False):
                continue
            target_table = getattr(field, "referenced_table", "")
            if not target_table:
                continue
            relations.append(
                {
                    "from_table": table_name,
                    "from_field": getattr(field, "name", ""),
                    "to_table": target_table,
                    "to_field": "row_id",
                }
            )

        for other_name, other_table in self._xdic_parser.tables.items():
            if other_name == table_name:
                continue
            other_fields = getattr(other_table, "fields", {})
            for field in getattr(other_fields, "values", lambda: [])():
                if not getattr(field, "is_foreign_key", False):
                    continue
                if getattr(field, "referenced_table", "") != table_name:
                    continue
                relations.append(
                    {
                        "from_table": other_name,
                        "from_field": getattr(field, "name", ""),
                        "to_table": table_name,
                        "to_field": "row_id",
                    }
                )

        return relations

    def _get_related_table_name(self, relation: dict[str, str], table_name: str) -> str | None:
        from_table = relation.get("from_table")
        to_table = relation.get("to_table")
        if from_table == table_name:
            return to_table
        if to_table == table_name:
            return from_table
        return None

    def _format_relation(
        self,
        relation: dict[str, str],
        table_name: str,
        all_selected: set[str],
    ) -> str | None:
        from_table = relation.get("from_table")
        from_field = relation.get("from_field")
        to_table = relation.get("to_table")
        to_field = relation.get("to_field") or "row_id"

        if not from_table or not from_field or not to_table:
            return None
        if from_table not in all_selected or to_table not in all_selected:
            return None
        if table_name not in {from_table, to_table}:
            return None

        return f"{from_table}.{from_field} → {to_table}.{to_field}"
