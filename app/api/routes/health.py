from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from app.core.models import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _search_path_contains_schema(search_path: str | None, schema: str | None) -> bool:
    if not search_path or not schema:
        return False

    normalized_parts = {
        part.strip().strip('"').lower()
        for part in search_path.split(",")
    }
    return schema.lower() in normalized_parts


@router.get("/health")
async def health(request: Request) -> HealthStatus:
    ollama = request.app.state.ollama
    db = request.app.state.db
    chroma = request.app.state.chroma
    required_schema = getattr(db, "db_schema", None)

    async def _check_ollama() -> bool:
        try:
            return await ollama.is_available()
        except Exception:
            logger.exception("Проверка Ollama завершилась ошибкой")
            return False

    async def _check_db() -> bool:
        try:
            return await db.is_connected()
        except Exception:
            logger.exception("Проверка PostgreSQL завершилась ошибкой")
            return False

    async def _get_search_path() -> str | None:
        try:
            return await db.get_search_path()
        except Exception:
            logger.exception("Чтение PostgreSQL search_path завершилось ошибкой")
            return None

    ollama_ok, db_ok, search_path = await asyncio.gather(
        _check_ollama(),
        _check_db(),
        _get_search_path(),
    )

    try:
        chroma_count = chroma.get_table_count()
    except Exception:
        logger.exception("Проверка ChromaDB завершилась ошибкой")
        chroma_count = 0

    schema_ok = db_ok and _search_path_contains_schema(search_path, required_schema)

    if ollama_ok and db_ok and chroma_count > 0:
        status = "ok" if schema_ok else "warning"
    else:
        status = "error"

    return HealthStatus(
        status=status,
        ollama_available=ollama_ok,
        db_connected=db_ok,
        chroma_tables_count=chroma_count,
        search_path=search_path,
        schema_ok=schema_ok,
    )
