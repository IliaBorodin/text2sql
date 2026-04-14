from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from app.core.models import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> HealthStatus:
    ollama = request.app.state.ollama
    db = request.app.state.db
    chroma = request.app.state.chroma

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

    ollama_ok, db_ok = await asyncio.gather(_check_ollama(), _check_db())

    try:
        chroma_count = chroma.get_table_count()
    except Exception:
        logger.exception("Проверка ChromaDB завершилась ошибкой")
        chroma_count = 0

    status = "ok" if (ollama_ok and db_ok and chroma_count > 0) else "error"

    return HealthStatus(
        status=status,
        ollama_available=ollama_ok,
        db_connected=db_ok,
        chroma_tables_count=chroma_count,
    )
