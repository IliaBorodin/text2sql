"""Async PostgreSQL client for read-only SQL execution."""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings
from app.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


def _quote_ident(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


class PostgresClient:
    """Async PostgreSQL client implementing the database protocol."""

    def __init__(self, settings: Settings):
        self._db_url = settings.db_url
        self._sql_timeout = settings.sql_timeout
        self._max_rows = settings.sql_max_rows
        self._db_schema = settings.db_schema
        self._db_host = settings.db_host
        self._db_name = settings.db_name
        self._engine: AsyncEngine | None = None

    @property
    def db_schema(self) -> str:
        return self._db_schema

    async def connect(self) -> None:
        """Create the async engine and verify the database connection."""

        configured_search_path = f"{_quote_ident(self._db_schema)}, public"
        self._engine = create_async_engine(
            self._db_url,
            pool_size=5,
            max_overflow=5,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"statement_cache_size": 0},
            echo=False,
        )

        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_search_path(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f"SET search_path TO {configured_search_path}")
            finally:
                cursor.close()

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            if self._engine is not None:
                await self._engine.dispose()
                self._engine = None

            detail = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
            logger.error(
                "Failed to connect to PostgreSQL host=%s db=%s schema=%s",
                self._db_host,
                self._db_name,
                self._db_schema,
                exc_info=exc,
            )
            raise DatabaseError(
                f"Не удалось подключиться к PostgreSQL: {detail}"
            ) from exc

        logger.info(
            "PostgreSQL connected host=%s db=%s schema=%s search_path=%s",
            self._db_host,
            self._db_name,
            self._db_schema,
            configured_search_path,
        )

    async def disconnect(self) -> None:
        """Dispose the async engine if it exists."""

        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

        logger.info("PostgreSQL disconnected")

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute SQL in a read-only transaction with a statement timeout."""

        if self._engine is None:
            raise DatabaseError("Нет подключения к БД. Вызовите connect()")

        timeout_ms = self._sql_timeout * 1000
        started_at = time.perf_counter()
        logger.debug("Executing SQL: %s", sql[:200])

        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(f"SET LOCAL statement_timeout = '{timeout_ms}'")
                )
                await conn.execute(text("SET TRANSACTION READ ONLY"))

                if params is not None:
                    result = await conn.execute(text(sql), params)
                else:
                    result = await conn.execute(text(sql))

                rows = result.mappings().fetchmany(self._max_rows)

            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "PostgreSQL query completed rows=%s duration_ms=%.2f schema=%s",
                len(rows),
                elapsed_ms,
                self._db_schema,
            )
            return [dict(row) for row in rows]
        except OperationalError as exc:
            error_str = str(exc.orig) if exc.orig else str(exc)
            logger.error("PostgreSQL execute failed sql=%s", sql, exc_info=exc)
            if "statement timeout" in error_str or "canceling statement" in error_str:
                raise DatabaseError(f"Таймаут запроса ({self._sql_timeout}с)") from exc
            raise DatabaseError(f"Ошибка БД: {error_str}") from exc
        except ProgrammingError as exc:
            error_str = str(exc.orig) if exc.orig else str(exc)
            logger.error("PostgreSQL execute failed sql=%s", sql, exc_info=exc)
            raise DatabaseError(f"Ошибка SQL: {error_str}") from exc
        except SQLAlchemyError as exc:
            logger.error("PostgreSQL execute failed sql=%s", sql, exc_info=exc)
            raise DatabaseError(f"Ошибка БД: {exc}") from exc

    async def is_connected(self) -> bool:
        """Return whether the database is reachable for health checks."""

        if self._engine is None:
            return False

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SET LOCAL statement_timeout = '5000'"))
                await conn.execute(text("SELECT 1"))
        except Exception:
            return False

        return True

    async def get_search_path(self) -> str | None:
        """Return the current PostgreSQL search_path for health checks."""

        if self._engine is None:
            return None

        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SHOW search_path"))
                return result.scalar_one_or_none()
        except Exception:
            logger.exception("Failed to read PostgreSQL search_path")
            return None
