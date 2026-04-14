from __future__ import annotations

import logging
import time

from app.core.exceptions import DatabaseError
from app.core.interfaces import DatabaseClient
from app.core.models import SQLResult

logger = logging.getLogger(__name__)


class SQLExecutor:
    def __init__(self, db_client: DatabaseClient):
        self._db_client = db_client

    async def execute(self, sql: str) -> SQLResult:
        logger.debug("SQL preview: %s", sql[:200])

        start = time.perf_counter()
        try:
            rows = await self._db_client.execute(sql)
            elapsed_ms = (time.perf_counter() - start) * 1000

            columns = list(rows[0].keys()) if rows else []
            row_count = len(rows)

            logger.info("SQL выполнен: %s строк за %.0fмс", row_count, elapsed_ms)

            return SQLResult(
                columns=columns,
                rows=rows,
                row_count=row_count,
                execution_time_ms=round(elapsed_ms, 2),
            )
        except DatabaseError:
            logger.error("Ошибка выполнения SQL", exc_info=True)
            raise
        except Exception as exc:
            logger.error("Ошибка выполнения SQL", exc_info=True)
            raise DatabaseError(f"Ошибка выполнения SQL: {exc}") from exc
