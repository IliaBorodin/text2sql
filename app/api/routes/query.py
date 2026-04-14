from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.models import QueryRequest, QueryResponse
from app.services.pipeline import Pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    sanitized_rows: list[dict[str, Any]] = []

    for row in rows:
        sanitized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, Decimal):
                sanitized_row[key] = float(value)
            elif isinstance(value, (date, datetime)):
                sanitized_row[key] = value.isoformat()
            elif isinstance(value, (bytes, memoryview)):
                sanitized_row[key] = "<binary>"
            elif value is None or isinstance(value, (str, int, float, bool, list, dict)):
                sanitized_row[key] = value
            else:
                sanitized_row[key] = str(value)
        sanitized_rows.append(sanitized_row)

    return sanitized_rows


@router.post("/query")
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline: Pipeline = request.app.state.pipeline

    logger.info("Запрос: %s", body.question[:100])

    response = await pipeline.execute(body.question)

    if response.success:
        logger.info(
            "Успех: %d строк за %.0fмс",
            response.result.row_count if response.result else 0,
            response.timings.total_ms,
        )
    else:
        logger.warning("Ошибка: %s", response.error)

    payload = response.model_dump(mode="python")
    result = response.result
    if result is not None:
        payload["result"] = {
            "columns": result.columns,
            "rows": _sanitize_rows(result.rows),
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
        }

    return JSONResponse(content=payload)
