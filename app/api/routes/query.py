from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.models import QueryProgressEvent, QueryRequest, QueryResponse
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


def _serialize_query_response(response: QueryResponse) -> dict[str, Any]:
    payload = response.model_dump(mode="python")
    result = response.result
    if result is not None:
        payload["result"] = {
            "columns": result.columns,
            "rows": _sanitize_rows(result.rows),
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
        }
    return payload


def _serialize_progress_event(event: QueryProgressEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": event.phase,
        "message": event.message,
        "data": event.data,
    }
    if event.response is not None:
        payload["response"] = _serialize_query_response(event.response)
    return payload


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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

    return JSONResponse(content=_serialize_query_response(response))


@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    pipeline: Pipeline = request.app.state.pipeline

    logger.info("SSE запрос: %s", body.question[:100])

    async def event_stream():
        async for event in pipeline.execute_with_progress(body.question):
            event_name = "done" if event.phase == "done" else "error" if event.phase == "error" else "progress"
            yield _format_sse(event_name, _serialize_progress_event(event))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
