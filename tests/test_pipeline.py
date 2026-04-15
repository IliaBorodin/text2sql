from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock
from unittest.mock import Mock

from app.core.models import PipelineTimings
from app.core.models import QueryProgressEvent
from app.core.models import QueryResponse


QUESTION = "Покажи активные лицевые счета"


def _assert_response_shape(response: QueryResponse) -> None:
    assert isinstance(response.success, bool)
    assert isinstance(response.timings, PipelineTimings)
    assert response.timings.total_ms >= 0
    assert response.timings.schema_retrieval_ms >= 0
    assert response.timings.prompt_build_ms >= 0
    assert response.timings.llm_generation_ms >= 0
    assert response.timings.validation_ms >= 0
    assert response.timings.execution_ms >= 0

    if response.success:
        assert response.sql is not None
        assert response.result is not None
        assert response.error is None
    else:
        assert response.error is not None


async def _collect_events(pipeline, question: str) -> list[QueryProgressEvent]:
    return [event async for event in pipeline.execute_with_progress(question)]


def _prepare_retrieval(
    chroma_client_mock: Mock,
    register_xdic_table: Callable[..., None],
    make_search_result: Callable[..., object],
) -> None:
    register_xdic_table(
        "stack.accounts",
        ddl=(
            "CREATE TABLE stack.accounts (\n"
            "    id bigint,\n"
            "    status text,\n"
            "    customer_id bigint\n"
            ");"
        ),
        description="Лицевые счета",
        relations=[
            {
                "from_field": "customer_id",
                "to_table": "stack.customers",
            }
        ],
    )
    register_xdic_table(
        "stack.customers",
        ddl=(
            "CREATE TABLE stack.customers (\n"
            "    id bigint,\n"
            "    full_name text\n"
            ");"
        ),
        description="Клиенты",
    )
    chroma_client_mock.search.return_value = [
        make_search_result("stack.accounts", 0.98),
        make_search_result("stack.customers", 0.91),
    ]


def test_pipeline_execute_happy_path(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    db_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.return_value = (
        "```sql\n"
        "SELECT id, status FROM stack.accounts WHERE status = 'active'\n"
        "```"
    )
    db_client_mock.execute.return_value = [
        {"id": 101, "status": "active"},
        {"id": 102, "status": "active"},
    ]

    response = asyncio.run(pipeline_factory().execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is True
    assert response.tables_used == ["stack.accounts", "stack.customers"]
    assert response.sql == (
        "SELECT id, status FROM stack.accounts WHERE status = 'active'\nLIMIT 100"
    )
    assert response.result is not None
    assert response.result.row_count == 2
    assert response.result.rows[0]["id"] == 101
    assert response.timings.total_ms > 0


def test_pipeline_self_correction_succeeds_on_second_attempt(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    db_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.side_effect = [
        "```sql\nSELECT missing_column FROM stack.accounts\n```",
        "```sql\nSELECT id FROM stack.accounts\n```",
    ]
    db_client_mock.execute.side_effect = [
        Exception('column "missing_column" does not exist'),
        [{"id": 101}],
    ]

    events = asyncio.run(_collect_events(pipeline_factory(max_retries=1), QUESTION))
    response = events[-1].response

    assert response is not None
    _assert_response_shape(response)
    assert response.success is True
    assert response.sql == "SELECT id FROM stack.accounts\nLIMIT 100"
    assert response.result is not None
    assert response.result.row_count == 1
    assert llm_client_mock.generate.await_count == 2
    assert db_client_mock.execute.await_count == 2
    assert any(
        event.phase == "generation" and event.data.get("attempt") == 1
        for event in events
    )
    assert any(
        event.phase == "execution"
        and event.data.get("attempt") == 1
        and event.message == "Исправленный SQL выполнен успешно"
        for event in events
    )
    correction_prompt = llm_client_mock.generate.await_args_list[1].kwargs["prompt"]
    assert "ИСПРАВЛЕНИЕ ОШИБКИ" in correction_prompt
    assert 'column "missing_column" does not exist' in correction_prompt


def test_pipeline_self_correction_fails_after_retry(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.side_effect = [
        "```sql\nDELETE FROM stack.accounts\n```",
        "```sql\nUPDATE stack.accounts SET status = 'closed'\n```",
    ]

    response = asyncio.run(pipeline_factory(max_retries=1).execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is False
    assert response.sql == "UPDATE stack.accounts SET status = 'closed'"
    assert response.error is not None
    assert "SQL не прошёл проверку после 1 попыток" in response.error
    assert "Update" in response.error
    assert llm_client_mock.generate.await_count == 2


def test_pipeline_returns_error_when_retrieval_finds_no_tables(
    pipeline_factory,
    chroma_client_mock: Mock,
) -> None:
    chroma_client_mock.search.return_value = []

    response = asyncio.run(pipeline_factory().execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is False
    assert response.sql is None
    assert response.result is None
    assert response.error == "Ошибка поиска таблиц: Не найдено релевантных таблиц"
    assert response.timings.prompt_build_ms == 0
    assert response.timings.llm_generation_ms == 0


def test_pipeline_returns_error_when_llm_returns_garbage(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.return_value = "Я не смог построить SQL, но могу объяснить предметную область."

    response = asyncio.run(pipeline_factory().execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is False
    assert response.sql is None
    assert response.error == "LLM не вернула SQL-запрос. Попробуйте переформулировать вопрос."
    assert response.timings.total_ms > 0
    assert response.timings.validation_ms == 0


def test_pipeline_returns_error_when_llm_is_unavailable(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.side_effect = ConnectionError("LLM timeout")

    response = asyncio.run(pipeline_factory().execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is False
    assert response.sql is None
    assert response.error is not None
    assert "Ошибка генерации SQL" in response.error
    assert "LLM timeout" in response.error


def test_pipeline_returns_error_when_database_is_unavailable(
    pipeline_factory,
    llm_client_mock: AsyncMock,
    db_client_mock: AsyncMock,
    chroma_client_mock: Mock,
    register_xdic_table,
    make_search_result,
) -> None:
    _prepare_retrieval(chroma_client_mock, register_xdic_table, make_search_result)
    llm_client_mock.generate.return_value = "```sql\nSELECT id FROM stack.accounts\n```"
    db_client_mock.execute.side_effect = ConnectionError("database unavailable")

    response = asyncio.run(pipeline_factory(max_retries=0).execute(QUESTION))

    _assert_response_shape(response)
    assert response.success is False
    assert response.sql == "SELECT id FROM stack.accounts\nLIMIT 100"
    assert response.result is None
    assert response.error is not None
    assert "SQL не прошёл проверку после 0 попыток" in response.error
    assert "Ошибка выполнения SQL: database unavailable" in response.error
