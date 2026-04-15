from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.services.sql_generator import SQLGenerator


def test_generate_returns_plain_sql(
    sql_generator: SQLGenerator,
    llm_client_mock: AsyncMock,
) -> None:
    llm_client_mock.generate.return_value = "SELECT * FROM stack.accounts LIMIT 10"

    result = asyncio.run(sql_generator.generate("system", "user"))

    assert result.raw_response == "SELECT * FROM stack.accounts LIMIT 10"
    assert result.extracted_sql == "SELECT * FROM stack.accounts LIMIT 10"
    assert result.explanation is None


def test_generate_extracts_sql_from_markdown_block(
    sql_generator: SQLGenerator,
    llm_client_mock: AsyncMock,
) -> None:
    llm_client_mock.generate.return_value = """```sql
SELECT id FROM stack.accounts
LIMIT 5
```"""

    result = asyncio.run(sql_generator.generate("system", "user"))

    assert result.extracted_sql == "SELECT id FROM stack.accounts\nLIMIT 5"
    assert result.explanation is None


def test_generate_extracts_sql_and_explanation_from_mixed_response(
    sql_generator: SQLGenerator,
    llm_client_mock: AsyncMock,
) -> None:
    llm_client_mock.generate.return_value = """
    Ниже SQL.

    ```sql
    SELECT id FROM stack.accounts
    ```

    Этот запрос получает идентификаторы.
    """

    result = asyncio.run(sql_generator.generate("system", "user"))

    assert result.extracted_sql == "SELECT id FROM stack.accounts"
    assert result.explanation == "Ниже SQL.\n\nЭтот запрос получает идентификаторы."


def test_generate_returns_no_sql_for_garbage_response(
    sql_generator: SQLGenerator,
    llm_client_mock: AsyncMock,
) -> None:
    llm_client_mock.generate.return_value = "Не могу построить запрос, вот общие рассуждения."

    result = asyncio.run(sql_generator.generate("system", "user"))

    assert result.raw_response == "Не могу построить запрос, вот общие рассуждения."
    assert result.extracted_sql is None
    assert result.explanation is None


def test_generate_returns_no_sql_for_empty_response(
    sql_generator: SQLGenerator,
    llm_client_mock: AsyncMock,
) -> None:
    llm_client_mock.generate.return_value = ""

    result = asyncio.run(sql_generator.generate("system", "user"))

    assert result.raw_response == ""
    assert result.extracted_sql is None
    assert result.explanation is None
