from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator


@pytest.fixture
def sql_validator() -> SQLValidator:
    return SQLValidator(default_limit=100)


@pytest.fixture
def llm_client_mock() -> AsyncMock:
    client = AsyncMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def sql_generator(
    llm_client_mock: AsyncMock,
    sql_validator: SQLValidator,
) -> SQLGenerator:
    return SQLGenerator(llm_client=llm_client_mock, sql_validator=sql_validator)
