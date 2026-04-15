from __future__ import annotations

import pytest

from app.services.sql_validator import SQLValidator


def test_validate_accepts_quoted_cyrillic_identifiers(
    sql_validator: SQLValidator,
) -> None:
    result = sql_validator.validate('SELECT "Номер" FROM stack."Лицевые счета"')

    assert result.is_valid is True
    assert result.errors == []
    assert result.fixed_sql == 'SELECT "Номер" FROM stack."Лицевые счета"\nLIMIT 100'
    assert not any("Кириллические таблицы без явной схемы" in warning for warning in result.warnings)


def test_validate_accepts_select_with_join_where_group_order_and_limit(
    sql_validator: SQLValidator,
) -> None:
    sql = """
    SELECT a.id, COUNT(b.id) AS cnt
    FROM stack.accounts a
    JOIN stack.payments b ON b.account_id = a.id
    WHERE a.status = 'active'
    GROUP BY a.id
    ORDER BY cnt DESC
    LIMIT 10
    """

    result = sql_validator.validate(sql)

    assert result.is_valid is True
    assert result.errors == []
    assert result.fixed_sql == sql.strip()
    assert result.warnings == []


def test_validate_accepts_select_with_stack_schema(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("SELECT * FROM stack.accounts")

    assert result.is_valid is True
    assert result.fixed_sql == "SELECT * FROM stack.accounts\nLIMIT 100"


def test_validate_accepts_select_without_stack_schema(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("SELECT * FROM accounts")

    assert result.is_valid is True
    assert result.errors == []
    assert result.fixed_sql == "SELECT * FROM accounts\nLIMIT 100"


@pytest.mark.parametrize(
    ("sql", "keyword"),
    [
        ("INSERT INTO accounts(id) VALUES (1)", "Insert"),
        ("UPDATE accounts SET status = 'x'", "Update"),
        ("DELETE FROM accounts", "Delete"),
        ("DROP TABLE accounts", "Drop"),
        ("ALTER TABLE accounts ADD COLUMN note TEXT", "Alter"),
        ("TRUNCATE TABLE accounts", "TRUNCATE"),
    ],
)
def test_validate_rejects_dangerous_queries(
    sql_validator: SQLValidator,
    sql: str,
    keyword: str,
) -> None:
    result = sql_validator.validate(sql)

    assert result.is_valid is False
    assert result.errors
    assert any(keyword in error for error in result.errors)


def test_validate_rejects_multiple_statements(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("SELECT 1; SELECT 2;")

    assert result.is_valid is False
    assert result.errors == ["Ожидается ровно один SQL-запрос"]


def test_validate_rejects_empty_string(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("   ")

    assert result.is_valid is False
    assert result.errors == ["Пустой SQL-запрос"]


def test_validate_adds_default_limit_when_missing(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("SELECT * FROM stack.accounts")

    assert result.is_valid is True
    assert result.fixed_sql == "SELECT * FROM stack.accounts\nLIMIT 100"
    assert "Добавлен LIMIT 100" in result.warnings


def test_validate_does_not_duplicate_existing_limit(sql_validator: SQLValidator) -> None:
    result = sql_validator.validate("SELECT * FROM stack.accounts LIMIT 5")

    assert result.is_valid is True
    assert result.fixed_sql == "SELECT * FROM stack.accounts LIMIT 5"
    assert not any("Добавлен LIMIT" in warning for warning in result.warnings)


def test_extract_sql_from_sql_markdown_block(sql_validator: SQLValidator) -> None:
    response = """```sql
SELECT * FROM stack.accounts
LIMIT 5
```"""

    assert sql_validator.extract_sql(response) == "SELECT * FROM stack.accounts\nLIMIT 5"


def test_extract_sql_from_text_around_sql_block(sql_validator: SQLValidator) -> None:
    response = """
    Вот запрос:

    ```sql
    SELECT id FROM stack.accounts
    ```

    Он вернёт идентификаторы.
    """

    assert sql_validator.extract_sql(response) == "SELECT id FROM stack.accounts"


def test_extract_sql_without_markdown_returns_select_as_is(
    sql_validator: SQLValidator,
) -> None:
    response = "SELECT id FROM stack.accounts LIMIT 1"

    assert sql_validator.extract_sql(response) == response


def test_validate_warns_about_unqualified_cyrillic_table(
    sql_validator: SQLValidator,
) -> None:
    result = sql_validator.validate('SELECT * FROM "Лицевые счета"')

    assert result.is_valid is True
    assert any(
        'Кириллические таблицы без явной схемы: "Лицевые счета"' == warning
        for warning in result.warnings
    )


def test_validate_does_not_warn_about_qualified_cyrillic_table(
    sql_validator: SQLValidator,
) -> None:
    result = sql_validator.validate('SELECT * FROM stack."Лицевые счета"')

    assert result.is_valid is True
    assert not any("Кириллические таблицы без явной схемы" in warning for warning in result.warnings)
