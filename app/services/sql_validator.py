from __future__ import annotations

import logging
import re
from typing import Any

import sqlglot
from sqlglot import exp as sqlglot_exp

from app.core.models import ValidationResult

logger = logging.getLogger(__name__)


class SQLValidator:
    FORBIDDEN_TYPES: set[type[Any]] = {
        sqlglot_exp.Insert,
        sqlglot_exp.Update,
        sqlglot_exp.Delete,
        sqlglot_exp.Drop,
        sqlglot_exp.Create,
        sqlglot_exp.Alter,
        sqlglot_exp.Command,
    }
    ALLOWED_TYPES: set[type[Any]] = {
        sqlglot_exp.Select,
        sqlglot_exp.Union,
        sqlglot_exp.Intersect,
        sqlglot_exp.Except,
    }
    FORBIDDEN_PATTERNS: list[str] = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "ALTER ",
        "TRUNCATE ",
        "CREATE ",
        "GRANT ",
        "REVOKE ",
    ]

    _CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
    _STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
    _SQL_BLOCK_RE = re.compile(r"```sql\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
    _GENERIC_BLOCK_RE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)
    _SELECT_RE = re.compile(r"(SELECT\b.*?)(?:;|$)", re.DOTALL | re.IGNORECASE)

    def __init__(self, default_limit: int = 100):
        self._default_limit = default_limit

    def validate(self, sql: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            sql_clean = self._strip_code_fences(sql).rstrip(";").strip()
            if not sql_clean:
                return ValidationResult(
                    is_valid=False,
                    original_sql=sql,
                    errors=["Пустой SQL-запрос"],
                )

            try:
                expressions = sqlglot.parse(sql_clean, dialect="postgres")
            except sqlglot.errors.ParseError as parse_error:
                fixed = self._try_fix_cyrillic_quotes(sql_clean)
                if fixed != sql_clean:
                    try:
                        expressions = sqlglot.parse(fixed, dialect="postgres")
                        sql_clean = fixed
                        warnings.append(
                            "Автоматически добавлены кавычки к кириллическим идентификаторам"
                        )
                    except sqlglot.errors.ParseError as fixed_error:
                        return ValidationResult(
                            is_valid=False,
                            original_sql=sql,
                            errors=[f"Синтаксическая ошибка SQL: {fixed_error}"],
                            warnings=warnings,
                        )
                else:
                    return ValidationResult(
                        is_valid=False,
                        original_sql=sql,
                        errors=[f"Синтаксическая ошибка SQL: {parse_error}"],
                    )

            if len(expressions) != 1:
                errors.append("Ожидается ровно один SQL-запрос")
                return ValidationResult(
                    is_valid=False,
                    original_sql=sql,
                    fixed_sql=sql_clean,
                    errors=errors,
                    warnings=warnings,
                )

            expr = expressions[0]
            if expr is None:
                errors.append("Не удалось распознать SQL-запрос")
                return ValidationResult(
                    is_valid=False,
                    original_sql=sql,
                    fixed_sql=sql_clean,
                    errors=errors,
                    warnings=warnings,
                )

            if isinstance(expr, tuple(self.ALLOWED_TYPES)):
                pass
            elif isinstance(expr, tuple(self.FORBIDDEN_TYPES)):
                errors.append(
                    f"Запрещённый тип запроса: {type(expr).__name__}. Разрешены только SELECT."
                )
                return ValidationResult(
                    is_valid=False,
                    original_sql=sql,
                    fixed_sql=sql_clean,
                    errors=errors,
                    warnings=warnings,
                )
            else:
                errors.append(f"Нераспознанный тип запроса: {type(expr).__name__}")
                return ValidationResult(
                    is_valid=False,
                    original_sql=sql,
                    fixed_sql=sql_clean,
                    errors=errors,
                    warnings=warnings,
                )

            for node in expr.walk():
                if isinstance(node, tuple(self.FORBIDDEN_TYPES)):
                    errors.append("Обнаружена запрещённая подоперация внутри запроса")
                    break

            upper_sql = self._strip_string_literals(sql_clean).upper()
            for pattern in self.FORBIDDEN_PATTERNS:
                if pattern in upper_sql:
                    errors.append(f"Обнаружен запрещённый оператор: {pattern.strip()}")

            has_limit = expr.find(sqlglot_exp.Limit) is not None or "LIMIT" in sql_clean.upper()
            if not has_limit:
                sql_clean = sql_clean + f"\nLIMIT {self._default_limit}"
                warnings.append(f"Добавлен LIMIT {self._default_limit}")

            for node in expr.walk():
                if isinstance(node, sqlglot_exp.Identifier):
                    name = node.name
                    quoted = bool(node.args.get("quoted"))
                    if name and self._contains_cyrillic(name) and not quoted:
                        warnings.append(f"Идентификатор без кавычек: {name}")
                elif isinstance(node, sqlglot_exp.Table):
                    name = node.name
                    identifier = node.this if isinstance(node.this, sqlglot_exp.Identifier) else None
                    quoted = bool(identifier and identifier.args.get("quoted"))
                    if name and self._contains_cyrillic(name) and not quoted:
                        warnings.append(f"Идентификатор без кавычек: {name}")

            return ValidationResult(
                is_valid=len(errors) == 0,
                original_sql=sql,
                fixed_sql=sql_clean,
                errors=errors,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.exception("Unexpected SQL validation failure")
            return ValidationResult(
                is_valid=False,
                original_sql=sql,
                errors=[f"Ошибка валидации SQL: {exc}"],
            )

    def _try_fix_cyrillic_quotes(self, sql: str) -> str:
        try:
            return self._apply_outside_string_literals(sql, self._fix_cyrillic_segment)
        except Exception:  # pragma: no cover - best effort fixer
            logger.exception("Failed to auto-fix Cyrillic identifiers")
            return sql

    def extract_sql(self, llm_response: str) -> str | None:
        try:
            for pattern in (self._SQL_BLOCK_RE, self._GENERIC_BLOCK_RE, self._SELECT_RE):
                match = pattern.search(llm_response)
                if not match:
                    continue
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
            return None
        except Exception:  # pragma: no cover - defensive safety net
            logger.exception("Failed to extract SQL from LLM response")
            return None

    def _strip_code_fences(self, sql: str) -> str:
        sql_clean = sql.strip()
        if sql_clean.lower().startswith("```sql"):
            parts = sql_clean.splitlines()
            sql_clean = "\n".join(parts[1:]) if len(parts) > 1 else ""
        elif sql_clean.startswith("```"):
            parts = sql_clean.splitlines()
            sql_clean = "\n".join(parts[1:]) if len(parts) > 1 else ""

        sql_clean = sql_clean.strip()
        if sql_clean.endswith("```"):
            parts = sql_clean.splitlines()
            sql_clean = "\n".join(parts[:-1]) if len(parts) > 1 else ""
        return sql_clean.strip()

    def _strip_string_literals(self, sql: str) -> str:
        return self._STRING_LITERAL_RE.sub("''", sql)

    def _contains_cyrillic(self, value: str) -> bool:
        return bool(self._CYRILLIC_RE.search(value))

    def _apply_outside_string_literals(self, sql: str, fixer: Any) -> str:
        parts = re.split(r"('(?:''|[^'])*')", sql)
        fixed_parts = [
            part if index % 2 else fixer(part)
            for index, part in enumerate(parts)
        ]
        return "".join(fixed_parts)

    def _fix_cyrillic_segment(self, segment: str) -> str:
        fixed = segment

        table_pattern = re.compile(
            r'(?i)\b(FROM|JOIN)\s+(?!")(?P<schema>[A-Za-z_][\w$]*)\.(?P<name>[а-яА-ЯёЁ][а-яА-ЯёЁ0-9 _-]*)'
        )
        fixed = table_pattern.sub(
            lambda match: f'{match.group(1)} {match.group("schema")}."{match.group("name").strip()}"',
            fixed,
        )

        standalone_table_pattern = re.compile(
            r'(?i)\b(FROM|JOIN)\s+(?!")(?P<name>[а-яА-ЯёЁ][а-яА-ЯёЁ0-9 _-]*)(?=\s|,|\)|$)'
        )
        fixed = standalone_table_pattern.sub(
            lambda match: f'{match.group(1)} "{match.group("name").strip()}"',
            fixed,
        )

        dotted_identifier_pattern = re.compile(
            r'(?<!")\.(?P<name>[а-яА-ЯёЁ][а-яА-ЯёЁ0-9 _-]*)(?=\s|,|\)|=|<|>|!|\+|-|\*|/|$)'
        )
        fixed = dotted_identifier_pattern.sub(
            lambda match: f'."{match.group("name").strip()}"',
            fixed,
        )

        keyword_identifier_pattern = re.compile(
            r'(?i)\b(SELECT|WHERE|AND|OR|ON|GROUP BY|ORDER BY|HAVING|BY|AS|,)\s+(?!")(?P<name>[а-яА-ЯёЁ][а-яА-ЯёЁ0-9 _-]*)(?=\s|,|\)|=|<|>|!|\+|-|\*|/|$)'
        )
        fixed = keyword_identifier_pattern.sub(
            lambda match: f'{match.group(1)} "{match.group("name").strip()}"',
            fixed,
        )

        return fixed
