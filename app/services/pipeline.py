from __future__ import annotations

import logging
import time

from app.core.exceptions import DatabaseError, LLMError, Text2SQLError
from app.core.models import PipelineTimings, QueryResponse
from app.services.prompt_builder import PromptBuilder
from app.services.schema_retrieval import SchemaRetrievalService
from app.services.sql_executor import SQLExecutor
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        schema_retrieval: SchemaRetrievalService,
        prompt_builder: PromptBuilder,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        max_retries: int = 1,
    ) -> None:
        self._schema_retrieval = schema_retrieval
        self._prompt_builder = prompt_builder
        self._sql_generator = sql_generator
        self._sql_validator = sql_validator
        self._sql_executor = sql_executor
        self._max_retries = max(0, max_retries)

    async def execute(self, question: str) -> QueryResponse:
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        tables_used: list[str] = []
        error: str | None = None

        logger.info("Pipeline started for question: %s", question)

        logger.info("Phase 1/7: schema retrieval started")
        try:
            start = time.perf_counter()
            tables = await self._schema_retrieval.retrieve(question)
            timings["schema_retrieval_ms"] = self._elapsed_ms(start)
            tables_used = [table.name for table in tables]
            logger.info("Найдено %d таблиц: %s", len(tables), tables_used)
        except Text2SQLError as exc:
            logger.error("Schema retrieval failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка поиска таблиц: {exc}",
                timings,
                total_start,
            )
        except Exception as exc:
            logger.error("Schema retrieval failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка поиска таблиц: {exc}",
                timings,
                total_start,
            )

        logger.info("Phase 2/7: prompt build started")
        try:
            start = time.perf_counter()
            system_prompt, user_prompt = self._prompt_builder.build(question, tables)
            timings["prompt_build_ms"] = self._elapsed_ms(start)
            logger.debug(
                "Промт собран: system=%d, user=%d символов",
                len(system_prompt),
                len(user_prompt),
            )
        except Text2SQLError as exc:
            logger.error("Prompt build failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка сборки промта: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
        except Exception as exc:
            logger.error("Prompt build failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка сборки промта: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )

        logger.info("Phase 3/7: SQL generation started")
        try:
            start = time.perf_counter()
            generation = await self._sql_generator.generate(system_prompt, user_prompt)
            timings["llm_generation_ms"] = self._elapsed_ms(start)
            logger.info(
                "LLM ответил: sql=%s",
                "найден" if generation.extracted_sql else "НЕ найден",
            )
        except LLMError as exc:
            logger.error("SQL generation failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка генерации SQL: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
        except Exception as exc:
            logger.error("SQL generation failed: %s", exc)
            return self._error_response(
                question,
                f"Ошибка генерации SQL: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )

        if generation.extracted_sql is None:
            return self._error_response(
                question,
                "LLM не вернула SQL-запрос. Попробуйте переформулировать вопрос.",
                timings,
                total_start,
                tables_used=tables_used,
            )

        current_sql = generation.extracted_sql
        result = None

        logger.info("Phase 4/7: validation started")
        start = time.perf_counter()
        validation = self._sql_validator.validate(current_sql)
        timings["validation_ms"] = self._elapsed_ms(start)

        if validation.is_valid:
            current_sql = validation.fixed_sql or current_sql
            logger.info("Валидация OK. Warnings: %s", validation.warnings)
        else:
            error = "; ".join(validation.errors) or "Неизвестная ошибка валидации"
            logger.warning("Валидация FAILED: %s", validation.errors)

        if error is None:
            logger.info("Phase 5/7: execution started")
            start = time.perf_counter()
            try:
                result = await self._sql_executor.execute(current_sql)
                timings["execution_ms"] = self._elapsed_ms(start)
                logger.info(
                    "Выполнено: %d строк за %.0fмс",
                    result.row_count,
                    result.execution_time_ms,
                )
            except DatabaseError as exc:
                timings["execution_ms"] = self._elapsed_ms(start)
                error = exc.message
                logger.warning("Execution FAILED: %s", exc.message)
            except Exception as exc:
                timings["execution_ms"] = self._elapsed_ms(start)
                logger.error("Unexpected execution error: %s", exc)
                return self._error_response(
                    question,
                    f"Ошибка выполнения: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )

        retries_left = self._max_retries
        failed_sql = current_sql

        while error is not None:
            logger.info(
                "Phase 6/7: self-correction started, retries_left=%d",
                retries_left,
            )
            if retries_left <= 0:
                return self._error_response(
                    question,
                    f"SQL не прошёл проверку после {self._max_retries} попыток: {error}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )

            correction_prompt = (
                user_prompt
                + "\n\n"
                + self._build_correction_block(failed_sql, error)
            )

            logger.info("Self-correction: повторная генерация SQL")
            try:
                start = time.perf_counter()
                retry_generation = await self._sql_generator.generate(
                    system_prompt,
                    correction_prompt,
                )
                timings["llm_generation_ms"] = timings.get("llm_generation_ms", 0.0) + self._elapsed_ms(start)
            except Text2SQLError as exc:
                logger.error("Retry SQL generation failed: %s", exc)
                return self._error_response(
                    question,
                    f"Ошибка повторной генерации SQL: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )
            except Exception as exc:
                logger.error("Retry SQL generation failed: %s", exc)
                return self._error_response(
                    question,
                    f"Ошибка повторной генерации SQL: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )

            if retry_generation.extracted_sql is None:
                return self._error_response(
                    question,
                    "LLM не вернула SQL-запрос при исправлении. Попробуйте переформулировать вопрос.",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )

            logger.info("Self-correction: повторная валидация SQL")
            start = time.perf_counter()
            retry_validation = self._sql_validator.validate(retry_generation.extracted_sql)
            timings["validation_ms"] = timings.get("validation_ms", 0.0) + self._elapsed_ms(start)

            if not retry_validation.is_valid:
                retry_error = "; ".join(retry_validation.errors) or "Неизвестная ошибка валидации"
                logger.warning("Self-correction validation FAILED: %s", retry_validation.errors)
                current_sql = retry_validation.fixed_sql or retry_generation.extracted_sql
                retries_left -= 1
                failed_sql = current_sql
                error = retry_error
                continue

            current_sql = retry_validation.fixed_sql or retry_generation.extracted_sql

            logger.info("Self-correction: повторное выполнение SQL")
            start = time.perf_counter()
            try:
                result = await self._sql_executor.execute(current_sql)
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + self._elapsed_ms(start)
                logger.info("Self-correction: выполнено, %d строк", result.row_count)
                error = None
            except DatabaseError as exc:
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + self._elapsed_ms(start)
                logger.warning("Self-correction execution FAILED: %s", exc.message)
                retries_left -= 1
                failed_sql = current_sql
                error = exc.message
                continue
            except Exception as exc:
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + self._elapsed_ms(start)
                logger.error("Unexpected self-correction execution error: %s", exc)
                return self._error_response(
                    question,
                    f"Ошибка выполнения: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )

        logger.info("Phase 7/7: building successful response")
        total_ms = self._elapsed_ms(total_start)
        return QueryResponse(
            question=question,
            sql=current_sql,
            result=result,
            tables_used=tables_used,
            timings=PipelineTimings(
                schema_retrieval_ms=timings.get("schema_retrieval_ms", 0),
                prompt_build_ms=timings.get("prompt_build_ms", 0),
                llm_generation_ms=timings.get("llm_generation_ms", 0),
                validation_ms=timings.get("validation_ms", 0),
                execution_ms=timings.get("execution_ms", 0),
                total_ms=total_ms,
            ),
            error=None,
            success=True,
        )

    def _build_correction_block(self, failed_sql: str, error: str) -> str:
        return (
            "══════════════════════════════\n"
            "ИСПРАВЛЕНИЕ ОШИБКИ\n"
            "══════════════════════════════\n\n"
            "Предыдущая попытка:\n"
            "```sql\n"
            f"{failed_sql}\n"
            "```\n\n"
            f"Ошибка: {error}\n\n"
            "Исправь SQL-запрос с учётом этой ошибки. "
            "Верни исправленный запрос в блоке ```sql."
        )

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    @staticmethod
    def _error_response(
        question: str,
        error: str,
        timings: dict[str, float],
        total_start: float,
        tables_used: list[str] | None = None,
        sql: str | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            question=question,
            sql=sql,
            result=None,
            tables_used=tables_used or [],
            timings=PipelineTimings(
                schema_retrieval_ms=timings.get("schema_retrieval_ms", 0),
                prompt_build_ms=timings.get("prompt_build_ms", 0),
                llm_generation_ms=timings.get("llm_generation_ms", 0),
                validation_ms=timings.get("validation_ms", 0),
                execution_ms=timings.get("execution_ms", 0),
                total_ms=round((time.perf_counter() - total_start) * 1000, 2),
            ),
            error=error,
            success=False,
        )


__all__ = ["Pipeline"]
