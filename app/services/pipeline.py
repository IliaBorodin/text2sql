from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from app.core.exceptions import DatabaseError, LLMError, Text2SQLError
from app.core.models import PipelineTimings, QueryProgressEvent, QueryResponse
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
        final_response: QueryResponse | None = None

        async for event in self.execute_with_progress(question):
            if event.response is not None:
                final_response = event.response

        if final_response is None:
            raise RuntimeError("Pipeline finished without a final response")

        return final_response

    async def execute_with_progress(
        self,
        question: str,
    ) -> AsyncIterator[QueryProgressEvent]:
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        tables_used: list[str] = []

        logger.info("Pipeline started for question: %s", question)

        logger.info("Phase 1/7: schema retrieval started")
        try:
            start = time.perf_counter()
            tables = await self._schema_retrieval.retrieve(question)
            timings["schema_retrieval_ms"] = self._elapsed_ms(start)
            tables_used = [table.name for table in tables]
            logger.info("Найдено %d таблиц: %s", len(tables), tables_used)
            yield self._progress_event(
                phase="retrieval",
                message="Поиск таблиц завершён",
                data={
                    "tables_used": tables_used,
                    "table_count": len(tables_used),
                    "duration_ms": timings["schema_retrieval_ms"],
                },
            )
        except Text2SQLError as exc:
            logger.error("Schema retrieval failed: %s", exc)
            yield self._error_event(
                question,
                f"Ошибка поиска таблиц: {exc}",
                timings,
                total_start,
            )
            return
        except Exception as exc:
            logger.error("Schema retrieval failed: %s", exc)
            yield self._error_event(
                question,
                f"Ошибка поиска таблиц: {exc}",
                timings,
                total_start,
            )
            return

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
            yield self._error_event(
                question,
                f"Ошибка сборки промта: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
            return
        except Exception as exc:
            logger.error("Prompt build failed: %s", exc)
            yield self._error_event(
                question,
                f"Ошибка сборки промта: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
            return

        logger.info("Phase 3/7: SQL generation started")
        try:
            start = time.perf_counter()
            generation = await self._sql_generator.generate(system_prompt, user_prompt)
            timings["llm_generation_ms"] = self._elapsed_ms(start)
            logger.info(
                "LLM ответил: sql=%s",
                "найден" if generation.extracted_sql else "НЕ найден",
            )
            yield self._progress_event(
                phase="generation",
                message="Генерация SQL завершена",
                data={
                    "duration_ms": timings["llm_generation_ms"],
                    "sql_found": generation.extracted_sql is not None,
                    "sql": generation.extracted_sql,
                    "explanation": generation.explanation,
                },
            )
        except LLMError as exc:
            logger.error("SQL generation failed: %s", exc)
            yield self._error_event(
                question,
                f"Ошибка генерации SQL: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
            return
        except Exception as exc:
            logger.error("SQL generation failed: %s", exc)
            yield self._error_event(
                question,
                f"Ошибка генерации SQL: {exc}",
                timings,
                total_start,
                tables_used=tables_used,
            )
            return

        if generation.extracted_sql is None:
            yield self._error_event(
                question,
                "LLM не вернула SQL-запрос. Попробуйте переформулировать вопрос.",
                timings,
                total_start,
                tables_used=tables_used,
            )
            return

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
            logger.warning("Валидация FAILED: %s", validation.errors)

        error = (
            None
            if validation.is_valid
            else "; ".join(validation.errors) or "Неизвестная ошибка валидации"
        )
        yield self._progress_event(
            phase="validation",
            message="Валидация SQL завершена",
            data={
                "duration_ms": timings["validation_ms"],
                "is_valid": validation.is_valid,
                "sql": current_sql,
                "warnings": validation.warnings,
                "errors": validation.errors,
            },
        )

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
                yield self._progress_event(
                    phase="execution",
                    message="SQL выполнен успешно",
                    data={
                        "duration_ms": timings["execution_ms"],
                        "row_count": result.row_count,
                        "execution_time_ms": result.execution_time_ms,
                        "sql": current_sql,
                    },
                )
            except DatabaseError as exc:
                timings["execution_ms"] = self._elapsed_ms(start)
                error = exc.message
                logger.warning("Execution FAILED: %s", exc.message)
                yield self._progress_event(
                    phase="execution",
                    message="Ошибка выполнения SQL, запускаем самокоррекцию",
                    data={
                        "duration_ms": timings["execution_ms"],
                        "sql": current_sql,
                        "error": exc.message,
                        "attempt": 0,
                    },
                )
            except Exception as exc:
                timings["execution_ms"] = self._elapsed_ms(start)
                logger.error("Unexpected execution error: %s", exc)
                yield self._error_event(
                    question,
                    f"Ошибка выполнения: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )
                return

        retries_left = self._max_retries
        failed_sql = current_sql

        while error is not None:
            logger.info(
                "Phase 6/7: self-correction started, retries_left=%d",
                retries_left,
            )
            if retries_left <= 0:
                yield self._error_event(
                    question,
                    f"SQL не прошёл проверку после {self._max_retries} попыток: {error}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )
                return

            attempt = self._max_retries - retries_left + 1
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
                timings["llm_generation_ms"] = timings.get(
                    "llm_generation_ms",
                    0.0,
                ) + self._elapsed_ms(start)
                yield self._progress_event(
                    phase="generation",
                    message="Повторная генерация SQL завершена",
                    data={
                        "attempt": attempt,
                        "duration_ms": timings["llm_generation_ms"],
                        "sql_found": retry_generation.extracted_sql is not None,
                        "sql": retry_generation.extracted_sql,
                    },
                )
            except Text2SQLError as exc:
                logger.error("Retry SQL generation failed: %s", exc)
                yield self._error_event(
                    question,
                    f"Ошибка повторной генерации SQL: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )
                return
            except Exception as exc:
                logger.error("Retry SQL generation failed: %s", exc)
                yield self._error_event(
                    question,
                    f"Ошибка повторной генерации SQL: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )
                return

            if retry_generation.extracted_sql is None:
                yield self._error_event(
                    question,
                    "LLM не вернула SQL-запрос при исправлении. Попробуйте переформулировать вопрос.",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=failed_sql,
                )
                return

            logger.info("Self-correction: повторная валидация SQL")
            start = time.perf_counter()
            retry_validation = self._sql_validator.validate(
                retry_generation.extracted_sql,
            )
            timings["validation_ms"] = timings.get("validation_ms", 0.0) + (
                self._elapsed_ms(start)
            )
            yield self._progress_event(
                phase="validation",
                message="Повторная валидация SQL завершена",
                data={
                    "attempt": attempt,
                    "duration_ms": timings["validation_ms"],
                    "is_valid": retry_validation.is_valid,
                    "sql": retry_validation.fixed_sql
                    or retry_generation.extracted_sql,
                    "warnings": retry_validation.warnings,
                    "errors": retry_validation.errors,
                },
            )

            if not retry_validation.is_valid:
                retry_error = (
                    "; ".join(retry_validation.errors)
                    or "Неизвестная ошибка валидации"
                )
                logger.warning(
                    "Self-correction validation FAILED: %s",
                    retry_validation.errors,
                )
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
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + (
                    self._elapsed_ms(start)
                )
                logger.info("Self-correction: выполнено, %d строк", result.row_count)
                yield self._progress_event(
                    phase="execution",
                    message="Исправленный SQL выполнен успешно",
                    data={
                        "attempt": attempt,
                        "duration_ms": timings["execution_ms"],
                        "row_count": result.row_count,
                        "execution_time_ms": result.execution_time_ms,
                        "sql": current_sql,
                    },
                )
                error = None
            except DatabaseError as exc:
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + (
                    self._elapsed_ms(start)
                )
                logger.warning("Self-correction execution FAILED: %s", exc.message)
                yield self._progress_event(
                    phase="execution",
                    message="Исправленный SQL снова завершился ошибкой",
                    data={
                        "attempt": attempt,
                        "duration_ms": timings["execution_ms"],
                        "sql": current_sql,
                        "error": exc.message,
                    },
                )
                retries_left -= 1
                failed_sql = current_sql
                error = exc.message
                continue
            except Exception as exc:
                timings["execution_ms"] = timings.get("execution_ms", 0.0) + (
                    self._elapsed_ms(start)
                )
                logger.error("Unexpected self-correction execution error: %s", exc)
                yield self._error_event(
                    question,
                    f"Ошибка выполнения: {exc}",
                    timings,
                    total_start,
                    tables_used=tables_used,
                    sql=current_sql,
                )
                return

        logger.info("Phase 7/7: building successful response")
        response = QueryResponse(
            question=question,
            sql=current_sql,
            result=result,
            tables_used=tables_used,
            timings=self._build_timings(timings, total_start),
            error=None,
            success=True,
        )
        yield self._progress_event(
            phase="done",
            message="Пайплайн завершён успешно",
            data={
                "sql": current_sql,
                "row_count": result.row_count if result is not None else 0,
                "total_ms": response.timings.total_ms,
            },
            response=response,
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

    def _error_event(
        self,
        question: str,
        error: str,
        timings: dict[str, float],
        total_start: float,
        tables_used: list[str] | None = None,
        sql: str | None = None,
    ) -> QueryProgressEvent:
        response = QueryResponse(
            question=question,
            sql=sql,
            result=None,
            tables_used=tables_used or [],
            timings=self._build_timings(timings, total_start),
            error=error,
            success=False,
        )
        return self._progress_event(
            phase="error",
            message=error,
            data={
                "sql": sql,
                "tables_used": tables_used or [],
                "total_ms": response.timings.total_ms,
            },
            response=response,
        )

    def _build_timings(
        self,
        timings: dict[str, float],
        total_start: float,
    ) -> PipelineTimings:
        return PipelineTimings(
            schema_retrieval_ms=timings.get("schema_retrieval_ms", 0),
            prompt_build_ms=timings.get("prompt_build_ms", 0),
            llm_generation_ms=timings.get("llm_generation_ms", 0),
            validation_ms=timings.get("validation_ms", 0),
            execution_ms=timings.get("execution_ms", 0),
            total_ms=round((time.perf_counter() - total_start) * 1000, 2),
        )

    @staticmethod
    def _progress_event(
        phase: str,
        message: str,
        data: dict | None = None,
        response: QueryResponse | None = None,
    ) -> QueryProgressEvent:
        return QueryProgressEvent(
            phase=phase,
            message=message,
            data=data or {},
            response=response,
        )


__all__ = ["Pipeline"]
