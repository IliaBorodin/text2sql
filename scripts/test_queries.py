from __future__ import annotations

import asyncio
import json
import logging
import site
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def bootstrap_project_root() -> Path:
    """Make the project package importable for standalone script execution."""

    project_root = Path(__file__).resolve().parents[1]
    site.addsitedir(str(project_root))
    return project_root


PROJECT_ROOT = bootstrap_project_root()

from app.core.config import Settings
from app.core.models import QueryResponse
from app.infrastructure.chroma_client import ChromaClient
from app.infrastructure.ollama_client import OllamaClient
from app.infrastructure.postgres_client import PostgresClient
from app.infrastructure.xdic.parser import XdicParser
from app.services.pipeline import Pipeline
from app.services.prompt_builder import PromptBuilder
from app.services.schema_retrieval import SchemaRetrievalService
from app.services.sql_executor import SQLExecutor
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("app").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


REPORT_WIDTH = 56
SQL_PREVIEW_LENGTH = 120


@dataclass
class TestCase:
    id: int
    question: str
    expected_tables: list[str]
    expected_patterns: list[str]
    description: str


@dataclass
class TestResult:
    test_case: TestCase
    success: bool
    sql_generated: bool
    sql_executed: bool
    tables_found: bool
    patterns_found: dict[str, bool]
    sql: str | None
    row_count: int
    error: str | None
    time_s: float
    pipeline_response: QueryResponse | None


TEST_CASES = [
    TestCase(
        id=1,
        question="Покажи общую площадь для лицевых счетов",
        expected_tables=["Лицевые счета", "Свойства", "Виды параметров"],
        expected_patterns=['"Тип" = 5', "ОБЩПЛОЩАДЬ", '"ДатНач"'],
        description="3-table JOIN: ЛС + Свойства + Виды параметров + фильтр даты",
    ),
    TestCase(
        id=2,
        question="Покажи все действующие параметры лицевых счетов на прошлый месяц",
        expected_tables=["Лицевые счета", "Свойства", "Виды параметров"],
        expected_patterns=['"Тип" = 5'],
        description="Вычисление даты прошлого месяца",
    ),
    TestCase(
        id=3,
        question="Покажи действующие приборы учёта на 01.09.2022",
        expected_tables=["Список объектов"],
        expected_patterns=["2022-09-01", '"ДатНач"'],
        description="Приборы учёта с фильтром действующий на дату",
    ),
    TestCase(
        id=4,
        question="Покажи реестр лицевых счетов с действующими приборами учёта",
        expected_tables=["Лицевые счета", "Список объектов"],
        expected_patterns=['"Тип" = 5'],
        description="DISTINCT реестр ЛС с ПУ",
    ),
    TestCase(
        id=5,
        question="Сколько приборов учёта установлено с 01.01.2020",
        expected_tables=["Список объектов"],
        expected_patterns=["COUNT", "2020"],
        description="COUNT с фильтром по дате установки",
    ),
    TestCase(
        id=6,
        question="Сколько лицевых счетов в базе?",
        expected_tables=["Лицевые счета"],
        expected_patterns=["COUNT", '"Тип" = 5'],
        description="Простой COUNT с пониманием Тип=5",
    ),
    TestCase(
        id=7,
        question="Покажи все улицы",
        expected_tables=["Лицевые счета"],
        expected_patterns=['"Тип" = 2'],
        description="Понимание что улицы - Тип=2 в Лицевые счета",
    ),
    TestCase(
        id=8,
        question="Покажи закрытые лицевые счета",
        expected_tables=["Лицевые счета"],
        expected_patterns=['"Тип" = 5', '"ДатаЗакрытия"'],
        description="Закрытые ЛС = ДатаЗакрытия IS NOT NULL",
    ),
]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


def supports_color() -> bool:
    return sys.stdout.isatty() and not bool(getattr(sys.stdout, "closed", False))


def colorize(text: str, color: str) -> str:
    if not supports_color():
        return text
    return f"{color}{text}{Colors.RESET}"


def success_icon(ok: bool) -> str:
    return colorize("✅", Colors.GREEN) if ok else colorize("❌", Colors.RED)


def warning_icon() -> str:
    return colorize("⚠️", Colors.YELLOW)


def truncate_sql(sql: str | None, max_length: int = SQL_PREVIEW_LENGTH) -> str:
    if not sql:
        return "<не сгенерирован>"
    compact = " ".join(sql.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


def format_duration(total_seconds: float) -> str:
    seconds = int(round(total_seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def serialize_query_response(response: QueryResponse | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return response.model_dump(mode="json")


async def setup_pipeline() -> tuple[Pipeline, list[Callable[[], Any]]]:
    settings = Settings()

    ollama = OllamaClient(settings)
    db = PostgresClient(settings)
    await db.connect()
    chroma = ChromaClient(settings)

    xdic = XdicParser(settings.xdic_path)
    xdic.parse()

    schema_retrieval = SchemaRetrievalService(ollama, chroma, xdic)
    prompt_builder = PromptBuilder(settings)
    validator = SQLValidator(settings.sql_default_limit)
    generator = SQLGenerator(ollama, validator)
    executor = SQLExecutor(db)

    pipeline = Pipeline(
        schema_retrieval,
        prompt_builder,
        generator,
        validator,
        executor,
    )

    return pipeline, [ollama.close, db.disconnect]


async def run_test(pipeline: Pipeline, test_case: TestCase) -> TestResult:
    start = time.perf_counter()

    try:
        response = await pipeline.execute(test_case.question)
        elapsed = time.perf_counter() - start

        sql_generated = response.sql is not None
        sql_executed = response.success and response.result is not None
        tables_found = (
            all(table in response.sql for table in test_case.expected_tables)
            if response.sql
            else False
        )
        patterns_found = {
            pattern: (pattern in response.sql if response.sql else False)
            for pattern in test_case.expected_patterns
        }

        return TestResult(
            test_case=test_case,
            success=sql_generated and sql_executed,
            sql_generated=sql_generated,
            sql_executed=sql_executed,
            tables_found=tables_found,
            patterns_found=patterns_found,
            sql=response.sql,
            row_count=response.result.row_count if response.result else 0,
            error=response.error,
            time_s=elapsed,
            pipeline_response=response,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception("Тест %s завершился с необработанным исключением", test_case.id)
        return TestResult(
            test_case=test_case,
            success=False,
            sql_generated=False,
            sql_executed=False,
            tables_found=False,
            patterns_found={pattern: False for pattern in test_case.expected_patterns},
            sql=None,
            row_count=0,
            error=str(exc) or exc.__class__.__name__,
            time_s=elapsed,
            pipeline_response=None,
        )


def print_report(results: list[TestResult]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    executed = sum(1 for result in results if result.success)
    tables_ok = sum(1 for result in results if result.tables_found)
    pattern_hits = sum(
        1 for result in results for found in result.patterns_found.values() if found
    )
    pattern_total = sum(len(result.patterns_found) for result in results)
    total_time_s = sum(result.time_s for result in results)

    print("\n" + "═" * REPORT_WIDTH)
    print("Text2SQL ЖКХ - Integration Test Report")
    print(now)
    print("═" * REPORT_WIDTH)

    for index, result in enumerate(results, start=1):
        checks_ok = result.tables_found and all(result.patterns_found.values())
        status = success_icon(result.success)
        if result.success and not checks_ok:
            status = warning_icon()

        print(f"\n[{index}/{total}] {status} {result.test_case.description}")
        print(f"SQL:    {truncate_sql(result.sql)}")

        tables_line = " ".join(
            f"{success_icon(table in (result.sql or ''))} {table}"
            for table in result.test_case.expected_tables
        )
        print(f"Tables: {tables_line}")

        checks_line = " ".join(
            f"{success_icon(found)} {pattern}"
            for pattern, found in result.patterns_found.items()
        )
        if checks_line:
            print(f"Checks: {checks_line}")

        if result.error:
            print(f"Error:  {result.error}")

        print(f"Result: {result.row_count} rows | {result.time_s:.1f}s")

    print("\n" + "═" * REPORT_WIDTH)
    print("ИТОГО")
    print("═" * REPORT_WIDTH)
    print(
        f"Выполнено:  {executed}/{total} ({executed / total:.0%})"
        "    - SQL сгенерирован и выполнен"
    )
    print(
        f"Таблицы:    {tables_ok}/{total} ({tables_ok / total:.0%})"
        "     - правильные таблицы в SQL"
    )
    print(
        f"Паттерны:   {pattern_hits}/{pattern_total} ({(pattern_hits / pattern_total) if pattern_total else 0:.0%})"
        "   - ожидаемые паттерны найдены"
    )
    print(f"Время:      {format_duration(total_time_s)}        - общее время")
    print("═" * REPORT_WIDTH)


def save_report(results: list[TestResult]) -> Path:
    Path("logs").mkdir(exist_ok=True)

    timestamp = datetime.now()
    total = len(results)
    passed = sum(1 for result in results if result.success)
    failed = total - passed
    tables_ok = sum(1 for result in results if result.tables_found)
    pattern_hits = sum(
        1 for result in results for found in result.patterns_found.values() if found
    )
    pattern_total = sum(len(result.patterns_found) for result in results)

    payload = {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "execution_rate": passed / total if total else 0.0,
        "tables_accuracy": tables_ok / total if total else 0.0,
        "patterns_accuracy": pattern_hits / pattern_total if pattern_total else 0.0,
        "total_time_s": round(sum(result.time_s for result in results), 3),
        "results": [],
    }

    for result in results:
        response_data = serialize_query_response(result.pipeline_response)
        timings = (response_data or {}).get("timings") or {}
        payload["results"].append(
            {
                "id": result.test_case.id,
                "question": result.test_case.question,
                "description": result.test_case.description,
                "success": result.success,
                "sql_generated": result.sql_generated,
                "sql_executed": result.sql_executed,
                "tables_found": result.tables_found,
                "patterns": dict(result.patterns_found),
                "sql": result.sql,
                "row_count": result.row_count,
                "error": result.error,
                "time_s": round(result.time_s, 3),
                "timings": timings,
            }
        )

    path = Path("logs") / f"test_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def main() -> None:
    print("═" * REPORT_WIDTH)
    print("Text2SQL ЖКХ - Integration Test Runner")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═" * REPORT_WIDTH)

    pipeline: Pipeline | None = None
    cleanup_fns: list[Callable[[], Any]] = []
    results: list[TestResult] = []
    setup_failed = False

    try:
        pipeline, cleanup_fns = await setup_pipeline()

        for index, test_case in enumerate(TEST_CASES, start=1):
            print(f"\n[{index}/{len(TEST_CASES)}] Выполняю: {test_case.question[:60]}...")
            result = await run_test(pipeline, test_case)
            results.append(result)
            status = success_icon(result.success)
            print(
                f"  {status} {result.time_s:.1f}s | rows={result.row_count} | "
                f"sql={'да' if result.sql_generated else 'нет'}"
            )

        print_report(results)

        report_path = save_report(results)
        print(f"\nОтчёт сохранён: {report_path}")
    except Exception:
        setup_failed = True
        logger.exception("Не удалось выполнить интеграционные тесты")
        print(f"\n{success_icon(False)} Ошибка инициализации pipeline. См. логи выше.")
    finally:
        for cleanup in cleanup_fns:
            try:
                await cleanup()
            except Exception:
                logger.exception("Ошибка при освобождении ресурсов")

    if setup_failed:
        sys.exit(1)

    passed = sum(1 for result in results if result.success)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
