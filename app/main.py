from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.core.config import Settings
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
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings | None = None
    ollama: OllamaClient | None = None
    db: PostgresClient | None = None

    try:
        settings = Settings()
        logging.getLogger().setLevel(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        )

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

        app.state.settings = settings
        app.state.pipeline = pipeline
        app.state.ollama = ollama
        app.state.db = db
        app.state.chroma = chroma

        logger.info(
            "Text2SQL ЖКХ запущен. Таблиц в ChromaDB: %s. Модель: %s",
            chroma.get_table_count(),
            settings.llm_model,
        )

        yield
    except Exception:
        logger.critical("Ошибка инициализации приложения Text2SQL ЖКХ", exc_info=True)
        if ollama is not None:
            await ollama.close()
        if db is not None:
            await db.disconnect()
        raise
    else:
        if ollama is not None:
            await ollama.close()
        if db is not None:
            await db.disconnect()
        logger.info("Text2SQL ЖКХ остановлен")


app = FastAPI(
    title="Text2SQL ЖКХ",
    description="API для преобразования вопросов на русском языке в SQL-запросы к БД ЖКХ",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api")
app.include_router(health_router, prefix="/api")


@app.get("/")
async def root():
    return {"service": "Text2SQL ЖКХ", "version": "0.1.0", "docs": "/docs"}
