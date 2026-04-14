"""HTTP client for Ollama LLM and embedding APIs."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async Ollama client implementing LLM and embedding operations."""

    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url
        self._llm_model = settings.llm_model
        self._embed_model = settings.embed_model
        self._llm_timeout = settings.llm_timeout
        self._embed_timeout = settings.embed_timeout

        self._llm_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._llm_timeout,
        )
        self._embed_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._embed_timeout,
        )

    async def generate(self, prompt: str, system: str) -> str:
        """Generate text from Ollama chat API."""

        logger.debug(
            "Generating text with prompt_length=%s system_length=%s",
            len(prompt),
            len(system),
        )
        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            },
        }
        started_at = time.perf_counter()

        try:
            response = await self._request_with_retry(
                client=self._llm_client,
                method="POST",
                url="/api/chat",
                json_data=payload,
            )
            response_json = response.json()
            content = response_json["message"]["content"]
            if not content:
                raise LLMError("LLM вернула пустой ответ")
        except httpx.TimeoutException as exc:
            message = f"Ollama таймаут ({self._llm_timeout}с)"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except httpx.ConnectError as exc:
            message = f"Ollama недоступна: {self._base_url}"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except httpx.HTTPStatusError as exc:
            message = self._build_llm_http_error(exc, self._llm_model)
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except LLMError as exc:
            logger.error(str(exc), exc_info=exc)
            raise
        except (KeyError, TypeError, ValueError) as exc:
            message = "Некорректный ответ от Ollama LLM"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc

        elapsed = time.perf_counter() - started_at
        logger.info(
            "Ollama generate completed model=%s duration=%.3fs content_length=%s",
            self._llm_model,
            elapsed,
            len(content),
        )
        return content

    async def embed(self, text: str) -> list[float]:
        """Fetch embedding vector from Ollama embed API."""

        logger.debug("Generating embedding for text_length=%s", len(text))
        payload = {
            "model": self._embed_model,
            "input": text,
        }

        try:
            response = await self._request_with_retry(
                client=self._embed_client,
                method="POST",
                url="/api/embed",
                json_data=payload,
            )
            response_json = response.json()
            embedding = response_json["embeddings"][0]
            if not embedding:
                raise LLMError("Embedding вернула пустой вектор")
        except httpx.TimeoutException as exc:
            message = f"Ollama embedding таймаут ({self._embed_timeout}с)"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except httpx.ConnectError as exc:
            message = f"Ollama embedding недоступна: {self._base_url}"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except httpx.HTTPStatusError as exc:
            message = self._build_embedding_http_error(exc, self._embed_model)
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc
        except LLMError as exc:
            logger.error(str(exc), exc_info=exc)
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            message = "Некорректный ответ от Ollama embedding"
            logger.error(message, exc_info=exc)
            raise LLMError(message) from exc

        logger.debug("Received embedding vector with dimension=%s", len(embedding))
        return embedding

    async def is_available(self) -> bool:
        """Check whether Ollama responds to a basic availability probe."""

        try:
            response = await self._llm_client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        """Close underlying HTTP clients."""

        await self._llm_client.aclose()
        await self._embed_client.aclose()

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        json_data: dict[str, Any],
        max_retries: int = 2,
    ) -> httpx.Response:
        """Send request with retries for Ollama 5xx responses."""

        last_status_code: int | None = None

        for attempt in range(1, max_retries + 2):
            response = await client.request(method=method, url=url, json=json_data)
            if response.status_code < 500:
                response.raise_for_status()
                return response

            last_status_code = response.status_code
            if attempt > max_retries:
                break

            await asyncio.sleep(attempt)

        raise LLMError(
            "Ollama вернула серверную ошибку "
            f"({last_status_code}) после {max_retries + 1} попыток"
        )

    @staticmethod
    def _build_llm_http_error(exc: httpx.HTTPStatusError, model: str) -> str:
        if exc.response.status_code == 404:
            return f"Модель {model} не найдена в Ollama"
        return f"Ollama LLM HTTP ошибка: {exc.response.status_code}"

    @staticmethod
    def _build_embedding_http_error(exc: httpx.HTTPStatusError, model: str) -> str:
        if exc.response.status_code == 404:
            return f"Модель {model} не найдена в Ollama"
        return f"Ollama embedding HTTP ошибка: {exc.response.status_code}"
