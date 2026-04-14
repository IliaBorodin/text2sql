from __future__ import annotations

import logging
import re

from app.core.exceptions import LLMError
from app.core.interfaces import LLMClient
from app.core.models import GenerationResult
from app.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class SQLGenerator:
    def __init__(self, llm_client: LLMClient, sql_validator: SQLValidator):
        self._llm_client = llm_client
        self._sql_validator = sql_validator

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        try:
            raw_response = await self._llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
            )

            clean_response, explanation = self._strip_think_block(raw_response)
            extracted_sql = self._sql_validator.extract_sql(clean_response)

            if extracted_sql is None:
                logger.warning("SQL не найден в ответе LLM")
                logger.debug("raw_response: %s", raw_response[:500])
            elif explanation is None:
                sql_block_match = re.search(r"```sql", clean_response, re.IGNORECASE)
                if sql_block_match:
                    prefix = clean_response[: sql_block_match.start()].strip()
                    suffix_match = re.search(
                        r"```sql\s*\n?.*?```(.*)$",
                        clean_response,
                        re.DOTALL | re.IGNORECASE,
                    )
                    suffix = suffix_match.group(1).strip() if suffix_match else ""
                    parts = [part for part in (prefix, suffix) if part]
                    explanation = "\n\n".join(parts) if parts else None

            if extracted_sql is None:
                logger.info("SQL не найден в ответе")
            else:
                logger.info("SQL сгенерирован")

            logger.debug("raw_response length: %s", len(raw_response))
            logger.debug(
                "extracted_sql preview: %s",
                extracted_sql[:200] if extracted_sql is not None else None,
            )

            return GenerationResult(
                raw_response=raw_response,
                extracted_sql=extracted_sql,
                explanation=explanation,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ошибка генерации SQL: {exc}") from exc

    def _strip_think_block(self, text: str) -> tuple[str, str | None]:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if not match:
            return text, None

        thinking_content = match.group(1).strip()
        clean_text = text[: match.start()] + text[match.end() :]
        return clean_text.strip(), thinking_content
