import logging
import time
from typing import Any, Callable

from google.genai import errors

from app.domain.ports.llm_client import LLMClient
from app.infrastructure.llm.error_messages import plain_error
from app.infrastructure.llm.gemini_client import GeminiClient
from app.infrastructure.network_errors import is_network_error

logger = logging.getLogger(__name__)


class FallbackLLMClient(LLMClient):
    """Use `primary`; when it is down, busy or out of quota, answer with `fallback` instead.

    After a failure the primary is skipped for `cooldown_seconds`, so a quota that is gone for hours
    does not add a slow failed attempt to every question.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient, cooldown_seconds: float = 60.0):
        self.primary = primary
        self.fallback = fallback
        self.cooldown_seconds = cooldown_seconds
        self._primary_down_until = 0.0

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._run(lambda c: c.generate(prompt, system_instruction=system_instruction))

    def generate_json(
        self,
        prompt: str,
        schema_dict: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        return self._run(
            lambda c: c.generate_json(prompt, schema_dict=schema_dict, system_instruction=system_instruction)
        )

    def _run(self, call: Callable[[LLMClient], str]) -> str:
        if time.monotonic() < self._primary_down_until:
            return call(self.fallback)
        try:
            return call(self.primary)
        except Exception as err:
            if not self._primary_unavailable(err):
                raise
            self._primary_down_until = time.monotonic() + self.cooldown_seconds
            logger.debug("Primary LLM failure detail: %s", err)
            logger.warning(
                "Gemini is not available (%s). Using the local Ollama model for the next %.0fs.",
                plain_error(err),
                self.cooldown_seconds,
            )
            return call(self.fallback)

    @staticmethod
    def _primary_unavailable(err: Exception) -> bool:
        if isinstance(err, errors.APIError):
            return GeminiClient._should_fall_back(err)
        return is_network_error(err)
