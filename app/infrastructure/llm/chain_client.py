import logging
import time
from typing import Any, Callable

from app.domain.ports.llm_client import LLMClient
from app.infrastructure.llm.error_messages import LLMUnavailableError, is_daily_quota, is_service_error, plain_error
from app.infrastructure.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class ChainLLMClient(LLMClient):
    """Try several answer models in order; the first one that works answers.

    A model that just failed is skipped for a while, so a quota that is gone for hours does not add a slow
    failed attempt to every question. The last model is always tried, so there is always a real error to show.
    """

    def __init__(
        self,
        entries: list[tuple[str, LLMClient]],
        cooldown_seconds: float = 60.0,
        daily_quota_cooldown_seconds: float = 600.0,
    ):
        if not entries:
            raise ValueError("A chain needs at least one model.")
        self.entries = entries
        self.cooldown_seconds = cooldown_seconds
        self.daily_quota_cooldown_seconds = daily_quota_cooldown_seconds
        self._down_until: dict[str, float] = {}

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
        last = len(self.entries) - 1
        for index, (name, client) in enumerate(self.entries):
            if index < last and time.monotonic() < self._down_until.get(name, 0.0):
                continue
            try:
                return call(client)
            except Exception as err:
                if index == last or not self._can_switch(err):
                    raise
                wait = self.daily_quota_cooldown_seconds if is_daily_quota(err) else self.cooldown_seconds
                self._down_until[name] = time.monotonic() + wait
                logger.debug("%s failure detail: %s", name, err)
                logger.warning("%s: %s. Switching to %s.", name, plain_error(err), self.entries[index + 1][0])
        raise RuntimeError("Unexpected failure in the model chain.")

    @staticmethod
    def _can_switch(err: Exception) -> bool:
        """Busy, out of quota, missing or unreachable: another model may still answer. A bad request will not."""
        if isinstance(err, LLMUnavailableError):
            return True
        if is_service_error(err):
            return GeminiClient._should_fall_back(err)
        return False
