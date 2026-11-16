from dataclasses import dataclass
import json
import logging
from typing import Any

from app.domain.entities.message import Message
from app.domain.ports.llm_client import LLMClient
from app.services.prompts.reason_prompts import (
    REASON_INTENT_SYSTEM_PROMPT,
    build_reason_prompt,
)

logger = logging.getLogger(__name__)

INTENT_COURSE_CONTENT = "COURSE_CONTENT"
INTENT_SMALL_TALK = "SMALL_TALK"
INTENT_OFF_TOPIC = "OFF_TOPIC"
INTENT_STUDENT_DATA = "STUDENT_DATA"

VALID_INTENTS = {INTENT_COURSE_CONTENT, INTENT_SMALL_TALK, INTENT_OFF_TOPIC, INTENT_STUDENT_DATA}


@dataclass(frozen=True)
class ReasonResult:
    """The outcome of the reasoning stage."""

    intent: str
    reason: str
    search_needed: bool


class ReasonStage:
    """Stage 2: Analyzes query intent using LLM JSON mode."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def reason(self, message: str, history: list[Message] | None = None) -> ReasonResult:
        """Classify message intent into COURSE_CONTENT, SMALL_TALK, or OFF_TOPIC."""
        history_summary = self._summarize_history(history or [])
        prompt = build_reason_prompt(message=message, history_summary=history_summary)

        try:
            raw_json = self.llm.generate_json(
                prompt=prompt,
                system_instruction=REASON_INTENT_SYSTEM_PROMPT,
            )
            data: dict[str, Any] = json.loads(raw_json)
            intent = str(data.get("intent", INTENT_COURSE_CONTENT)).strip().upper()
            reason = str(data.get("reason", "Standard classification"))
            search_needed = bool(data.get("search_needed", intent == INTENT_COURSE_CONTENT))

            if intent not in VALID_INTENTS:
                logger.warning("Unrecognized intent '%s' returned by model. Defaulting to COURSE_CONTENT.", intent)
                intent = INTENT_COURSE_CONTENT
                search_needed = True

            return ReasonResult(
                intent=intent,
                reason=reason,
                search_needed=search_needed,
            )

        except Exception as exc:
            logger.warning("Reasoning classification failed (%s). Safely defaulting to COURSE_CONTENT.", exc)
            return ReasonResult(
                intent=INTENT_COURSE_CONTENT,
                reason="Default fallback due to classification error",
                search_needed=True,
            )

    @staticmethod
    def _summarize_history(history: list[Message], max_turns: int = 3) -> str:
        recent = history[-max_turns:] if len(history) > max_turns else history
        return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in recent if m.content)

