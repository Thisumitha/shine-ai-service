import json
import logging
from typing import Any

from app.domain.entities.chunk import RetrievedChunk
from app.domain.entities.citation import Citation
from app.domain.entities.unanswered import UnansweredQuery
from app.domain.ports.chat_store import ChatStore
from app.domain.ports.llm_client import LLMClient
from app.services.answer_service import REFUSAL_MESSAGE
from app.services.prompts.answer_prompts import build_context_block
from app.services.prompts.self_check_prompts import (
    SELF_CHECK_INSTRUCTION,
    build_self_check_prompt,
)

logger = logging.getLogger(__name__)


class SelfCheckService:
    """Verifies that generated answers are factually supported by course excerpts.

    Refuses answers and logs unanswered queries if claims fail verification.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        chat_store: ChatStore,
        score_threshold: float = 0.7,
    ) -> None:
        self.llm = llm_client
        self.chat_store = chat_store
        self.score_threshold = score_threshold

    def verify(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        citations: list[Citation],
    ) -> tuple[str, list[Citation], bool]:
        """Verify the candidate answer against the chunks.

        Returns (final_answer, final_citations, is_grounded).
        """
        # If the answer already triggered refusal or there are no source chunks
        if not chunks or answer.strip() == REFUSAL_MESSAGE:
            self._log_refusal(user_id, course_id, query, reason="NO_CONTEXT_OR_EXPLICIT_REFUSAL")
            return REFUSAL_MESSAGE, [], False

        chunk_dicts = [
            {
                "file_name": c.file_name,
                "page_number": c.page_number,
                "heading_path": c.heading_path,
                "content": c.content,
            }
            for c in chunks
        ]
        context_block = build_context_block(chunk_dicts)
        prompt = build_self_check_prompt(answer=answer, context_block=context_block)

        try:
            raw_json = self.llm.generate_json(
                prompt=prompt,
                system_instruction=SELF_CHECK_INSTRUCTION,
            )
            result: dict[str, Any] = json.loads(raw_json)
        except Exception as exc:
            logger.warning("Self-check JSON verification failed or timed out: %s. Defaulting to strict check.", exc)
            self._log_refusal(user_id, course_id, query, reason=f"VERIFICATION_ERROR_{exc}")
            return REFUSAL_MESSAGE, [], False

        score = float(result.get("score", 0.0))
        is_grounded = bool(result.get("grounded", False))

        if not is_grounded or score < self.score_threshold:
            unsupported = result.get("unsupported_claims", [])
            logger.info(
                "Answer rejected for course %s query '%s'. Score: %.2f. Unsupported claims: %s",
                course_id,
                query,
                score,
                unsupported,
            )
            self._log_refusal(
                user_id=user_id,
                course_id=course_id,
                query=query,
                reason=f"UNSUPPORTED_CLAIMS_SCORE_{score:.2f}",
            )
            return REFUSAL_MESSAGE, [], False

        return answer, citations, True

    def _log_refusal(self, user_id: int | str, course_id: int | str, query: str, reason: str) -> None:
        """Record unanswerable or ungrounded questions for course instructor analytics."""
        try:
            self.chat_store.log_unanswered(
                UnansweredQuery(
                    id=None,
                    user_id=user_id,
                    course_id=course_id,
                    query=query,
                    reason=reason,
                )
            )
        except Exception as exc:
            logger.error("Failed to log unanswered query: %s", exc)

