import logging
from typing import Any

from app.domain.entities.chunk import Chunk, RetrievedChunk
from app.domain.entities.citation import Citation
from app.domain.entities.message import Message
from app.domain.ports.retriever import Retriever
from app.services.agent.reason import (
    INTENT_COURSE_CONTENT,
    INTENT_OFF_TOPIC,
    INTENT_SMALL_TALK,
    ReasonResult,
)
from app.services.answer_service import AnswerService
from app.services.self_check_service import SelfCheckService

logger = logging.getLogger(__name__)

OFF_TOPIC_REPLY = (
    "I am Shine AI, your dedicated course academic assistant. I can only assist with questions "
    "related to your course modules, lecture slides, and academic notes. Please ask any question "
    "about your course materials!"
)

SMALL_TALK_REPLY = (
    "Hello! I am Shine AI, your academic tutor. How can I help you with your course lectures, "
    "assignments, or exam preparation today?"
)


class ActStage:
    """Stage 3: Executes the appropriate action based on reasoning intent."""

    def __init__(
        self,
        retriever: Retriever,
        answer_service: AnswerService,
        self_check_service: SelfCheckService,
    ) -> None:
        self.retriever = retriever
        self.answer_service = answer_service
        self.self_check_service = self_check_service

    def act(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        reason_result: ReasonResult,
        history: list[Message] | None = None,
    ) -> tuple[str, list[Citation]]:
        """Execute action and return (final_answer, citations)."""
        intent = reason_result.intent

        if intent == INTENT_SMALL_TALK:
            return SMALL_TALK_REPLY, []

        if intent == INTENT_OFF_TOPIC:
            return OFF_TOPIC_REPLY, []

        # Default / COURSE_CONTENT path:
        return self._handle_course_content(
            user_id=user_id,
            course_id=course_id,
            query=query,
            history=history or [],
        )

    def _handle_course_content(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        history: list[Message],
    ) -> tuple[str, list[Citation]]:
        """Coordinate retrieval, answer generation with [S1] tags, and self-check audit."""
        try:
            numeric_course_id = int(course_id)
            course_ids = {numeric_course_id}
        except (ValueError, TypeError):
            course_ids = set()

        # 1. Search via Member 2's abstract Retriever port
        try:
            raw_results = self.retriever.search(query=query, course_ids=course_ids, top_k=4)
        except Exception as exc:
            logger.error("Retriever search failed: %s", exc)
            raw_results = []

        # 2. Normalize results into strongly-typed RetrievedChunk objects
        chunks = self._normalize_chunks(raw_results, default_course_id=course_id)

        # 3. Generate candidate answer with citations
        draft_answer, candidate_citations = self.answer_service.answer(
            query=query,
            chunks=chunks,
            history=history,
        )

        # 4. Self-check verifier: Audits entailment and triggers refusal if unsupported
        final_answer, final_citations, _ = self.self_check_service.verify(
            user_id=user_id,
            course_id=course_id,
            query=query,
            answer=draft_answer,
            chunks=chunks,
            citations=candidate_citations,
        )

        return final_answer, final_citations

    @staticmethod
    def _normalize_chunks(raw_results: list[Any], default_course_id: int | str) -> list[RetrievedChunk]:
        """Convert varied retriever outputs (dicts, Chunk, RetrievedChunk) into RetrievedChunk."""
        normalized: list[RetrievedChunk] = []

        for item in raw_results:
            if isinstance(item, RetrievedChunk):
                normalized.append(item)
            elif isinstance(item, Chunk):
                normalized.append(RetrievedChunk.from_chunk(item))
            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                content = item.get("text") or item.get("content", "")
                chunk_id = str(item.get("id") or metadata.get("chunk_id", ""))
                file_name = metadata.get("file_name") or metadata.get("title", "Course Document")
                page_number = int(metadata.get("page_number", 1))
                score = float(item.get("score", 1.0 - float(item.get("distance", 0.0))))
                heading_path = metadata.get("heading_path", "")
                c_id = metadata.get("course_id", default_course_id)

                normalized.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        course_id=c_id,
                        file_name=file_name,
                        page_number=page_number,
                        content=content,
                        score=score,
                        heading_path=heading_path,
                    )
                )

        return normalized

