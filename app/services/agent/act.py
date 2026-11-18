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
    INTENT_STUDENT_DATA,
    ReasonResult,
)
from app.services.agent.rewrite import QueryRewriter
from app.services.agent.state import AgentTrace, RetrievalStep, RetrievedHit
from app.services.agent.student_data import StudentDataStage
from app.services.answer_service import AnswerService, REFUSAL_MESSAGE
from app.services.self_check_service import SelfCheckService

logger = logging.getLogger(__name__)

TOP_K = 4  # chunks per search
MAX_MERGED_CHUNKS = 6  # cap for a multi-part question, to keep the prompt small
MAX_ATTEMPTS = 2  # first try plus one retry with a rewritten query

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
        student_data_stage: StudentDataStage | None = None,
        query_rewriter: QueryRewriter | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.student_data_stage = student_data_stage
        self.retriever = retriever
        self.answer_service = answer_service
        self.self_check_service = self_check_service
        self.query_rewriter = query_rewriter
        self.max_attempts = max(1, max_attempts)

    def act(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        reason_result: ReasonResult,
        history: list[Message] | None = None,
        course_ids: set[int] | list[int] | None = None,
        student_email: str | None = None,
    ) -> tuple[str, list[Citation]]:
        """Execute action and return (final_answer, citations)."""
        return self.act_traced(
            user_id=user_id,
            course_id=course_id,
            query=query,
            reason_result=reason_result,
            trace=AgentTrace(),
            history=history,
            course_ids=course_ids,
            student_email=student_email,
        )

    def act_traced(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        reason_result: ReasonResult,
        trace: AgentTrace,
        history: list[Message] | None = None,
        course_ids: set[int] | list[int] | None = None,
        student_email: str | None = None,
    ) -> tuple[str, list[Citation]]:
        """Like act(), and records the searches, scores and self-check verdict in `trace`."""
        intent = reason_result.intent

        if intent == INTENT_SMALL_TALK:
            return SMALL_TALK_REPLY, []

        if intent == INTENT_OFF_TOPIC:
            return OFF_TOPIC_REPLY, []

        if intent == INTENT_STUDENT_DATA and self.student_data_stage is not None:
            return self.student_data_stage.answer(student_email, query, history), []

        # Default / COURSE_CONTENT path:
        return self._handle_course_content(
            user_id=user_id,
            course_id=course_id,
            query=query,
            history=history or [],
            course_ids=course_ids,
            sub_queries=list(reason_result.sub_queries),
            trace=trace,
        )

    def _handle_course_content(
        self,
        user_id: int | str,
        course_id: int | str,
        query: str,
        history: list[Message],
        course_ids: set[int] | list[int] | None = None,
        sub_queries: list[str] | None = None,
        trace: AgentTrace | None = None,
    ) -> tuple[str, list[Citation]]:
        """Bounded act loop: retrieve, answer with [S1] tags, self-check, and retry once if it fails.

        A retry rewrites the search query when nothing was retrieved or the self-check rejected
        the answer. After `max_attempts` the agent refuses instead of guessing.
        """
        trace = trace if trace is not None else AgentTrace()
        target_course_ids: set[int] = set()
        if course_ids:
            for cid in course_ids:
                try:
                    target_course_ids.add(int(cid))
                except (ValueError, TypeError):
                    pass
        elif course_id is not None:
            try:
                target_course_ids.add(int(course_id))
            except (ValueError, TypeError):
                pass

        if not target_course_ids:
            trace.note("No course available to search.")
            return REFUSAL_MESSAGE, []

        searches = list(sub_queries or []) or [query]
        trace.sub_queries = list(searches)

        for attempt in range(1, self.max_attempts + 1):
            trace.attempts = attempt
            can_retry = attempt < self.max_attempts

            # 1. Search via the abstract Retriever port, once per planned sub-query
            chunks = self._gather(searches, attempt, target_course_ids, course_id, trace)

            if not chunks:
                trace.note(f"Attempt {attempt}: nothing retrieved.")
                searches = self._rewritten(searches, query, can_retry, trace)
                if searches:
                    continue
                return REFUSAL_MESSAGE, []

            # 2. Generate candidate answer with citations
            draft_answer, candidate_citations = self.answer_service.answer(
                query=query,
                chunks=chunks,
                history=history,
            )

            # 3. Self-check verifier: audits entailment and triggers refusal if unsupported
            final_answer, final_citations, grounded = self.self_check_service.verify(
                user_id=user_id,
                course_id=course_id,
                query=query,
                answer=draft_answer,
                chunks=chunks,
                citations=candidate_citations,
            )
            trace.grounded = grounded
            if grounded:
                return final_answer, final_citations

            trace.note(f"Attempt {attempt}: self-check rejected the answer.")
            searches = self._rewritten(searches, query, can_retry, trace)
            if not searches:
                return final_answer, final_citations

        return REFUSAL_MESSAGE, []

    def _rewritten(
        self,
        searches: list[str],
        query: str,
        can_retry: bool,
        trace: AgentTrace,
    ) -> list[str]:
        """The next round of searches, or [] when the agent should stop."""
        if not can_retry or self.query_rewriter is None:
            return []
        rewritten = self.query_rewriter.rewrite(query)
        if rewritten is None:
            trace.note("Rewrite gave no new query; stopping.")
            return []
        trace.note(f"Retrying with rewritten query: {rewritten!r}")
        return [rewritten]

    def _gather(
        self,
        searches: list[str],
        attempt: int,
        course_ids: set[int],
        default_course_id: int | str,
        trace: AgentTrace,
    ) -> list[RetrievedChunk]:
        """Run every search, record each step, and merge the chunks (round-robin, no duplicates)."""
        per_search: list[list[RetrievedChunk]] = []
        for sub_query in searches:
            try:
                raw_results = self.retriever.search(query=sub_query, course_ids=course_ids, top_k=TOP_K)
            except Exception as exc:
                logger.error("Retriever search failed: %s", exc)
                raw_results = []
            trace.steps.append(
                RetrievalStep(
                    sub_query=sub_query,
                    attempt=attempt,
                    hits=tuple(RetrievedHit.from_raw(r) for r in raw_results if isinstance(r, dict)),
                )
            )
            # Normalize results into strongly-typed RetrievedChunk objects
            per_search.append(self._normalize_chunks(raw_results, default_course_id=default_course_id))

        merged: list[RetrievedChunk] = []
        seen: set[str] = set()
        limit = TOP_K if len(per_search) == 1 else MAX_MERGED_CHUNKS
        for rank in range(max((len(p) for p in per_search), default=0)):
            for results in per_search:
                if rank < len(results) and results[rank].chunk_id not in seen and len(merged) < limit:
                    seen.add(results[rank].chunk_id)
                    merged.append(results[rank])
        return merged

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
                file_name = metadata.get("title") or metadata.get("file_name", "Course Document")
                page_number = int(metadata.get("page_number", 1))
                score = float(item.get("score", 1.0 - float(item.get("distance", 0.0))))
                heading_path = metadata.get("heading_path", "")
                c_id = metadata.get("course_id", default_course_id)
                item_id = metadata.get("item_id")
                if item_id is None and "-" in chunk_id:
                    prefix = chunk_id.split("-")[0]
                    if prefix.isdigit():
                        item_id = int(prefix)

                normalized.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        course_id=c_id,
                        file_name=file_name,
                        page_number=page_number,
                        content=content,
                        score=score,
                        heading_path=heading_path,
                        item_id=int(item_id) if item_id is not None else None,
                    )
                )

        return normalized

