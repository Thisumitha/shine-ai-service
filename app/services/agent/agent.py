from dataclasses import dataclass
import logging

from app.domain.entities.citation import Citation
from app.domain.ports.chat_store import ChatStore
from app.domain.ports.llm_client import LLMClient
from app.domain.ports.retriever import Retriever
from app.services.agent.act import ActStage
from app.services.agent.observe import ObserveStage
from app.services.agent.persist import PersistStage
from app.services.agent.reason import ReasonStage
from app.services.answer_service import AnswerService
from app.services.self_check_service import SelfCheckService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResponse:
    """The final structured response returned by the chat agent."""

    answer: str
    citations: list[Citation]
    session_id: str
    message_id: str
    intent: str


class ChatAgent:
    """The 4-stage conversational agent coordinating Observe, Reason, Act, and Persist."""

    def __init__(
        self,
        observe_stage: ObserveStage,
        reason_stage: ReasonStage,
        act_stage: ActStage,
        persist_stage: PersistStage,
    ) -> None:
        self.observe = observe_stage
        self.reason = reason_stage
        self.act = act_stage
        self.persist = persist_stage

    @classmethod
    def create(
        cls,
        chat_store: ChatStore,
        llm_client: LLMClient,
        retriever: Retriever,
        answer_service: AnswerService,
        self_check_service: SelfCheckService,
    ) -> "ChatAgent":
        """Factory method to instantiate ChatAgent with all 4 stages wired cleanly."""
        return cls(
            observe_stage=ObserveStage(chat_store=chat_store),
            reason_stage=ReasonStage(llm_client=llm_client),
            act_stage=ActStage(
                retriever=retriever,
                answer_service=answer_service,
                self_check_service=self_check_service,
            ),
            persist_stage=PersistStage(chat_store=chat_store),
        )

    def handle_message(
        self,
        user_id: int | str,
        course_id: int | str | None = None,
        message: str = "",
        session_id: str | None = None,
        course_ids: set[int] | list[int] | None = None,
    ) -> AgentResponse:
        """Run the complete agent pipeline for an incoming user message."""
        # Determine effective course ID and set of course IDs
        effective_course_ids: set[int] = set()
        if course_ids:
            for cid in course_ids:
                try:
                    effective_course_ids.add(int(cid))
                except (ValueError, TypeError):
                    pass
        elif course_id is not None:
            try:
                effective_course_ids.add(int(course_id))
            except (ValueError, TypeError):
                pass

        effective_course_id = course_id if course_id is not None else (
            next(iter(effective_course_ids)) if effective_course_ids else "general"
        )

        # 1. Observe: Load conversation history or create session
        active_session_id, history = self.observe.observe(
            user_id=user_id,
            course_id=effective_course_id,
            session_id=session_id,
        )

        # 2. Reason: Classify intent in JSON mode
        reason_result = self.reason.reason(
            message=message,
            history=history,
        )
        logger.info(
            "Agent reasoned for session %s: intent=%s, reason=%s",
            active_session_id,
            reason_result.intent,
            reason_result.reason,
        )

        # 3. Act: Retrieval, grounded answer generation, and self-check
        answer, citations = self.act.act(
            user_id=user_id,
            course_id=effective_course_id,
            query=message,
            reason_result=reason_result,
            history=history,
            course_ids=effective_course_ids,
        )

        # 4. Persist: Save user query and assistant answer to SQLite
        saved_msg = self.persist.persist(
            session_id=active_session_id,
            query=message,
            answer=answer,
            citations=citations,
        )

        return AgentResponse(
            answer=answer,
            citations=citations,
            session_id=active_session_id,
            message_id=saved_msg.message_id,
            intent=reason_result.intent,
        )

