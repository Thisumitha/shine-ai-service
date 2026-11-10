from dataclasses import dataclass
from typing import Any
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routers.internal import check_secret
from app.api.routers.internal_chat import internal_chat_endpoint
from app.api.schemas.internal_chat import (
    InternalChatRequest,
    InternalChatResponse,
    InternalSourceSchema,
)
from app.config import Settings
from app.domain.entities.chunk import RetrievedChunk
from app.domain.ports.llm_client import LLMClient
from app.domain.ports.retriever import Retriever
from app.infrastructure.memory.sqlite_chat_store import SqliteChatStore
from app.main import create_app
from app.services.agent.act import ActStage
from app.services.agent.agent import ChatAgent
from app.services.agent.observe import ObserveStage
from app.services.agent.persist import PersistStage
from app.services.agent.reason import INTENT_COURSE_CONTENT, ReasonStage
from app.services.answer_service import AnswerService
from app.services.self_check_service import SelfCheckService


class MockLLM(LLMClient):
    """Deterministic LLM for internal chat testing."""

    def __init__(self) -> None:
        self.reason_intent = INTENT_COURSE_CONTENT
        self.answer_text = "Photosynthesis is the process by which green plants convert light into chemical energy. [S1]"
        self.entailment_score = 0.95

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self.answer_text

    def generate_json(
        self,
        prompt: str,
        schema_dict: dict | None = None,
        system_instruction: str | None = None,
    ) -> str:
        if "CANDIDATE ANSWER" in prompt:
            grounded = self.entailment_score >= 0.70
            return (
                f'{{"grounded": {str(grounded).lower()}, '
                f'"score": {self.entailment_score}, '
                f'"unsupported_claims": [], '
                f'"reason": "Verified against biology excerpt"}}'
            )
        return f'{{"intent": "{self.reason_intent}", "reason": "Academic query", "search_needed": true}}'


class MockMultiCourseRetriever(Retriever):
    """Retriever that returns biology notes when matching authorized course IDs."""

    def __init__(self) -> None:
        self.last_course_ids: set[int] | None = None

    def search(self, query: str, course_ids: set[int], top_k: int = 5) -> list[RetrievedChunk]:
        self.last_course_ids = course_ids
        if 12 in course_ids:
            return [
                RetrievedChunk(
                    chunk_id="345-0",
                    course_id=12,
                    file_name="Biology Notes",
                    page_number=4,
                    content="Photosynthesis is the process by which green plants convert light into chemical energy.",
                    score=0.96,
                    item_id=345,
                )
            ]
        return []


@dataclass
class MockContainer:
    settings: Settings
    chat_store: SqliteChatStore
    chat_agent: ChatAgent
    retriever: MockMultiCourseRetriever
    job_store: Any = None
    ocr: Any = None
    worker: Any = None


@pytest.fixture
def internal_chat_harness():
    llm = MockLLM()
    chat_store = SqliteChatStore(db_path=":memory:")
    retriever = MockMultiCourseRetriever()
    answer_service = AnswerService(llm_client=llm)
    self_check_service = SelfCheckService(llm_client=llm, chat_store=chat_store, score_threshold=0.70)

    observe = ObserveStage(chat_store)
    reason = ReasonStage(llm)
    act = ActStage(retriever, answer_service, self_check_service)
    persist = PersistStage(chat_store)
    chat_agent = ChatAgent(observe, reason, act, persist)

    settings = Settings(internal_secret="top-secret-123")
    mock_container = MockContainer(
        settings=settings,
        chat_store=chat_store,
        chat_agent=chat_agent,
        retriever=retriever,
    )

    return mock_container, llm, retriever


def test_internal_chat_success(internal_chat_harness):
    container, llm, retriever = internal_chat_harness

    request = InternalChatRequest(
        question="What is photosynthesis?",
        courseIds=[12, 15],
    )

    response: InternalChatResponse = internal_chat_endpoint(body=request, container=container)

    # 1. Passed as set of ints across all courseIds
    assert retriever.last_course_ids == {12, 15}

    # 2. Answer and generated sessionId
    assert "Photosynthesis" in response.answer
    assert response.session_id is not None
    assert len(response.session_id) > 0
    assert response.is_refusal is False

    # 3. Sources check: courseId, itemId, materialTitle, page
    assert len(response.sources) == 1
    src = response.sources[0]
    assert src.course_id == 12
    assert src.item_id == 345
    assert src.material_title == "Biology Notes"
    assert src.page == 4

    # 4. JSON serialization matches Spring camelCase schema
    data = response.model_dump(by_alias=True)
    assert data["sessionId"] == response.session_id
    assert data["sources"][0]["courseId"] == 12
    assert data["sources"][0]["itemId"] == 345
    assert data["sources"][0]["materialTitle"] == "Biology Notes"
    assert data["sources"][0]["page"] == 4


def test_internal_chat_preserves_session_id(internal_chat_harness):
    container, llm, retriever = internal_chat_harness

    request = InternalChatRequest(
        question="What is photosynthesis?",
        courseIds=[12],
        sessionId="spring-sess-999",
    )

    response = internal_chat_endpoint(body=request, container=container)
    assert response.session_id == "spring-sess-999"


def test_internal_chat_refusal_when_no_material_matches(internal_chat_harness):
    container, llm, retriever = internal_chat_harness

    # Course 99 has no material in mock retriever
    request = InternalChatRequest(
        question="What is photosynthesis?",
        courseIds=[99],
    )

    response = internal_chat_endpoint(body=request, container=container)

    # Spring expects 200 with refusal text in answer and empty sources list
    assert response.is_refusal is True
    assert "not found in your course material" in response.answer.lower() or "couldn't find" in response.answer.lower()
    assert response.sources == []
    assert len(response.sources) == 0


def test_check_secret_rejects_wrong_or_missing_secret(internal_chat_harness):
    container, llm, retriever = internal_chat_harness

    # Wrong secret -> 401
    with pytest.raises(HTTPException) as exc_wrong:
        check_secret(x_internal_secret="wrong", container=container)
    assert exc_wrong.value.status_code == 401

    # Missing secret -> 401
    with pytest.raises(HTTPException) as exc_missing:
        check_secret(x_internal_secret="", container=container)
    assert exc_missing.value.status_code == 401

    # Valid secret -> None (passes)
    check_secret(x_internal_secret="top-secret-123", container=container)


def test_internal_chat_route_registered_in_openapi():
    app = create_app()
    paths = set(app.openapi()["paths"].keys())
    assert "/internal/chat" in paths

