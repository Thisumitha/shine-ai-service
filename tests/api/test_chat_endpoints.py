from dataclasses import dataclass
from typing import Any
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routers.chat import chat_endpoint
from app.api.routers.feedback import submit_feedback
from app.api.routers.sessions import get_session_detail, list_user_sessions
from app.api.schemas.chat import ChatRequest, CitationSchema
from app.api.schemas.feedback import FeedbackRequest
from app.domain.entities.chunk import RetrievedChunk
from app.domain.entities.user import CurrentUser
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
    """Deterministic LLM for API integration testing."""

    def __init__(self) -> None:
        self.reason_intent = INTENT_COURSE_CONTENT
        self.answer_text = "Binary search divides the sorted array in half in O(log n) time. [S1]"
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
                f'"reason": "Verified against course excerpt"}}'
            )
        return f'{{"intent": "{self.reason_intent}", "reason": "Academic inquiry", "search_needed": true}}'


class MockRetriever(Retriever):
    """Returns fixed course chunks for testing."""

    def search(self, query: str, course_ids: set[int], top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="ch-algo-1",
                course_id=101,
                file_name="Algorithms_101.pdf",
                page_number=14,
                content="Binary search finds items in sorted lists in O(log n) time.",
                score=0.92,
            )
        ]


@dataclass
class MockContainer:
    chat_store: SqliteChatStore
    chat_agent: ChatAgent
    job_store: Any = None
    ocr: Any = None
    worker: Any = None


@pytest.fixture
def api_harness():
    """Builds a test harness with mocked LLM and retriever over SQLite memory."""
    llm = MockLLM()
    chat_store = SqliteChatStore(db_path=":memory:")
    retriever = MockRetriever()
    answer_service = AnswerService(llm_client=llm)
    self_check_service = SelfCheckService(llm_client=llm, chat_store=chat_store, score_threshold=0.70)

    observe = ObserveStage(chat_store)
    reason = ReasonStage(llm)
    act = ActStage(retriever, answer_service, self_check_service)
    persist = PersistStage(chat_store)
    chat_agent = ChatAgent(observe, reason, act, persist)

    mock_container = MockContainer(chat_store=chat_store, chat_agent=chat_agent)
    test_user = CurrentUser(email="student@shine.lk", role="student", user_id=1)

    return mock_container, test_user, llm


# --- 1. Chat Endpoint Tests ---

def test_chat_endpoint_success(api_harness):
    container, user, llm = api_harness
    request = ChatRequest(
        session_id="sess-test-01",
        course_id=101,
        message="What is the time complexity of binary search?",
    )

    response = chat_endpoint(body=request, user=user, container=container)

    assert response.session_id == "sess-test-01"
    assert "Binary search" in response.answer
    assert response.is_refusal is False
    assert len(response.citations) == 1
    assert response.citations[0].file_name == "Algorithms_101.pdf"
    assert response.citations[0].page_number == 14


def test_chat_endpoint_refusal(api_harness):
    container, user, llm = api_harness
    llm.entailment_score = 0.20  # Below 0.70 threshold

    request = ChatRequest(
        session_id="sess-refusal-01",
        course_id=101,
        message="Explain quantum entanglement in Chapter 1.",
    )

    response = chat_endpoint(body=request, user=user, container=container)

    assert response.is_refusal is True
    assert "not found in your course material" in response.answer.lower() or "couldn't find" in response.answer.lower()


def test_chat_endpoint_validation_error():
    # Empty message should be rejected by Pydantic min_length=1
    with pytest.raises(ValidationError):
        ChatRequest(session_id="sess-bad", course_id=101, message="")


# --- 2. Session Endpoints Tests ---

def test_list_and_get_session_detail(api_harness):
    container, user, llm = api_harness

    # 1. First trigger a chat to create a session and message
    request = ChatRequest(session_id="sess-list-01", course_id=101, message="Hello class!")
    chat_endpoint(body=request, user=user, container=container)

    # 2. List sessions
    sessions = list_user_sessions(course_id=None, user=user, container=container)
    assert len(sessions) >= 1
    assert any(s.session_id == "sess-list-01" for s in sessions)

    # 3. Filter by course
    filtered = list_user_sessions(course_id=101, user=user, container=container)
    assert len(filtered) >= 1

    empty = list_user_sessions(course_id=999, user=user, container=container)
    assert len(empty) == 0

    # 4. Get session detail
    detail = get_session_detail(session_id="sess-list-01", user=user, container=container)
    assert detail.session_id == "sess-list-01"
    assert len(detail.messages) == 2  # 1 user + 1 assistant turn
    assert detail.messages[0].role == "user"
    assert detail.messages[1].role == "assistant"


def test_get_nonexistent_session_returns_404(api_harness):
    container, user, llm = api_harness
    with pytest.raises(HTTPException) as exc_info:
        get_session_detail(session_id="non-existent-session-id", user=user, container=container)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


# --- 3. Feedback Endpoint Tests ---

def test_submit_valid_positive_feedback(api_harness):
    container, user, llm = api_harness
    request = FeedbackRequest(
        message_id="msg-12345",
        rating=1,
        comment="Very helpful slide citation!",
    )

    response = submit_feedback(body=request, user=user, container=container)
    assert response.status == "success"


def test_submit_valid_negative_feedback(api_harness):
    container, user, llm = api_harness
    request = FeedbackRequest(
        message_id="msg-12346",
        rating=-1,
        comment="Answer was slightly confusing.",
    )

    response = submit_feedback(body=request, user=user, container=container)
    assert response.status == "success"


def test_submit_invalid_rating_rejected():
    # Ratings must be strictly 1 or -1
    with pytest.raises(ValidationError):
        FeedbackRequest(message_id="msg-12347", rating=5, comment="5 stars")


# --- 4. Router Wiring Verification ---

def test_api_routes_registered():
    """Verifies that all Phase 6 routes are registered in the FastAPI OpenAPI specification."""
    app = create_app()
    paths = set(app.openapi()["paths"].keys())

    assert "/api/v1/ai/chat" in paths
    assert "/api/v1/ai/sessions" in paths
    assert "/api/v1/ai/sessions/{session_id}" in paths
    assert "/api/v1/ai/feedback" in paths
