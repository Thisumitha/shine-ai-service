import json
from unittest.mock import MagicMock

from app.domain.entities.citation import Citation
from app.domain.ports.chat_store import ChatStore
from app.domain.ports.llm_client import LLMClient
from app.domain.ports.retriever import Retriever
from app.infrastructure.memory.sqlite_chat_store import SqliteChatStore
from app.services.agent.act import OFF_TOPIC_REPLY, SMALL_TALK_REPLY, ActStage
from app.services.agent.agent import ChatAgent
from app.services.agent.observe import ObserveStage
from app.services.agent.persist import PersistStage
from app.services.agent.reason import ReasonResult, ReasonStage
from app.services.answer_service import AnswerService, REFUSAL_MESSAGE
from app.services.self_check_service import SelfCheckService


class StubRetriever(Retriever):
    """A lightweight stub retriever for isolated agent tests."""

    def __init__(self, canned_results: list[dict] | None = None) -> None:
        self.canned_results = canned_results or []
        self.last_query = ""

    def search(self, query: str, course_ids: set[int], top_k: int) -> list[dict]:
        self.last_query = query
        return self.canned_results


def test_agent_handles_small_talk_flow():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    # Reason stage outputs SMALL_TALK in JSON
    mock_llm.generate_json.return_value = json.dumps({
        "intent": "SMALL_TALK",
        "reason": "Student is greeting the assistant",
        "search_needed": False,
    })

    stub_retriever = StubRetriever()
    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=store)

    agent = ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=stub_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="user_1",
        course_id=101,
        message="Hello!",
    )

    assert response.intent == "SMALL_TALK"
    assert response.answer == SMALL_TALK_REPLY
    assert response.citations == []
    assert response.session_id is not None

    # Retriever was not called
    assert stub_retriever.last_query == ""

    # Message was persisted in SQLite
    history = store.get_history(response.session_id)
    assert len(history) == 2
    assert history[0].content == "Hello!"
    assert history[1].content == SMALL_TALK_REPLY


def test_agent_handles_off_topic_flow():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "intent": "OFF_TOPIC",
        "reason": "Movie recommendations are outside academic scope",
        "search_needed": False,
    })

    stub_retriever = StubRetriever()
    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=store)

    agent = ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=stub_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="user_1",
        course_id=101,
        message="Can you recommend a good movie to watch?",
    )

    assert response.intent == "OFF_TOPIC"
    assert response.answer == OFF_TOPIC_REPLY
    assert response.citations == []
    assert stub_retriever.last_query == ""


def test_agent_handles_course_content_successful_rag_flow():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)

    # 1. Reason stage returns COURSE_CONTENT
    # 2. Self-check returns grounded=True, score=0.95
    mock_llm.generate_json.side_effect = [
        json.dumps({
            "intent": "COURSE_CONTENT",
            "reason": "Question asks about search algorithms",
            "search_needed": True,
        }),
        json.dumps({
            "grounded": True,
            "score": 0.95,
            "unsupported_claims": [],
            "reason": "Claims match course document",
        }),
    ]

    # Generate answer text with citation marker
    mock_llm.generate.return_value = "Binary search divides the interval in half [S1]."

    # Canned match returned by retriever
    canned_chunks = [
        {
            "id": "chunk_10",
            "text": "Binary search is an algorithm that divides the search interval in half repeatedly.",
            "metadata": {
                "file_name": "algorithms_lecture.pdf",
                "page_number": 14,
                "course_id": 101,
            },
            "distance": 0.12,
        }
    ]
    stub_retriever = StubRetriever(canned_results=canned_chunks)

    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=store)

    agent = ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=stub_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="user_1",
        course_id=101,
        message="How does binary search work?",
    )

    assert response.intent == "COURSE_CONTENT"
    assert "[S1]" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].file_name == "algorithms_lecture.pdf"
    assert response.citations[0].page_number == 14

    # Check persistence in SQLite
    history = store.get_history(response.session_id)
    assert len(history) == 2
    assert len(history[1].citations) == 1
    assert history[1].citations[0].page_number == 14


def test_agent_handles_refusal_when_unsupported_or_no_chunks():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)

    mock_llm.generate_json.return_value = json.dumps({
        "intent": "COURSE_CONTENT",
        "reason": "Academic query but no material exists",
        "search_needed": True,
    })

    # Empty retriever results
    stub_retriever = StubRetriever(canned_results=[])
    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=store)

    agent = ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=stub_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="user_1",
        course_id=101,
        message="Explain quantum string theory in our biology class?",
    )

    assert response.answer == REFUSAL_MESSAGE
    assert response.citations == []


def test_agent_session_continuity():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "intent": "SMALL_TALK",
        "reason": "Greeting",
        "search_needed": False,
    })

    stub_retriever = StubRetriever()
    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=store)

    agent = ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=stub_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    # First turn
    resp1 = agent.handle_message(user_id="u1", course_id=10, message="Hi")
    session_id = resp1.session_id

    # Second turn passing same session_id
    resp2 = agent.handle_message(user_id="u1", course_id=10, message="Thanks!", session_id=session_id)
    assert resp2.session_id == session_id

    # History contains both turns (4 messages: u1, bot, u1, bot)
    history = store.get_history(session_id)
    assert len(history) == 4

