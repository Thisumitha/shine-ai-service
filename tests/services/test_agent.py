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



# --- Agentic loop: planning, bounded retry, trace -------------------------------------------


class SequenceRetriever(Retriever):
    """Returns a different canned result list on each call and records every query."""

    def __init__(self, *results: list[dict]) -> None:
        self.results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, course_ids: set[int], top_k: int) -> list[dict]:
        self.queries.append(query)
        return self.results.pop(0) if self.results else []


def _hit(chunk_id: str, page: int, text: str = "Binary search halves the interval.") -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "score": 0.03,
        "dense_rank": 1,
        "sparse_rank": 2,
        "dense_distance": 0.2,
        "sparse_math": {"dot_product": 1.0, "query_magnitude": 2.0, "chunk_magnitude": 3.0, "cosine": 0.17},
        "metadata": {"title": "algorithms.pdf", "page_number": page, "course_id": 101},
    }


def _build_agent(retriever: Retriever, mock_llm: MagicMock, store: SqliteChatStore) -> ChatAgent:
    return ChatAgent.create(
        chat_store=store,
        llm_client=mock_llm,
        retriever=retriever,
        answer_service=AnswerService(llm_client=mock_llm),
        self_check_service=SelfCheckService(llm_client=mock_llm, chat_store=store),
    )


def _reason_json(sub_queries: list[str] | None = None) -> str:
    return json.dumps({
        "intent": "COURSE_CONTENT",
        "reason": "course question",
        "search_needed": True,
        "sub_queries": sub_queries or [],
    })


GROUNDED = json.dumps({"grounded": True, "score": 0.95, "unsupported_claims": []})
UNGROUNDED = json.dumps({"grounded": False, "score": 0.2, "unsupported_claims": ["x"]})


def test_multi_part_question_is_planned_into_separate_searches():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.side_effect = [_reason_json(["what is binary search", "what is linear search"]), GROUNDED]
    mock_llm.generate.return_value = "Binary search halves the range [S1]; linear search scans [S2]."
    retriever = SequenceRetriever([_hit("1-1", 3)], [_hit("1-2", 5)])

    response = _build_agent(retriever, mock_llm, store).handle_message(
        user_id="u", course_id=101, message="Compare binary and linear search"
    )

    assert retriever.queries == ["what is binary search", "what is linear search"]
    assert [s.sub_query for s in response.trace.steps] == retriever.queries
    assert response.trace.sub_queries == retriever.queries
    assert response.trace.grounded is True
    assert response.trace.attempts == 1
    assert {c.page_number for c in response.citations} == {3, 5}


def test_trace_keeps_retrieval_scores_and_math():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.side_effect = [_reason_json(), GROUNDED]
    mock_llm.generate.return_value = "It halves the interval [S1]."

    response = _build_agent(SequenceRetriever([_hit("1-1", 3)]), mock_llm, store).handle_message(
        user_id="u", course_id=101, message="How does binary search work?"
    )

    hit = response.trace.steps[0].hits[0]
    assert (hit.rrf_score, hit.dense_rank, hit.sparse_rank, hit.dense_distance) == (0.03, 1, 2, 0.2)
    assert hit.sparse_math["cosine"] == 0.17


def test_retries_with_rewritten_query_when_nothing_is_retrieved():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.side_effect = [_reason_json(), GROUNDED]
    # first generate() call is the query rewrite, second is the answer
    mock_llm.generate.side_effect = ["binary search algorithm halving interval", "It halves the interval [S1]."]
    retriever = SequenceRetriever([], [_hit("1-1", 3)])

    response = _build_agent(retriever, mock_llm, store).handle_message(
        user_id="u", course_id=101, message="how does the thing that splits in two work"
    )

    assert retriever.queries == ["how does the thing that splits in two work", "binary search algorithm halving interval"]
    assert response.trace.attempts == 2
    assert response.citations[0].page_number == 3
    assert any("rewritten" in n for n in response.trace.notes)


def test_retries_after_self_check_rejects_the_first_answer():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.side_effect = [_reason_json(), UNGROUNDED, GROUNDED]
    # answer, rewrite, answer
    mock_llm.generate.side_effect = ["Unsupported claim [S1].", "binary search halving", "It halves the interval [S1]."]
    retriever = SequenceRetriever([_hit("1-1", 3)], [_hit("1-2", 4)])

    response = _build_agent(retriever, mock_llm, store).handle_message(
        user_id="u", course_id=101, message="How does binary search work?"
    )

    assert response.answer == "It halves the interval [S1]."
    assert response.trace.attempts == 2
    assert response.trace.grounded is True


def test_refuses_after_max_attempts_when_every_answer_is_rejected():
    store = SqliteChatStore(db_path=":memory:")
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.side_effect = [_reason_json(), UNGROUNDED, UNGROUNDED]
    mock_llm.generate.side_effect = ["Bad [S1].", "rewritten query", "Still bad [S1]."]
    retriever = SequenceRetriever([_hit("1-1", 3)], [_hit("1-2", 4)])

    response = _build_agent(retriever, mock_llm, store).handle_message(
        user_id="u", course_id=101, message="How does binary search work?"
    )

    assert response.answer == REFUSAL_MESSAGE
    assert response.trace.attempts == 2  # bounded: no third try
    assert response.trace.grounded is False


def test_single_part_question_ignores_a_one_item_plan():
    assert ReasonStage._clean_sub_queries(["only one"], "COURSE_CONTENT") == ()
    assert ReasonStage._clean_sub_queries(["a", "A", "b", "c", "d"], "COURSE_CONTENT") == ("a", "b", "c")
    assert ReasonStage._clean_sub_queries(["a", "b"], "SMALL_TALK") == ()
