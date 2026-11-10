import json
from unittest.mock import MagicMock

import pytest

from app.domain.entities.chunk import Chunk
from app.domain.ports.llm_client import LLMClient
from app.domain.ports.retriever import Retriever
from app.domain.ports.sparse_indexer import SparseIndexer
from app.infrastructure.memory.sqlite_chat_store import SqliteChatStore
from app.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from app.services.agent.act import OFF_TOPIC_REPLY, SMALL_TALK_REPLY
from app.services.agent.agent import ChatAgent
from app.services.answer_service import AnswerService
from app.services.self_check_service import SelfCheckService


class MockDenseRetriever(Retriever):
    """Dense retriever returning semantic vector matches."""

    def __init__(self, canned_matches: list[dict]) -> None:
        self.canned_matches = canned_matches

    def search(self, query: str, course_ids: set[int], top_k: int) -> list[dict]:
        return [
            m for m in self.canned_matches
            if m["metadata"]["course_id"] in course_ids
        ][:top_k]


class MockSparseIndexer(SparseIndexer):
    """Sparse indexer returning keyword matches."""

    def __init__(self, canned_matches: list[dict]) -> None:
        self.canned_matches = canned_matches

    def index_item(self, item_id: int, chunks: list[Chunk]) -> None:
        pass

    def delete_by_item_id(self, item_id: int) -> None:
        pass

    def search(self, query: str, course_ids: set[int], top_k: int) -> list[dict]:
        return [
            m for m in self.canned_matches
            if m["metadata"]["course_id"] in course_ids
        ][:top_k]


@pytest.fixture
def end_to_end_harness():
    """Builds a real Member 2 HybridRetriever wired directly into Member 3's ChatAgent."""
    # Sample chunks representing Member 1's extracted PDF slides
    canned_dense = [
        {
            "id": "1-0",
            "text": "Binary search is an efficient algorithm that finds items in sorted lists in O(log n) time.",
            "distance": 0.12,
            "metadata": {
                "file_name": "Data_Structures.pdf",
                "page_number": 14,
                "course_id": 101,
                "heading_path": "Chapter 2 > Search",
            },
        }
    ]

    canned_sparse = [
        {
            "id": "1-0",
            "text": "Binary search is an efficient algorithm that finds items in sorted lists in O(log n) time.",
            "score": 4.85,
            "metadata": {
                "file_name": "Data_Structures.pdf",
                "page_number": 14,
                "course_id": 101,
                "heading_path": "Chapter 2 > Search",
            },
        }
    ]

    # Instantiate Member 2's REAL HybridRetriever with RRF algorithm
    dense_retriever = MockDenseRetriever(canned_dense)
    sparse_indexer = MockSparseIndexer(canned_sparse)
    real_hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_indexer=sparse_indexer,
        rrf_k=60,
    )

    # Instantiate Member 3's in-memory SQLite ChatStore
    chat_store = SqliteChatStore(db_path=":memory:")

    return {
        "hybrid_retriever": real_hybrid_retriever,
        "chat_store": chat_store,
    }


def test_end_to_end_academic_query_returns_cited_answer(end_to_end_harness):
    """End-to-End: Student asks course question -> Member 2's HybridRetriever RRF fuses results

    -> Member 3's AnswerService cites exact page -> SelfCheckService verifies claims -> SQLite persists.
    """
    hybrid_retriever = end_to_end_harness["hybrid_retriever"]
    chat_store = end_to_end_harness["chat_store"]

    mock_llm = MagicMock(spec=LLMClient)

    # 1. ReasonStage: Intent classified as COURSE_CONTENT
    # 2. SelfCheckService: Claim entailed by Page 14 (score 0.98)
    mock_llm.generate_json.side_effect = [
        json.dumps({
            "intent": "COURSE_CONTENT",
            "reason": "Student is asking about binary search time complexity",
            "search_needed": True,
        }),
        json.dumps({
            "grounded": True,
            "score": 0.98,
            "unsupported_claims": [],
            "reason": "Entailed by Page 14 excerpt",
        }),
    ]

    # Model generates answer with citation marker
    mock_llm.generate.return_value = (
        "Binary search operates in O(log n) time complexity on sorted lists [S1]."
    )

    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=chat_store)

    agent = ChatAgent.create(
        chat_store=chat_store,
        llm_client=mock_llm,
        retriever=hybrid_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    # Act: Send message
    response = agent.handle_message(
        user_id="student_101",
        course_id=101,
        message="What is the time complexity of binary search?",
    )

    # Assertions
    assert response.intent == "COURSE_CONTENT"
    assert "[S1]" in response.answer
    assert len(response.citations) == 1

    citation = response.citations[0]
    assert citation.source_id == "S1"
    assert citation.file_name == "Data_Structures.pdf"
    assert citation.page_number == 14
    assert "Binary search" in citation.snippet

    # Verify conversation history persisted in SQLite
    history = chat_store.get_history(response.session_id)
    assert len(history) == 2
    assert history[0].content == "What is the time complexity of binary search?"
    assert history[1].citations[0].page_number == 14


def test_end_to_end_off_topic_query_declined(end_to_end_harness):
    """End-to-End: Off-topic query -> Agent politely declines without querying HybridRetriever."""
    hybrid_retriever = end_to_end_harness["hybrid_retriever"]
    chat_store = end_to_end_harness["chat_store"]

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "intent": "OFF_TOPIC",
        "reason": "Sports question is outside academic course scope",
        "search_needed": False,
    })

    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=chat_store)

    agent = ChatAgent.create(
        chat_store=chat_store,
        llm_client=mock_llm,
        retriever=hybrid_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="student_101",
        course_id=101,
        message="Who won the cricket match?",
    )

    assert response.intent == "OFF_TOPIC"
    assert response.answer == OFF_TOPIC_REPLY
    assert response.citations == []


def test_end_to_end_small_talk_greeting(end_to_end_harness):
    """End-to-End: Greeting -> Agent replies with friendly academic greeting."""
    hybrid_retriever = end_to_end_harness["hybrid_retriever"]
    chat_store = end_to_end_harness["chat_store"]

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "intent": "SMALL_TALK",
        "reason": "Greeting message",
        "search_needed": False,
    })

    answer_service = AnswerService(llm_client=mock_llm)
    self_check_service = SelfCheckService(llm_client=mock_llm, chat_store=chat_store)

    agent = ChatAgent.create(
        chat_store=chat_store,
        llm_client=mock_llm,
        retriever=hybrid_retriever,
        answer_service=answer_service,
        self_check_service=self_check_service,
    )

    response = agent.handle_message(
        user_id="student_101",
        course_id=101,
        message="Hello! Good morning",
    )

    assert response.intent == "SMALL_TALK"
    assert response.answer == SMALL_TALK_REPLY
    assert response.citations == []
