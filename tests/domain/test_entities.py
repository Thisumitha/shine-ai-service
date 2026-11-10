from dataclasses import FrozenInstanceError
import pytest

from app.domain.entities import (
    ChatSession,
    Citation,
    Feedback,
    Message,
    RetrievedChunk,
    UnansweredQuery,
)
from app.domain.ports import ChatStore, LLMClient


def test_citation_creation_and_immutability():
    citation = Citation(
        source_id="S1",
        file_name="lecture1.pdf",
        page_number=5,
        snippet="Photosynthesis is the process...",
    )
    assert citation.source_id == "S1"
    assert citation.file_name == "lecture1.pdf"
    assert citation.page_number == 5
    assert "Photosynthesis" in citation.snippet

    with pytest.raises(FrozenInstanceError):
        citation.page_number = 6  # type: ignore[misc]


def test_retrieved_chunk_creation_and_defaults():
    chunk = RetrievedChunk(
        chunk_id="chunk_42",
        course_id=101,
        file_name="biology.pdf",
        page_number=3,
        content="Plant cells contain chloroplasts.",
    )
    assert chunk.score == 0.0
    assert chunk.heading_path == ""

    chunk_with_score = RetrievedChunk(
        chunk_id="chunk_43",
        course_id="CS101",
        file_name="intro.pdf",
        page_number=1,
        content="Binary trees...",
        score=0.88,
        heading_path="Chapter 1 > Data Structures",
    )
    assert chunk_with_score.score == 0.88
    assert chunk_with_score.heading_path == "Chapter 1 > Data Structures"


def test_message_creation_and_citations():
    citation = Citation(
        source_id="S1",
        file_name="doc.pdf",
        page_number=2,
        snippet="Important concept",
    )
    msg = Message(
        message_id="msg_1",
        session_id="sess_1",
        role="assistant",
        content="Here is the explanation [S1].",
        citations=[citation],
    )
    assert msg.role == "assistant"
    assert len(msg.citations) == 1
    assert msg.citations[0].source_id == "S1"

    with pytest.raises(FrozenInstanceError):
        msg.content = "New text"  # type: ignore[misc]


def test_chat_session_creation():
    session = ChatSession(
        session_id="sess_100",
        user_id=1,
        course_id=10,
        title="Cell Biology Discussion",
    )
    assert session.session_id == "sess_100"
    assert session.user_id == 1
    assert session.title == "Cell Biology Discussion"


def test_feedback_creation():
    fb = Feedback(
        id=None,
        message_id="msg_1",
        user_id=1,
        rating=1,
        comment="Great explanation!",
    )
    assert fb.rating == 1
    assert fb.comment == "Great explanation!"


def test_unanswered_query_creation():
    uq = UnansweredQuery(
        id=None,
        user_id=1,
        course_id=10,
        query="What is quantum gravity?",
        reason="OUT_OF_SCOPE",
    )
    assert uq.reason == "OUT_OF_SCOPE"
    assert uq.query == "What is quantum gravity?"


def test_ports_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        ChatStore()  # type: ignore[abstract]

