from app.domain.entities.citation import Citation
from app.domain.entities.feedback import Feedback
from app.domain.entities.unanswered import UnansweredQuery
from app.infrastructure.memory.sqlite_chat_store import SqliteChatStore
import pytest


@pytest.fixture
def store():
    """Use an in-memory database for isolated, lightning-fast tests."""
    return SqliteChatStore(db_path=":memory:")


def test_create_and_get_session(store):
    session = store.create_session(user_id=42, course_id=101, title="Data Structures Intro")

    assert session.session_id is not None
    assert session.user_id == "42"
    assert session.course_id == "101"
    assert session.title == "Data Structures Intro"

    fetched = store.get_session(session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id
    assert fetched.title == "Data Structures Intro"

    # Non-existent session
    assert store.get_session("non-existent-id") is None


def test_list_sessions_by_user(store):
    sess1 = store.create_session(user_id="user_A", course_id="C1", title="Session 1")
    sess2 = store.create_session(user_id="user_A", course_id="C2", title="Session 2")
    sess_b = store.create_session(user_id="user_B", course_id="C1", title="Session B")

    sessions_a = store.list_sessions(user_id="user_A")
    assert len(sessions_a) == 2
    session_ids = [s.session_id for s in sessions_a]
    assert sess1.session_id in session_ids
    assert sess2.session_id in session_ids

    sessions_b = store.list_sessions(user_id="user_B")
    assert len(sessions_b) == 1
    assert sessions_b[0].session_id == sess_b.session_id


def test_save_message_and_get_history_with_citations(store):
    session = store.create_session(user_id=1, course_id=10, title="Biology")

    # 1. User message
    msg_user = store.save_message(
        session_id=session.session_id,
        role="user",
        content="What is mitochondria?",
    )
    assert msg_user.role == "user"
    assert msg_user.content == "What is mitochondria?"
    assert msg_user.citations == []

    # 2. Assistant message with citation
    citation = Citation(
        source_id="S1",
        file_name="cell_biology.pdf",
        page_number=12,
        snippet="Mitochondria are the powerhouses of the cell.",
    )
    msg_assistant = store.save_message(
        session_id=session.session_id,
        role="assistant",
        content="Mitochondria generate cellular energy [S1].",
        citations=[citation],
    )
    assert msg_assistant.role == "assistant"
    assert len(msg_assistant.citations) == 1
    assert msg_assistant.citations[0].page_number == 12

    # 3. Retrieve history (should return chronological order: user then assistant)
    history = store.get_history(session.session_id, limit=5)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert len(history[1].citations) == 1
    assert history[1].citations[0].file_name == "cell_biology.pdf"
    assert history[1].citations[0].snippet == "Mitochondria are the powerhouses of the cell."


def test_save_feedback(store):
    session = store.create_session(user_id=1, course_id=10, title="Math")
    msg = store.save_message(session_id=session.session_id, role="assistant", content="Answer")

    fb = Feedback(
        id=None,
        message_id=msg.message_id,
        user_id="1",
        rating=1,
        comment="Very helpful!",
    )
    store.save_feedback(fb)


def test_log_unanswered_query(store):
    unanswered = UnansweredQuery(
        id=None,
        user_id="1",
        course_id="101",
        query="Explain quantum foam",
        reason="OUT_OF_SCOPE",
    )
    store.log_unanswered(unanswered)


def test_disk_persisted_sqlite(tmp_path):
    db_file = str(tmp_path / "test_chat.sqlite")
    disk_store = SqliteChatStore(db_path=db_file)

    session = disk_store.create_session(user_id="user_disk", course_id="C99", title="Persistent Test")
    disk_store.save_message(session_id=session.session_id, role="user", content="Hello disk")

    # Open a new store instance pointing to the same file
    disk_store_2 = SqliteChatStore(db_path=db_file)
    fetched = disk_store_2.get_session(session.session_id)
    assert fetched is not None
    assert fetched.title == "Persistent Test"

    history = disk_store_2.get_history(session.session_id)
    assert len(history) == 1
    assert history[0].content == "Hello disk"
