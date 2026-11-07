import json
from unittest.mock import MagicMock

from app.domain.entities.chunk import RetrievedChunk
from app.domain.entities.citation import Citation
from app.domain.ports.chat_store import ChatStore
from app.domain.ports.llm_client import LLMClient
from app.services.answer_service import REFUSAL_MESSAGE
from app.services.self_check_service import SelfCheckService


def test_self_check_passes_when_claims_supported():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "grounded": True,
        "score": 0.95,
        "unsupported_claims": [],
        "reason": "All statements match source text.",
    })
    mock_store = MagicMock(spec=ChatStore)

    service = SelfCheckService(llm_client=mock_llm, chat_store=mock_store, score_threshold=0.7)

    chunk = RetrievedChunk(chunk_id="c1", course_id=1, file_name="f.pdf", page_number=1, content="Text")
    citation = Citation(source_id="S1", file_name="f.pdf", page_number=1, snippet="Text")

    final_answer, final_citations, is_grounded = service.verify(
        user_id="user_1",
        course_id="101",
        query="What is X?",
        answer="X is defined as Text [S1].",
        chunks=[chunk],
        citations=[citation],
    )

    assert is_grounded is True
    assert final_answer == "X is defined as Text [S1]."
    assert len(final_citations) == 1
    mock_store.log_unanswered.assert_not_called()


def test_self_check_rejects_and_logs_when_score_low():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_json.return_value = json.dumps({
        "grounded": False,
        "score": 0.35,
        "unsupported_claims": ["Outside knowledge claim"],
        "reason": "Claim not found in excerpts.",
    })
    mock_store = MagicMock(spec=ChatStore)

    service = SelfCheckService(llm_client=mock_llm, chat_store=mock_store, score_threshold=0.7)

    chunk = RetrievedChunk(chunk_id="c1", course_id=1, file_name="f.pdf", page_number=1, content="Text")
    citation = Citation(source_id="S1", file_name="f.pdf", page_number=1, snippet="Text")

    final_answer, final_citations, is_grounded = service.verify(
        user_id="user_1",
        course_id="101",
        query="Explain Y?",
        answer="Y includes outside knowledge claim [S1].",
        chunks=[chunk],
        citations=[citation],
    )

    assert is_grounded is False
    assert final_answer == REFUSAL_MESSAGE
    assert final_citations == []

    # Verify that the unanswerable question was logged to ChatStore
    mock_store.log_unanswered.assert_called_once()
    logged_obj = mock_store.log_unanswered.call_args[0][0]
    assert logged_obj.user_id == "user_1"
    assert logged_obj.course_id == "101"
    assert "UNSUPPORTED_CLAIMS" in logged_obj.reason


def test_self_check_returns_refusal_when_no_chunks():
    mock_llm = MagicMock(spec=LLMClient)
    mock_store = MagicMock(spec=ChatStore)

    service = SelfCheckService(llm_client=mock_llm, chat_store=mock_store)

    final_answer, final_citations, is_grounded = service.verify(
        user_id="user_1",
        course_id="101",
        query="Explain Z?",
        answer="Z is something.",
        chunks=[],
        citations=[],
    )

    assert is_grounded is False
    assert final_answer == REFUSAL_MESSAGE
    assert final_citations == []
    mock_llm.generate_json.assert_not_called()
    mock_store.log_unanswered.assert_called_once()

