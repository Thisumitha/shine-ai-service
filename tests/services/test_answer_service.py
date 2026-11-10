from unittest.mock import MagicMock

from app.domain.entities.chunk import RetrievedChunk
from app.domain.entities.message import Message
from app.domain.ports.llm_client import LLMClient
from app.services.answer_service import AnswerService, REFUSAL_MESSAGE


def test_answer_service_returns_refusal_when_no_chunks():
    mock_llm = MagicMock(spec=LLMClient)
    service = AnswerService(llm_client=mock_llm)

    answer, citations = service.answer(query="What is a binary tree?", chunks=[])
    assert answer == REFUSAL_MESSAGE
    assert citations == []
    mock_llm.generate.assert_not_called()


def test_answer_service_extracts_and_maps_citations():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = (
        "Binary trees consist of nodes with at most two children [S1]. "
        "A balanced tree provides O(log n) search time [S2]."
    )
    service = AnswerService(llm_client=mock_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            course_id=101,
            file_name="trees_lecture.pdf",
            page_number=4,
            content="A binary tree is a hierarchical data structure where each node has up to two children.",
        ),
        RetrievedChunk(
            chunk_id="c2",
            course_id=101,
            file_name="trees_lecture.pdf",
            page_number=8,
            content="Balanced binary trees guarantee O(log n) time complexity for search operations.",
        ),
    ]

    answer, citations = service.answer(query="Explain binary trees", chunks=chunks)

    assert "[S1]" in answer
    assert "[S2]" in answer
    assert len(citations) == 2

    assert citations[0].source_id == "S1"
    assert citations[0].file_name == "trees_lecture.pdf"
    assert citations[0].page_number == 4
    assert "hierarchical data structure" in citations[0].snippet

    assert citations[1].source_id == "S2"
    assert citations[1].file_name == "trees_lecture.pdf"
    assert citations[1].page_number == 8


def test_answer_service_deduplicates_repeated_citation_markers():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = "Point one [S1]. Another detail from the same place [S1]."
    service = AnswerService(llm_client=mock_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            course_id=101,
            file_name="doc.pdf",
            page_number=1,
            content="Source text here.",
        )
    ]

    answer, citations = service.answer(query="Question", chunks=chunks)
    assert len(citations) == 1
    assert citations[0].source_id == "S1"


def test_answer_service_handles_explicit_model_refusal():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = "I'm sorry, but this information is not found in your course material."
    service = AnswerService(llm_client=mock_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            course_id=101,
            file_name="doc.pdf",
            page_number=1,
            content="Unrelated text.",
        )
    ]

    answer, citations = service.answer(query="Unrelated question", chunks=chunks)
    assert answer == REFUSAL_MESSAGE
    assert citations == []


def test_answer_service_ignores_out_of_range_citation_markers():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = "Valid point [S1]. Hallucinated source [S99]."
    service = AnswerService(llm_client=mock_llm)

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            course_id=101,
            file_name="doc.pdf",
            page_number=1,
            content="Real chunk content.",
        )
    ]

    answer, citations = service.answer(query="Question", chunks=chunks)
    assert len(citations) == 1
    assert citations[0].source_id == "S1"

