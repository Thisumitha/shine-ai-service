from evaluation.metrics import (
    EvaluationReport,
    hit_at_k,
    reciprocal_rank,
    refusal_score,
    token_overlap_f1,
)
from evaluation.run_eval import load_dataset, run_evaluation


def test_hit_at_k():
    expected = [4]
    # Hit@1
    assert hit_at_k(retrieved_pages=[4, 5, 6], expected_pages=expected, k=1) == 1.0
    # Miss@1, Hit@3
    assert hit_at_k(retrieved_pages=[2, 3, 4], expected_pages=expected, k=1) == 0.0
    assert hit_at_k(retrieved_pages=[2, 3, 4], expected_pages=expected, k=3) == 1.0
    # Miss entirely
    assert hit_at_k(retrieved_pages=[1, 2, 3], expected_pages=expected, k=3) == 0.0
    # Empty expected
    assert hit_at_k(retrieved_pages=[1, 2], expected_pages=[], k=3) == 0.0


def test_reciprocal_rank():
    expected = [10]
    # Rank 1 -> 1.0
    assert reciprocal_rank(retrieved_pages=[10, 11, 12], expected_pages=expected) == 1.0
    # Rank 2 -> 0.5
    assert reciprocal_rank(retrieved_pages=[9, 10, 11], expected_pages=expected) == 0.5
    # Rank 4 -> 0.25
    assert reciprocal_rank(retrieved_pages=[1, 2, 3, 10], expected_pages=expected) == 0.25
    # Not found -> 0.0
    assert reciprocal_rank(retrieved_pages=[1, 2, 3], expected_pages=expected) == 0.0


def test_token_overlap_f1():
    cand = "Photosynthesis converts light into chemical energy"
    ref = "Photosynthesis is the process that converts light into chemical energy"
    score = token_overlap_f1(cand, ref)
    assert 0.70 < score <= 1.0

    # Exact match
    assert token_overlap_f1("exact match test", "exact match test") == 1.0

    # Completely disjoint
    assert token_overlap_f1("apple banana", "car truck") == 0.0


def test_refusal_score():
    # Unanswerable query correctly refused -> 1.0
    assert refusal_score(is_answerable=False, is_refusal=True) == 1.0
    # Unanswerable query hallucinated an answer -> 0.0
    assert refusal_score(is_answerable=False, is_refusal=False) == 0.0
    # Answerable query correctly answered -> 1.0
    assert refusal_score(is_answerable=True, is_refusal=False) == 1.0
    # Answerable query falsely refused -> 0.0
    assert refusal_score(is_answerable=True, is_refusal=True) == 0.0


def test_load_dataset():
    items = load_dataset("evaluation/dataset.jsonl")
    assert len(items) >= 40

    for item in items:
        assert "id" in item
        assert "question" in item
        assert "is_answerable" in item
        assert "expected_page_numbers" in item


def test_run_evaluation_mock():
    report = run_evaluation(dataset_path="evaluation/dataset.jsonl", config_name="hybrid", mock=True)

    assert isinstance(report, EvaluationReport)
    assert report.total_queries >= 40
    assert report.hit_at_1 > 0.80
    assert report.hit_at_3 > 0.85
    assert report.mrr > 0.80
    assert report.refusal_accuracy >= 0.90

    md = report.to_markdown()
    assert "Evaluation Benchmark Report" in md
    assert "Hit@1" in md
    assert "MRR" in md



def test_context_recall_counts_ground_truth_words_found_in_chunks():
    from evaluation.metrics import context_recall

    assert context_recall("binary search halves", ["Binary search halves the interval"]) == 1.0
    assert context_recall("binary search halves", ["linear scan"]) == 0.0
    assert context_recall("", ["anything"]) == 0.0


def test_sample_dataset_has_ground_truth_context_for_answerable_rows():
    items = load_dataset("evaluation/sample_dataset.jsonl")
    for item in items:
        assert bool(item["ground_truth_context"]) == item["is_answerable"]


def test_ablation_configs_pick_different_retrievers():
    from types import SimpleNamespace

    from evaluation.run_eval import _retriever_for

    container = SimpleNamespace(dense_retriever="dense", sparse_indexer="sparse", retriever="hybrid")
    assert _retriever_for("dense_only", container) == "dense"
    assert _retriever_for("sparse_only", container) == "sparse"
    assert _retriever_for("hybrid", container) == "hybrid"
    assert _retriever_for("hybrid_with_self_check", container) == "hybrid"


def test_refusals_are_not_scored_for_groundedness_and_count_as_false_refusals():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.services.answer_service import REFUSAL_MESSAGE

    retriever = MagicMock()
    retriever.search.return_value = [{"id": "1-0", "text": "t", "metadata": {"page_number": 4}}]
    agent = MagicMock()
    agent.handle_message.return_value = SimpleNamespace(answer=REFUSAL_MESSAGE)
    container = SimpleNamespace(
        chat_agent=agent, retriever=retriever, dense_retriever=retriever, sparse_indexer=retriever,
        embedder=MagicMock(embed=lambda texts: [[1.0, 0.0]] * len(texts)), llm_client=MagicMock(),
    )
    from evaluation.run_eval import evaluate_single_query

    result = evaluate_single_query(
        {"id": "q", "question": "?", "is_answerable": True, "expected_page_numbers": [4], "ground_truth_answer": "x"},
        config_name="hybrid_with_self_check", mock=False, container=container,
    )
    assert result["is_refusal"] is True
    assert result["groundedness"] is None
    assert result["refusal_score"] == 0.0


def test_groundedness_is_none_when_the_judge_fails_not_zero():
    import json
    from unittest.mock import MagicMock

    from evaluation.live_judges import groundedness

    chunks = [{"text": "Binary search halves the range.", "metadata": {"title": "a.pdf", "page_number": 3}}]
    llm = MagicMock()

    llm.generate_json.side_effect = RuntimeError("503 UNAVAILABLE")
    assert groundedness("An answer [S1].", chunks, llm) is None  # a failed judge says nothing about the answer

    llm.generate_json.side_effect = None
    llm.generate_json.return_value = json.dumps({"claims": []})
    assert groundedness("An answer [S1].", chunks, llm) is None

    llm.generate_json.return_value = json.dumps({"claims": [{"supported": True}, {"supported": False}]})
    assert groundedness("An answer [S1].", chunks, llm) == 0.5  # computed from the verdicts, not given by the model
