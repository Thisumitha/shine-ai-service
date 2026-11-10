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

