from dataclasses import dataclass, field
import re
from typing import Any


def hit_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float:
    """Computes Hit@K metric (1.0 if any expected page is in top-K retrieved, else 0.0)."""
    if not expected_pages:
        return 0.0
    top_k = set(retrieved_pages[:k])
    return 1.0 if any(p in top_k for p in expected_pages) else 0.0


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    """Computes Reciprocal Rank (1.0 / rank of first retrieved expected page, or 0.0)."""
    if not expected_pages:
        return 0.0
    target_set = set(expected_pages)
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in target_set:
            return 1.0 / rank
    return 0.0


def token_overlap_f1(candidate: str, reference: str) -> float:
    """Computes token-level precision, recall, and F1 score between candidate and reference."""
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counts: dict[str, int] = {}
    for t in cand_tokens:
        cand_counts[t] = cand_counts.get(t, 0) + 1

    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    shared_tokens = sum(min(cand_counts[t], ref_counts.get(t, 0)) for t in cand_counts)

    if shared_tokens == 0:
        return 0.0

    precision = shared_tokens / len(cand_tokens)
    recall = shared_tokens / len(ref_tokens)
    return 2.0 * (precision * recall) / (precision + recall)


def refusal_score(is_answerable: bool, is_refusal: bool) -> float:
    """Evaluates whether the system appropriately answered or refused.

    - Unanswerable question: 1.0 if refused, 0.0 if hallucinated an answer.
    - Answerable question: 1.0 if answered, 0.0 if falsely refused.
    """
    if not is_answerable:
        return 1.0 if is_refusal else 0.0
    return 1.0 if not is_refusal else 0.0


@dataclass
class EvaluationReport:
    """Summary of evaluation metrics across a dataset."""

    config_name: str
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    refusal_accuracy: float
    groundedness: float
    answer_f1: float
    per_item_results: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render evaluation metrics as a GitHub-flavored markdown table."""
        return (
            f"### Evaluation Benchmark Report: `{self.config_name}`\n\n"
            f"- **Total Queries Evaluated**: {self.total_queries}\n\n"
            "| Metric | Score | Target Standard |\n"
            "| :--- | :---: | :---: |\n"
            f"| **Hit@1** | {self.hit_at_1 * 100:.1f}% | &ge; 65.0% |\n"
            f"| **Hit@3** | {self.hit_at_3 * 100:.1f}% | &ge; 80.0% |\n"
            f"| **Hit@5** | {self.hit_at_5 * 100:.1f}% | &ge; 90.0% |\n"
            f"| **MRR** (Mean Reciprocal Rank) | {self.mrr:.3f} | &ge; 0.750 |\n"
            f"| **Refusal Accuracy** | {self.refusal_accuracy * 100:.1f}% | &ge; 90.0% |\n"
            f"| **Groundedness** (Entailment) | {self.groundedness * 100:.1f}% | &ge; 90.0% |\n"
            f"| **Answer Relevance (F1)** | {self.answer_f1 * 100:.1f}% | &ge; 70.0% |\n"
        )

