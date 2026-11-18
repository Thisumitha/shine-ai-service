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


def context_recall(ground_truth_context: str, retrieved_texts: list[str]) -> float:
    """Share of the ground-truth passage's distinct words that appear in the retrieved chunks.

    1.0 means everything the reference passage says was retrieved (at least lexically);
    it is computed without an LLM so the number is reproducible.
    """
    words = set(re.findall(r"\w+", ground_truth_context.lower()))
    if not words:
        return 0.0
    retrieved = set(re.findall(r"\w+", " ".join(retrieved_texts).lower()))
    return len(words & retrieved) / len(words)


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
    groundedness: float | None  # mean over answers that made claims; refusals are not scored
    answer_f1: float
    context_relevance: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    false_refusal_rate: float = 0.0  # answerable questions the system refused
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
            f"| **Groundedness** (claims supported by retrieved chunks) | {_pct(self.groundedness)} | &ge; 90.0% |\n"
            f"| **False refusal rate** (answerable but refused) | {self.false_refusal_rate * 100:.1f}% | &le; 10.0% |\n"
            f"| **Answer F1 vs ground truth** | {self.answer_f1 * 100:.1f}% | &ge; 70.0% |\n"
            + self._live_rows()
            + self._per_query_table()
        )

    def _live_rows(self) -> str:
        if self.context_relevance is None:
            return "\n_Synthetic run (mock): the numbers above are not real measurements._\n"
        rows = (
            f"| **Context Relevance** (cosine, question vs chunks) | {self.context_relevance:.3f} | &ge; 0.600 |\n"
            f"| **Answer Relevance** (cosine, question vs answer) | {_num(self.answer_relevance)} | &ge; 0.600 |\n"
        )
        if self.context_recall is not None:
            rows += f"| **Context Recall** (ground-truth passage found in chunks) | {self.context_recall:.3f} | &ge; 0.700 |\n"
        return rows

    def _per_query_table(self) -> str:
        rows = [r for r in self.per_item_results if r.get("context_relevance") is not None]
        if not rows:
            return ""
        lines = [
            "\n#### Per-query scores\n",
            "| ID | Question | Response | Context Rel. | Groundedness | Answer Rel. |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
        for r in rows:
            answer = " ".join(r["answer"].split()).replace("|", "/")
            lines.append(
                f"| {r['id']} | {r['question']} | {answer} | {r['context_relevance']:.2f} | "
                f"{_num(r.get('groundedness'), 2)} | {_num(r.get('answer_relevance'), 2)} |"
            )
        return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"

