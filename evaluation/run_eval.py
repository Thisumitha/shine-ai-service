"""Ablation benchmark runner for Shine AI Service.

Usage:
    # Live evaluation against the ingested PDFs (default, needs GEMINI_API_KEY)
    python -m evaluation.run_eval --config hybrid_with_self_check

    # Synthetic smoke test only (numbers are NOT real measurements)
    python -m evaluation.run_eval --mock
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import time

from app.domain.entities.chunk import RetrievedChunk
from app.logging_setup import quiet_libraries
from app.services.agent.act import ActStage
from app.services.answer_service import REFUSAL_MESSAGE
from evaluation import console, live_judges
from evaluation.metrics import (
    EvaluationReport,
    context_recall,
    hit_at_k,
    reciprocal_rank,
    refusal_score,
    token_overlap_f1,
)

quiet_libraries()  # no HTTP log lines; the console view below shows the progress instead
logger = logging.getLogger("evaluator")


EVAL_TOP_K = 5  # Hit@5 needs five retrieved chunks


def _retriever_for(config_name: str, container: Any) -> Any:
    """The retriever an ablation configuration measures."""
    chosen = {
        "dense_only": container.dense_retriever,
        "sparse_only": container.sparse_indexer,
    }.get(config_name, container.retriever)
    if chosen is None:
        raise RuntimeError(f"The '{config_name}' retriever is not available (check that PDFs are ingested).")
    return chosen


def load_dataset(dataset_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Loads benchmark Q&A pairs from JSON Lines file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if limit is not None and limit > 0:
        items = items[:limit]
    return items


class MockAblationRetriever:
    """Simulates realistic dense, sparse, and hybrid search hits for benchmark questions."""

    def __init__(self, mode: str = "hybrid") -> None:
        self.mode = mode

    def search(self, query: str, expected_pages: list[int], is_answerable: bool) -> list[RetrievedChunk]:
        if not is_answerable:
            return []

        retrieved: list[RetrievedChunk] = []
        target_page = expected_pages[0] if expected_pages else 1

        if self.mode == "dense_only":
            # Dense vector search finds target page in top 3 in ~82% of cases
            ranks = [target_page, target_page + 1, target_page - 1]
        elif self.mode == "sparse_only":
            # Sparse keyword search finds target page in ~75% of cases
            ranks = [target_page + 1, target_page, target_page + 2]
        else:  # hybrid or hybrid_with_self_check
            # Fused RRF puts true page at top rank in >92% of cases
            ranks = [target_page, target_page + 1, target_page + 2]

        for idx, page in enumerate(ranks[:4]):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=f"chunk-{idx}",
                    course_id=101,
                    file_name="Course_Material.pdf",
                    page_number=page,
                    content=f"Content excerpt for {query} on slide page {page}.",
                    score=0.95 - (idx * 0.1),
                    item_id=100 + idx,
                )
            )
        return retrieved


def evaluate_single_query(
    item: dict[str, Any],
    config_name: str,
    mock: bool = True,
    container: Any = None,
) -> dict[str, Any]:
    """Evaluates a single benchmark item under the specified configuration."""
    question = item["question"]
    expected_pages = item.get("expected_page_numbers", [])
    is_answerable = item.get("is_answerable", True)
    ground_truth = item.get("ground_truth_answer", "")
    live_scores: dict[str, float] = {}
    live = not (mock or container is None or container.chat_agent is None)

    if not live:
        retriever = MockAblationRetriever(mode=config_name)
        chunks = retriever.search(question, expected_pages, is_answerable)
        retrieved_pages = [c.page_number for c in chunks]

        if not is_answerable:
            is_refusal = True
            answer = REFUSAL_MESSAGE
        elif config_name == "sparse_only" and chunks and chunks[0].page_number != expected_pages[0]:
            is_refusal = False
            answer = ground_truth[:60] + "..."
        else:
            is_refusal = False
            answer = ground_truth
    else:
        # Live evaluation: retrieve with the retriever this configuration names, answer, then score.
        course_id = int(item.get("course_id", 1))
        hits = _retriever_for(config_name, container).search(query=question, course_ids={course_id}, top_k=EVAL_TOP_K)
        if config_name == "hybrid_with_self_check":
            # the full agent: plan, retrieve, answer, self-check, bounded retry
            answer = container.chat_agent.handle_message(
                user_id="eval_runner", course_id=course_id, message=question
            ).answer
        else:
            # same generation step, but no self-check verifier
            chunks = ActStage._normalize_chunks(hits[:4], default_course_id=course_id)
            answer = container.answer_service.answer(query=question, chunks=chunks)[0] if chunks else REFUSAL_MESSAGE
        is_refusal = answer.strip() == REFUSAL_MESSAGE.strip()
        retrieved_pages = [h["metadata"].get("page_number") for h in hits]
        if is_answerable:
            live_scores = {
                "context_relevance": live_judges.context_relevance(
                    question, [h["text"] for h in hits], container.embedder
                ),
                "answer_relevance": live_judges.answer_relevance(question, answer, container.embedder),
            }
            if not is_refusal:
                grounded = live_judges.groundedness(answer, hits[:4], container.llm_client)
                if grounded is not None:  # None = the judge failed, so there is no score to average
                    live_scores["groundedness"] = grounded
            if item.get("ground_truth_context"):
                live_scores["context_recall"] = context_recall(item["ground_truth_context"], [h["text"] for h in hits])

    # Calculate metrics for this query
    h1 = hit_at_k(retrieved_pages, expected_pages, k=1) if is_answerable else 1.0
    h3 = hit_at_k(retrieved_pages, expected_pages, k=3) if is_answerable else 1.0
    h5 = hit_at_k(retrieved_pages, expected_pages, k=5) if is_answerable else 1.0
    rr = reciprocal_rank(retrieved_pages, expected_pages) if is_answerable else 1.0
    ref_score = refusal_score(is_answerable=is_answerable, is_refusal=is_refusal)
    f1 = token_overlap_f1(answer, ground_truth) if is_answerable else ref_score
    if live:
        groundedness = live_scores.get("groundedness")  # None for refusals: they make no claims
    elif is_refusal:
        groundedness = None
    else:
        groundedness = 1.0 if h3 > 0.0 else 0.5  # synthetic heuristic, mock runs only

    return {
        "id": item["id"],
        "question": question,
        "is_answerable": is_answerable,
        "is_refusal": is_refusal,
        "hit_at_1": h1,
        "hit_at_3": h3,
        "hit_at_5": h5,
        "reciprocal_rank": rr,
        "refusal_score": ref_score,
        "answer_f1": f1,
        "groundedness": groundedness,
        "context_relevance": live_scores.get("context_relevance"),
        "answer_relevance": live_scores.get("answer_relevance"),
        "context_recall": live_scores.get("context_recall"),
        "answer": answer,
        "retrieved_pages": retrieved_pages,
        "expected_pages": expected_pages,
    }


def run_evaluation(
    dataset_path: str = "evaluation/sample_dataset.jsonl",
    config_name: str = "hybrid_with_self_check",
    limit: int | None = None,
    mock: bool = False,
    verbose: bool = True,
) -> EvaluationReport:
    """Executes evaluation across all dataset queries for a given configuration."""
    items = load_dataset(dataset_path, limit=limit)
    if verbose:
        print(console.header(config_name, len(items), mock), flush=True)

    container = None
    if not mock:
        try:
            from app.container import build_container
            container = build_container()
        except Exception as exc:
            raise RuntimeError(f"Could not start the live system: {exc}") from exc
        if container.chat_agent is None:
            raise RuntimeError("No chat agent: ingest a PDF first (python -m app.cli ingest <pdf>).")

    results = []
    for index, item in enumerate(items, 1):
        if verbose:
            print(console.question_start(index, len(items), item["question"]), flush=True)
        started = time.monotonic()
        res = evaluate_single_query(item, config_name=config_name, mock=mock, container=container)
        results.append(res)
        if verbose:
            print(console.question_result(res, time.monotonic() - started), flush=True)

    total = len(results)
    ans_items = [r for r in results if r["is_answerable"]]
    ans_total = len(ans_items) or 1

    avg_h1 = sum(r["hit_at_1"] for r in ans_items) / ans_total
    avg_h3 = sum(r["hit_at_3"] for r in ans_items) / ans_total
    avg_h5 = sum(r["hit_at_5"] for r in ans_items) / ans_total
    avg_mrr = sum(r["reciprocal_rank"] for r in ans_items) / ans_total
    avg_refusal = sum(r["refusal_score"] for r in results) / total
    claims = [r["groundedness"] for r in results if r["groundedness"] is not None]
    avg_groundedness = sum(claims) / len(claims) if claims else None
    false_refusal = sum(1 for r in ans_items if r["is_refusal"]) / ans_total
    avg_f1 = sum(r["answer_f1"] for r in ans_items) / ans_total
    live = [r for r in results if r.get("context_relevance") is not None]
    avg_ctx = sum(r["context_relevance"] for r in live) / len(live) if live else None
    avg_ans_rel = sum(r["answer_relevance"] for r in live) / len(live) if live else None
    recalls = [r["context_recall"] for r in results if r.get("context_recall") is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else None

    return EvaluationReport(
        config_name=config_name,
        total_queries=total,
        hit_at_1=avg_h1,
        hit_at_3=avg_h3,
        hit_at_5=avg_h5,
        mrr=avg_mrr,
        refusal_accuracy=avg_refusal,
        groundedness=avg_groundedness,
        answer_f1=avg_f1,
        context_relevance=avg_ctx,
        answer_relevance=avg_ans_rel,
        context_recall=avg_recall,
        false_refusal_rate=false_refusal,
        per_item_results=results,
    )


def run_ablation_study(
    dataset_path: str = "evaluation/sample_dataset.jsonl",
    limit: int | None = None,
    mock: bool = False,
) -> str:
    """Runs all 4 configurations and builds a comparative ablation matrix."""
    configs = ["dense_only", "sparse_only", "hybrid", "hybrid_with_self_check"]
    reports: list[EvaluationReport] = []

    for cfg in configs:
        rep = run_evaluation(dataset_path=dataset_path, config_name=cfg, limit=limit, mock=mock)
        reports.append(rep)
    print(console.ablation(reports, mock), flush=True)

    note = "_Synthetic run (mock): these numbers are not measurements._\n\n" if mock else ""
    table = (
        "### Architectural Ablation Study Results\n\n"
        + note
        + "| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Context Rel. | Context Recall | Groundedness | Answer Rel. | False refusals | Refusal acc. |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )

    def cell(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    for r in reports:
        table += (
            f"| **{r.config_name}** | {r.hit_at_1 * 100:.1f}% | {r.hit_at_3 * 100:.1f}% | {r.hit_at_5 * 100:.1f}% | "
            f"{r.mrr:.3f} | {cell(r.context_relevance)} | {cell(r.context_recall)} | "
            f"{'n/a' if r.groundedness is None else f'{r.groundedness * 100:.1f}%'} | {cell(r.answer_relevance)} | "
            f"{r.false_refusal_rate * 100:.1f}% | {r.refusal_accuracy * 100:.1f}% |\n"
        )

    return table


def main():
    parser = argparse.ArgumentParser(description="Run RAG & Agent Evaluation Benchmark")
    parser.add_argument(
        "--config",
        default="hybrid_with_self_check",
        choices=["dense_only", "sparse_only", "hybrid", "hybrid_with_self_check", "all"],
        help="Configuration to benchmark",
    )
    parser.add_argument("--dataset", default="evaluation/sample_dataset.jsonl", help="Path to benchmark dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated")
    parser.add_argument("--mock", action="store_true", default=False, help="Synthetic smoke test; the numbers are not real")
    parser.add_argument("--no-mock", dest="mock", action="store_false", help="Run against live components (default)")
    parser.add_argument("--output", default=None, help="File to save Markdown report")

    args = parser.parse_args()

    if args.config == "all":
        report_md = run_ablation_study(dataset_path=args.dataset, limit=args.limit, mock=args.mock)
    else:
        report = run_evaluation(dataset_path=args.dataset, config_name=args.config, limit=args.limit, mock=args.mock)
        report_md = report.to_markdown()
        print(console.summary(report, mock=args.mock))  # readable on screen; the Markdown is for the file

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        print(f"\nFull Markdown report saved to {args.output}")
    elif args.config == "all":
        print(report_md)


if __name__ == "__main__":
    main()

