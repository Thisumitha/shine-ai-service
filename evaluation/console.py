"""Readable terminal output for the evaluation (plain text, no log lines)."""

import textwrap

from evaluation.metrics import EvaluationReport

WIDTH = 78


def _short(text: str, width: int) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="...")


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def header(config: str, total: int, mock: bool) -> str:
    kind = "SYNTHETIC (mock) run - numbers are not real" if mock else "live run"
    return f"\n{'=' * WIDTH}\n Evaluating {total} questions | {config} | {kind}\n{'=' * WIDTH}"


def question_start(index: int, total: int, question: str) -> str:
    return f"\n[{index}/{total}] {_short(question, WIDTH - 8)}"


def question_result(result: dict, seconds: float) -> str:
    expected = result.get("expected_pages") or []
    retrieved = result.get("retrieved_pages") or []
    answerable = result["is_answerable"]
    refused = result["is_refusal"]

    if answerable:
        hit = "Hit@1 yes" if result["hit_at_1"] else ("Hit@5 yes" if result["hit_at_5"] else "MISS")
        pages = " ".join(f"p.{p}" for p in retrieved) or "nothing"
        wanted = " ".join(f"p.{p}" for p in expected) or "-"
        status = "FALSE REFUSAL (answerable question refused)" if refused else "answered"
    else:
        hit, pages, wanted = "n/a", " ".join(f"p.{p}" for p in retrieved) or "nothing", "none (not in the PDF)"
        status = "refused correctly" if refused else "ANSWERED (should have refused)"

    lines = [
        f"      retrieved : {pages}    expected: {wanted}    {hit}",
        f"      answer    : {_short(result['answer'], WIDTH - 20)}",
    ]
    scores = [
        f"context {_num(result.get('context_relevance'))}",
        f"answer-rel {_num(result.get('answer_relevance'))}",
        f"grounded {_num(result.get('groundedness'))}",
    ]
    if result.get("context_recall") is not None:
        scores.append(f"recall {_num(result['context_recall'])}")
    if answerable or result.get("context_relevance") is not None:
        lines.append("      scores    : " + " | ".join(scores))
    lines.append(f"      result    : {status}  ({seconds:.0f} s)")
    return "\n".join(lines)


def ablation(reports: list[EvaluationReport], mock: bool = False) -> str:
    """One compact row per configuration."""
    out = [
        f"\n{'=' * WIDTH}",
        " ABLATION: dense only vs TF-IDF only vs hybrid vs hybrid + self-check" + ("  (SYNTHETIC)" if mock else ""),
        "=" * WIDTH,
        f" {'Configuration':<24} {'Hit@3':<6} {'MRR':<5} {'Context':<8} {'Ground.':<8} {'AnsRel':<7} {'FalseRef':<8}",
    ]
    for r in reports:
        out.append(
            f" {r.config_name:<24} {_pct(r.hit_at_3):<6} {r.mrr:<5.2f} {_num(r.context_relevance):<8} "
            f"{_pct(r.groundedness):<8} {_num(r.answer_relevance):<7} {_pct(r.false_refusal_rate):<8}"
        )
    out.append("=" * WIDTH)
    return "\n".join(out)


def summary(report: EvaluationReport, mock: bool = False) -> str:
    out = [
        f"\n{'=' * WIDTH}",
        f" RESULT: {report.config_name}  |  {report.total_queries} questions" + ("  |  SYNTHETIC" if mock else ""),
        "=" * WIDTH,
        f" Retrieval   Hit@1 {_pct(report.hit_at_1)}   Hit@3 {_pct(report.hit_at_3)}   "
        f"Hit@5 {_pct(report.hit_at_5)}   MRR {report.mrr:.2f}",
        f" RAG quality Context relevance {_num(report.context_relevance)}   "
        f"Groundedness {_pct(report.groundedness)}   Answer relevance {_num(report.answer_relevance)}"
        + (f"   Context recall {_num(report.context_recall)}" if report.context_recall is not None else ""),
        f" Safety      Refusal accuracy {_pct(report.refusal_accuracy)}   "
        f"False refusals {_pct(report.false_refusal_rate)}",
        f" Overlap     Answer F1 vs ground truth {_pct(report.answer_f1)}",
        "-" * WIDTH,
        f" {'#':>2}  {'Question':<36} {'Hit@3':<6} {'Context':<8} {'Ground.':<8} {'AnsRel':<7}",
    ]
    for i, r in enumerate(report.per_item_results, 1):
        hit3 = "yes" if r["hit_at_3"] and r["is_answerable"] else ("-" if not r["is_answerable"] else "no")
        out.append(
            f" {i:>2}  {_short(r['question'], 36):<36} {hit3:<6} {_num(r.get('context_relevance')):<8} "
            f"{_num(r.get('groundedness')):<8} {_num(r.get('answer_relevance')):<7}"
        )
    out.append("=" * WIDTH)
    return "\n".join(out)
