"""Command line entry point: ingest PDFs, ask questions, run the evaluation.

    python -m app.cli ingest <file.pdf | folder>
    python -m app.cli ask "your question" [--show-math]
    python -m app.cli eval [--dataset path]
    python -m app.cli mcp
    python -m app.cli load-sample | reset

Needs only GEMINI_API_KEY in .env. No Spring backend, MySQL or JWT.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from app.config import get_settings
from app.container import Container, build_container
from app.logging_setup import quiet_libraries
from app.services.agent.state import AgentTrace

ROOT = Path(__file__).resolve().parents[1]
DEMO_STUDENT = "demo.student@shine.lk"
DEFAULT_COURSE = 1


def _container() -> Container:
    settings = get_settings()
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is empty. Copy .env.example to .env and set it.")
    for folder in (settings.pdf_dir, settings.extracted_dir, settings.chunks_dir, settings.chroma_dir):
        Path(folder).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return build_container(settings)


def _registry_path() -> Path:
    return Path(get_settings().extracted_dir).parent / "ingested.json"


def _load_registry() -> dict[str, int]:
    path = _registry_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cmd_ingest(args: argparse.Namespace) -> None:
    container = _container()
    if container.ingestion is None:
        sys.exit("Ingestion is not available (check pymupdf, chromadb and google-genai are installed).")

    if container.ocr is not None and (problem := container.ocr.problem()):
        print(f"[WARNING] {problem} Image-only pages will be skipped; set TESSERACT_CMD in .env.")

    target = Path(args.path)
    pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not pdfs or not all(p.is_file() and p.suffix.lower() == ".pdf" for p in pdfs):
        sys.exit(f"No PDF found at {target}")

    registry = _load_registry()
    for pdf in pdfs:
        item_id = registry.get(pdf.name) or max(registry.values(), default=0) + 1
        print(f"[INGEST] {pdf.name} -> item {item_id}, course {args.course}")
        container.ingestion.process(
            {"item_id": item_id, "pdf_path": str(pdf), "course_id": args.course, "title": pdf.name}
        )
        registry[pdf.name] = item_id
        _registry_path().write_text(json.dumps(registry, indent=1), encoding="utf-8")
        chunks = json.loads((Path(get_settings().chunks_dir) / f"{item_id}.json").read_text(encoding="utf-8"))
        pages = sorted({c["page_number"] for c in chunks})
        span = f"pages {pages[0]}-{pages[-1]}" if pages else "no text found"
        print(f"         {len(chunks)} chunks ({span}) embedded into Chroma and indexed with TF-IDF")


def _print_steps(container: Container, trace: AgentTrace, show_math: bool) -> None:
    """Print what the agent actually retrieved, taken from its own trace (nothing is searched twice)."""
    for step in trace.steps:
        print(
            f"\n[RETRIEVE] attempt {step.attempt}, search {step.sub_query!r}: hybrid (dense + TF-IDF, "
            f"fused with Reciprocal Rank Fusion), {len(step.hits)} chunks"
        )
        for i, hit in enumerate(step.hits, 1):
            print(
                f"  #{i} {hit.file_name} p.{hit.page_number}  RRF={hit.rrf_score:.4f}  "
                f"dense_rank={hit.dense_rank} sparse_rank={hit.sparse_rank}"
            )
            if show_math:
                if hit.dense_distance is not None:
                    print(
                        f"      dense: cosine distance = {hit.dense_distance:.4f}, "
                        f"cosine similarity = 1 - distance = {1 - hit.dense_distance:.4f}"
                    )
                math = hit.sparse_math
                if math:
                    print(
                        f"      TF-IDF dot={math['dot_product']:.4f}  |q|={math['query_magnitude']:.4f}  "
                        f"|d|={math['chunk_magnitude']:.4f}  cosine = dot/(|q||d|) = {math['cosine']:.4f}"
                    )
        first = step.hits[0] if step.hits else None
        if show_math and first and container.sparse_indexer is not None:
            terms = container.sparse_indexer.explain_terms(step.sub_query, first.chunk_id)
            if terms:
                print("\n  Worked example for chunk #1 (dot product = sum of query weight x chunk weight per shared term):")
                for t in terms:
                    print(
                        f"      {t['term']!r:24} idf={t['idf']:.3f}  q={t['query_weight']:.3f}  "
                        f"d={t['chunk_weight']:.3f}  product={t['product']:.3f}"
                    )


def cmd_ask(args: argparse.Namespace) -> None:
    container = _container()
    agent = container.chat_agent
    if agent is None:
        sys.exit("Chat is not available. Ingest a PDF first: python -m app.cli ingest <pdf>")

    print(f"[OBSERVE] question: {args.question}")
    try:
        result = agent.handle_message(
            user_id="cli", course_id=args.course, message=args.question, student_email=DEMO_STUDENT
        )
    except Exception as exc:  # network down, quota, retired model: say why instead of a long traceback
        sys.exit(
            f"\nCould not get an answer: {type(exc).__name__}: {str(exc)[:300]}\n"
            "Check your internet connection and GEMINI_MODEL / GEMINI_API_KEY in .env, then try again."
        )
    trace = result.trace
    print(f"[REASON]  intent={result.intent} ({result.reason})")
    if trace and len(trace.sub_queries) > 1:
        print(f"[PLAN]    split into {len(trace.sub_queries)} searches: {trace.sub_queries}")
    if trace and trace.steps:
        _print_steps(container, trace, args.show_math)
    if trace:
        for note in trace.notes:
            print(f"[LOOP]    {note}")
        if trace.grounded is not None:
            print(f"[CHECK]   self-check verdict: {'grounded' if trace.grounded else 'rejected'} (attempts: {trace.attempts})")
    print(f"\n[ANSWER]\n{result.answer}\n")
    for c in result.citations:
        print(f"  source: {c.file_name} p.{c.page_number}")
    print("[PERSIST] conversation saved to data/service.sqlite")


def cmd_eval(args: argparse.Namespace) -> None:
    from evaluation import run_eval

    if not args.mock and not get_settings().gemini_api_key:
        sys.exit("GEMINI_API_KEY is empty. Copy .env.example to .env and set it (or use --mock for a smoke test).")
    argv = ["run_eval", "--dataset", args.dataset, "--config", args.config]
    argv.append("--mock" if args.mock else "--no-mock")
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.output:
        argv += ["--output", args.output]
    sys.argv = argv
    try:
        run_eval.main()
    except Exception as exc:  # a retired model or a bad key fails every question, so stop with the reason
        sys.exit(
            f"Evaluation stopped: {type(exc).__name__}: {str(exc)[:300]}\n"
            "If it says the model is not found or unavailable, set GEMINI_MODEL in .env (see .env.example)."
        )


def cmd_mcp(_: argparse.Namespace) -> None:
    import runpy

    runpy.run_module("app.mcp_server.server", run_name="__main__")


def cmd_load_sample(args: argparse.Namespace) -> None:
    """Index the PDF shipped in sample/ so ask and eval work straight away."""
    args.path = str(ROOT / "sample")
    args.course = DEFAULT_COURSE
    cmd_ingest(args)
    print('Try: python -m app.cli ask "What is the software crisis?" --show-math')


def cmd_reset(_: argparse.Namespace) -> None:
    data = Path(get_settings().chroma_dir).parent
    for sub in ("chroma", "chunks", "extracted", "pdfs"):
        shutil.rmtree(data / sub, ignore_errors=True)
    for name in ("ingested.json", "service.sqlite", "jobs.sqlite"):
        (data / name).unlink(missing_ok=True)
    print("data/ cleared.")


def main() -> None:
    quiet_libraries()  # no HTTP log lines (set SHINE_DEBUG=1 to see them)
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Shine AI standalone CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index a PDF or a folder of PDFs")
    p.add_argument("path")
    p.add_argument("--course", type=int, default=DEFAULT_COURSE)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question about the ingested PDFs")
    p.add_argument("question")
    p.add_argument("--course", type=int, default=DEFAULT_COURSE)
    p.add_argument("--show-math", action="store_true", help="print dot products, magnitudes and cosine scores")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="score the system on a ground-truth dataset")
    p.add_argument("--dataset", default="evaluation/sample_dataset.jsonl")
    p.add_argument("--config", default="hybrid_with_self_check")
    p.add_argument("--limit", type=int)
    p.add_argument("--output")
    p.add_argument("--mock", action="store_true", help="synthetic numbers, only for a quick smoke test")
    p.set_defaults(func=cmd_eval)

    sub.add_parser("mcp", help="start the student-data MCP server (mock data)").set_defaults(func=cmd_mcp)
    sub.add_parser("load-sample", help="ingest the sample PDF in sample/ (course 1)").set_defaults(func=cmd_load_sample)
    sub.add_parser("reset", help="delete everything under data/").set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
