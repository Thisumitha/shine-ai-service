"""The three RAG metrics, computed on live system output.

Context Relevance : mean cosine similarity between the question embedding and each retrieved chunk embedding.
Answer Relevance  : cosine similarity between the question embedding and the answer embedding.
Groundedness      : an LLM judge reads the retrieved chunks and scores how many answer claims they support
                    (the same verifier prompt the agent uses for its self-check).

cosine(a, b) = (a . b) / (|a| |b|), computed explicitly here so the maths is visible.
"""

import json
import math
from typing import Any

from app.domain.ports.embedder import Embedder
from app.domain.ports.llm_client import LLMClient
from app.services.prompts.answer_prompts import build_context_block
from app.services.prompts.self_check_prompts import SELF_CHECK_INSTRUCTION, build_self_check_prompt


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def context_relevance(question: str, chunk_texts: list[str], embedder: Embedder) -> float:
    if not chunk_texts:
        return 0.0
    vectors = embedder.embed([question, *chunk_texts])
    return sum(cosine(vectors[0], v) for v in vectors[1:]) / len(chunk_texts)


def answer_relevance(question: str, answer: str, embedder: Embedder) -> float:
    vectors = embedder.embed([question, answer])
    return cosine(vectors[0], vectors[1])


def groundedness(answer: str, chunks: list[dict[str, Any]], llm: LLMClient) -> float:
    if not chunks:
        return 0.0
    context = build_context_block(
        [
            {
                "file_name": c.get("metadata", {}).get("title", "document"),
                "page_number": c.get("metadata", {}).get("page_number", 1),
                "heading_path": "",
                "content": c["text"],
            }
            for c in chunks
        ]
    )
    try:
        raw = llm.generate_json(
            prompt=build_self_check_prompt(answer=answer, context_block=context),
            system_instruction=SELF_CHECK_INSTRUCTION,
        )
        return max(0.0, min(1.0, float(json.loads(raw).get("score", 0.0))))
    except Exception:
        return 0.0
