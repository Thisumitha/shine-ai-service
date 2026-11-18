"""The three RAG metrics, computed on live system output.

Context Relevance : mean cosine similarity between the question embedding and each retrieved chunk embedding.
Answer Relevance  : cosine similarity between the question embedding and the answer embedding.
Groundedness      : an LLM judge splits the answer into claims and marks each as supported or not by the
                    retrieved chunks; the score is supported / total, computed here (not by the model).

cosine(a, b) = (a . b) / (|a| |b|), computed explicitly here so the maths is visible.
"""

import json
import logging
import math
from typing import Any

from app.domain.ports.embedder import Embedder
from app.domain.ports.llm_client import LLMClient
from app.services.prompts.answer_prompts import build_context_block

logger = logging.getLogger(__name__)

# A separate prompt from the agent's own self-check, so the judge is not the verifier grading itself.
JUDGE_INSTRUCTION = """You are a strict faithfulness judge.
Split the ANSWER into its individual factual claims (ignore citation markers like [S1]).
For each claim decide whether the CONTEXT alone supports it. Use outside knowledge for nothing.
Output JSON only: {"claims": [{"claim": "...", "supported": true | false}]}"""


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


def groundedness(answer: str, chunks: list[dict[str, Any]], llm: LLMClient) -> float | None:
    """Share of the answer's claims the chunks support, or None when the judge itself failed.

    A failed judge call (timeout, 503, bad JSON) says nothing about the answer, so it is not scored as 0.
    """
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
            prompt=f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nCLAIMS JSON:",
            system_instruction=JUDGE_INSTRUCTION,
        )
        claims = json.loads(raw).get("claims", [])
        if not isinstance(claims, list) or not claims:
            logger.warning("Groundedness judge returned no claims; not scored.")
            return None
        # The score is computed here from the verdicts, not taken from the model.
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
        return supported / len(claims)
    except Exception as exc:
        logger.warning("Groundedness judge failed (%s); not scored.", exc)
        return None
