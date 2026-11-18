"""State carried through one agent turn, and the trace returned to the caller.

The trace is the explainability record of a turn: what the agent decided (Reason), what it
retrieved with which scores (Act), and whether the self-check accepted the answer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedHit:
    """One chunk returned by the retriever, with the scores that ranked it."""

    chunk_id: str
    file_name: str
    page_number: int
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_distance: float | None = None
    sparse_math: dict[str, float] | None = None  # dot_product, query_magnitude, chunk_magnitude, cosine

    @classmethod
    def from_raw(cls, item: dict[str, Any]) -> "RetrievedHit":
        metadata = item.get("metadata") or {}
        return cls(
            chunk_id=str(item.get("id", "")),
            file_name=str(metadata.get("title") or metadata.get("file_name") or "Course Document"),
            page_number=int(metadata.get("page_number", 1)),
            rrf_score=float(item.get("score", 0.0)),
            dense_rank=item.get("dense_rank"),
            sparse_rank=item.get("sparse_rank"),
            dense_distance=item.get("dense_distance", item.get("distance")),
            sparse_math=item.get("sparse_math") or item.get("math"),
        )


@dataclass(frozen=True)
class RetrievalStep:
    """One retrieval action: which query was searched, on which attempt, and what came back."""

    sub_query: str
    attempt: int
    hits: tuple[RetrievedHit, ...] = ()


@dataclass
class AgentTrace:
    """Everything the agent observed, decided and did during one turn."""

    intent: str = ""
    reason: str = ""
    sub_queries: list[str] = field(default_factory=list)
    steps: list[RetrievalStep] = field(default_factory=list)
    attempts: int = 0
    grounded: bool | None = None  # None when no answer was generated (small talk, no context)
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.notes.append(text)
