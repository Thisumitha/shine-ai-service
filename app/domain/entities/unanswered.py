from dataclasses import dataclass


@dataclass(frozen=True)
class UnansweredQuery:
    """A user query that could not be answered or was refused due to grounding/relevance check."""

    id: int | None
    user_id: int | str
    course_id: int | str
    query: str
    reason: str  # e.g., "OUT_OF_SCOPE", "LOW_RELEVANCE", "UNSUPPORTED_CLAIMS"
    logged_at: str | None = None

