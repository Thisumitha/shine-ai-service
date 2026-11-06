from dataclasses import dataclass, field
from app.domain.entities.citation import Citation


@dataclass(frozen=True)
class Message:
    """A single turn in a chat conversation."""

    message_id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    citations: list[Citation] = field(default_factory=list)
    created_at: str | None = None

