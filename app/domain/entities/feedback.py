from dataclasses import dataclass


@dataclass(frozen=True)
class Feedback:
    """User feedback for a generated message."""

    id: int | None
    message_id: str
    user_id: int | str
    rating: int  # +1 (helpful) or -1 (unhelpful)
    comment: str | None = None
    created_at: str | None = None

