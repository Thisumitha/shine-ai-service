from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSession:
    """A conversational session scoped to a user and course."""

    session_id: str
    user_id: int | str
    course_id: int | str
    title: str
    created_at: str | None = None
    updated_at: str | None = None

