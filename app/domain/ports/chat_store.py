from abc import ABC, abstractmethod

from app.domain.entities.citation import Citation
from app.domain.entities.feedback import Feedback
from app.domain.entities.message import Message
from app.domain.entities.session import ChatSession
from app.domain.entities.unanswered import UnansweredQuery


class ChatStore(ABC):
    """Abstract port for persisting conversational sessions, messages, feedback, and unanswerable questions."""

    @abstractmethod
    def create_session(self, user_id: int | str, course_id: int | str, title: str) -> ChatSession:
        """Create and return a new chat session."""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> ChatSession | None:
        """Retrieve a session by its unique ID."""
        ...

    @abstractmethod
    def list_sessions(self, user_id: int | str) -> list[ChatSession]:
        """List all chat sessions belonging to a specific user, ordered newest first."""
        ...

    @abstractmethod
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Citation] | None = None,
    ) -> Message:
        """Persist a message turn (user or assistant) and optional citations."""
        ...

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 10) -> list[Message]:
        """Return the most recent messages for a session in chronological order."""
        ...

    @abstractmethod
    def save_feedback(self, feedback: Feedback) -> None:
        """Store user feedback (+1 / -1) for an assistant message."""
        ...

    @abstractmethod
    def log_unanswered(self, unanswered: UnansweredQuery) -> None:
        """Log a question that was refused or could not be answered."""
        ...
