import logging

from app.domain.entities.message import Message
from app.domain.ports.chat_store import ChatStore

logger = logging.getLogger(__name__)


class ObserveStage:
    """Stage 1: Observes conversational context by loading session history or creating a new session."""

    def __init__(self, chat_store: ChatStore) -> None:
        self.chat_store = chat_store

    def observe(
        self,
        user_id: int | str,
        course_id: int | str,
        session_id: str | None = None,
        limit: int = 6,
    ) -> tuple[str, list[Message]]:
        """Return the active session_id and the recent history in chronological order."""
        active_session_id = session_id

        if active_session_id:
            existing = self.chat_store.get_session(active_session_id)
            if not existing:
                logger.info("Session %s not found. Creating a fresh session.", active_session_id)
                new_session = self.chat_store.create_session(
                    user_id=user_id,
                    course_id=course_id,
                    title="Study Session",
                    session_id=active_session_id,
                )
                return new_session.session_id, []

            history = self.chat_store.get_history(active_session_id, limit=limit)
            return active_session_id, history

        # No session_id provided: create a new session
        new_session = self.chat_store.create_session(
            user_id=user_id,
            course_id=course_id,
            title="Study Session",
        )
        return new_session.session_id, []

