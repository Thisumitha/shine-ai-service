import logging

from app.domain.entities.citation import Citation
from app.domain.entities.message import Message
from app.domain.ports.chat_store import ChatStore

logger = logging.getLogger(__name__)


class PersistStage:
    """Stage 4: Persists conversation turns and citations into SQLite ChatStore."""

    def __init__(self, chat_store: ChatStore) -> None:
        self.chat_store = chat_store

    def persist(
        self,
        session_id: str,
        query: str,
        answer: str,
        citations: list[Citation],
    ) -> Message:
        """Save user query and assistant response with citations to database."""
        # 1. Save user turn
        self.chat_store.save_message(
            session_id=session_id,
            role="user",
            content=query,
            citations=None,
        )

        # 2. Save assistant turn with citations
        assistant_message = self.chat_store.save_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations,
        )

        return assistant_message

