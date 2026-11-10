from datetime import datetime
import json
import logging
import os
import sqlite3
import uuid

from app.domain.entities.citation import Citation
from app.domain.entities.feedback import Feedback
from app.domain.entities.message import Message
from app.domain.entities.session import ChatSession
from app.domain.entities.unanswered import UnansweredQuery
from app.domain.ports.chat_store import ChatStore

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SqliteChatStore(ChatStore):
    """SQLite-backed persistence for chat sessions, messages, feedback, and unanswered queries."""

    def __init__(self, db_path: str = "./data/service.sqlite") -> None:
        self.db_path = db_path
        self._mem_conn: sqlite3.Connection | None = None

        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        course_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        citations_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS chat_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        rating INTEGER NOT NULL,
                        comment TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS unanswered_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        course_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        logged_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
                    """
                )
        finally:
            if self._mem_conn is None:
                conn.close()

    def create_session(
        self,
        user_id: int | str,
        course_id: int | str,
        title: str,
        session_id: str | None = None,
    ) -> ChatSession:
        session_id = session_id or str(uuid.uuid4())
        created_at = now_iso()
        updated_at = created_at
        user_id_str = str(user_id)
        course_id_str = str(course_id)

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, user_id, course_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, user_id_str, course_id_str, title, created_at, updated_at),
                )
        finally:
            if self._mem_conn is None:
                conn.close()

        return ChatSession(
            session_id=session_id,
            user_id=user_id_str,
            course_id=course_id_str,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_session(self, session_id: str) -> ChatSession | None:
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT session_id, user_id, course_id, title, created_at, updated_at FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            if self._mem_conn is None:
                conn.close()

        if not row:
            return None

        return ChatSession(
            session_id=row["session_id"],
            user_id=row["user_id"],
            course_id=row["course_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_sessions(self, user_id: int | str) -> list[ChatSession]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT session_id, user_id, course_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (str(user_id),),
            ).fetchall()
        finally:
            if self._mem_conn is None:
                conn.close()

        return [
            ChatSession(
                session_id=row["session_id"],
                user_id=row["user_id"],
                course_id=row["course_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Citation] | None = None,
    ) -> Message:
        message_id = str(uuid.uuid4())
        created_at = now_iso()
        citations_list = citations or []
        citations_json = json.dumps([
            {
                "source_id": c.source_id,
                "file_name": c.file_name,
                "page_number": c.page_number,
                "snippet": c.snippet,
            }
            for c in citations_list
        ])

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO chat_messages (message_id, session_id, role, content, citations_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (message_id, session_id, role, content, citations_json, created_at),
                )
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (created_at, session_id),
                )
        finally:
            if self._mem_conn is None:
                conn.close()

        return Message(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            citations=citations_list,
            created_at=created_at,
        )

    def get_history(self, session_id: str, limit: int = 10) -> list[Message]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT message_id, session_id, role, content, citations_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            if self._mem_conn is None:
                conn.close()

        ordered_rows = list(reversed(rows))
        messages: list[Message] = []
        for r in ordered_rows:
            citations_raw = r["citations_json"]
            citations: list[Citation] = []
            if citations_raw:
                try:
                    parsed = json.loads(citations_raw)
                    citations = [
                        Citation(
                            source_id=item.get("source_id", ""),
                            file_name=item.get("file_name", ""),
                            page_number=item.get("page_number", 0),
                            snippet=item.get("snippet", ""),
                        )
                        for item in parsed
                    ]
                except Exception as exc:
                    logger.warning("Failed to decode citations_json for message %s: %s", r["message_id"], exc)

            messages.append(
                Message(
                    message_id=r["message_id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    citations=citations,
                    created_at=r["created_at"],
                )
            )

        return messages

    def save_feedback(self, feedback: Feedback) -> None:
        created_at = feedback.created_at or now_iso()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO chat_feedback (message_id, user_id, rating, comment, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        feedback.message_id,
                        str(feedback.user_id),
                        feedback.rating,
                        feedback.comment,
                        created_at,
                    ),
                )
        finally:
            if self._mem_conn is None:
                conn.close()

    def log_unanswered(self, unanswered: UnansweredQuery) -> None:
        logged_at = unanswered.logged_at or now_iso()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO unanswered_queries (user_id, course_id, query, reason, logged_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(unanswered.user_id),
                        str(unanswered.course_id),
                        unanswered.query,
                        unanswered.reason,
                        logged_at,
                    ),
                )
        finally:
            if self._mem_conn is None:
                conn.close()
