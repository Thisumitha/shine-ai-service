import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_container, get_current_user
from app.api.schemas.chat import CitationSchema
from app.api.schemas.session import (
    ChatMessageSchema,
    SessionDetailResponse,
    SessionSummaryResponse,
)
from app.container import Container
from app.domain.entities.user import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Sessions"])


@router.get("/sessions", response_model=list[SessionSummaryResponse])
def list_user_sessions(
    course_id: int | None = Query(None, description="Optional filter by course ID"),
    user: CurrentUser = Depends(get_current_user),
    container: Container = Depends(get_container),
) -> list[SessionSummaryResponse]:
    """List all chat sessions belonging to the authenticated student."""
    user_id = str(user.user_id) if user.user_id is not None else user.email
    sessions = container.chat_store.list_sessions(user_id=user_id)

    if course_id is not None:
        sessions = [s for s in sessions if str(s.course_id) == str(course_id)]

    return [
        SessionSummaryResponse(
            session_id=s.session_id,
            user_id=str(s.user_id),
            course_id=str(s.course_id),
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session_detail(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    container: Container = Depends(get_container),
) -> SessionDetailResponse:
    """Retrieve the full conversation history and citations for a specific chat session."""
    session = container.chat_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found",
        )

    # Fetch dialogue history
    raw_messages = container.chat_store.get_history(session_id, limit=100)

    messages = [
        ChatMessageSchema(
            message_id=m.message_id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            citations=[
                CitationSchema(
                    source_id=c.source_id,
                    file_name=c.file_name,
                    page_number=c.page_number,
                    snippet=c.snippet,
                )
                for c in m.citations
            ],
            created_at=m.created_at,
        )
        for m in raw_messages
    ]

    return SessionDetailResponse(
        session_id=session.session_id,
        user_id=str(session.user_id),
        course_id=str(session.course_id),
        title=session.title,
        messages=messages,
    )
