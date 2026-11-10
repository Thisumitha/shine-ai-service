import logging
from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_current_user
from app.api.schemas.chat import ChatRequest, ChatResponse, CitationSchema
from app.container import Container
from app.domain.entities.user import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    container: Container = Depends(get_container),
) -> ChatResponse:
    """Send a question to the grounded 4-stage AI Chat Agent.

    The agent executes:
    1. Observe (fetches conversation context)
    2. Reason (intent classification: COURSE_CONTENT, SMALL_TALK, OFF_TOPIC)
    3. Act (hybrid search, grounded answer generation, self-check verification)
    4. Persist (saves turn & citations to SQLite)
    """
    user_id = str(user.user_id) if user.user_id is not None else user.email
    logger.info("Chat request from user %s for session %s (course %s)", user_id, body.session_id, body.course_id)

    result = container.chat_agent.handle_message(
        user_id=user_id,
        course_id=body.course_id,
        message=body.message,
        session_id=body.session_id,
    )

    citations_out = [
        CitationSchema(
            source_id=c.source_id,
            file_name=c.file_name,
            page_number=c.page_number,
            snippet=c.snippet,
        )
        for c in result.citations
    ]

    is_refusal = (
        "not found in your course material" in result.answer.lower()
        or "couldn't find" in result.answer.lower()
        or "could not find" in result.answer.lower()
    )

    return ChatResponse(
        session_id=result.session_id,
        answer=result.answer,
        citations=citations_out,
        is_refusal=is_refusal,
    )
