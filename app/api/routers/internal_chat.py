import logging
from fastapi import APIRouter, Depends

from app.api.routers.internal import check_secret, get_container
from app.api.schemas.internal_chat import (
    InternalChatRequest,
    InternalChatResponse,
    InternalSourceSchema,
)
from app.container import Container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/chat", response_model=InternalChatResponse, dependencies=[Depends(check_secret)])
def internal_chat_endpoint(
    body: InternalChatRequest,
    container: Container = Depends(get_container),
) -> InternalChatResponse:
    """Spring-facing internal chat endpoint.

    Spring has already verified student enrollment and provides the authorized course IDs.
    The AI service executes the 4-stage ChatAgent across all provided course IDs.
    """
    logger.info(
        "Internal chat request across %d courses (session_id=%s): %s",
        len(body.course_ids),
        body.session_id,
        body.question[:60],
    )

    # Spring calls AI service internally, user_id is internal
    result = container.chat_agent.handle_message(
        user_id=body.student_email or "spring_student",
        message=body.question,
        session_id=body.session_id,
        course_ids=body.course_ids,
        student_email=body.student_email,
    )

    is_refusal = (
        "not found in your course material" in result.answer.lower()
        or "couldn't find" in result.answer.lower()
        or "could not find" in result.answer.lower()
    )

    # If no course material matches or agent refused, return empty sources
    if is_refusal:
        sources_out: list[InternalSourceSchema] = []
    else:
        sources_out = [
            InternalSourceSchema(
                course_id=c.course_id if c.course_id is not None else (body.course_ids[0] if body.course_ids else 0),
                item_id=c.item_id if c.item_id is not None else 0,
                material_title=c.file_name,
                page=c.page_number,
                snippet=c.snippet,
            )
            for c in result.citations
        ]

    return InternalChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        sources=sources_out,
        is_refusal=is_refusal,
    )

