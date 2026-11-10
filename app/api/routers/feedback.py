import logging
from fastapi import APIRouter, Depends, status

from app.api.deps import get_container, get_current_user
from app.api.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.container import Container
from app.domain.entities.feedback import Feedback
from app.domain.entities.user import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
def submit_feedback(
    body: FeedbackRequest,
    user: CurrentUser = Depends(get_current_user),
    container: Container = Depends(get_container),
) -> FeedbackResponse:
    """Submit rating (+1 or -1) and optional commentary on an assistant response."""
    user_id = str(user.user_id) if user.user_id is not None else user.email

    feedback = Feedback(
        id=None,
        message_id=body.message_id,
        user_id=user_id,
        rating=body.rating,
        comment=body.comment,
    )

    container.chat_store.save_feedback(feedback)
    logger.info("Saved feedback (%+d) from user %s for message %s", body.rating, user_id, body.message_id)

    return FeedbackResponse(status="success", message="Feedback recorded successfully")

