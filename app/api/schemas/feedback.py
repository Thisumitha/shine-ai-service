from pydantic import BaseModel, Field, field_validator


class FeedbackRequest(BaseModel):
    """User rating and optional commentary for an assistant response."""

    message_id: str = Field(..., description="ID of the message being evaluated")
    rating: int = Field(..., description="+1 for helpful/positive, -1 for unhelpful/negative")
    comment: str | None = Field(None, max_length=1000, description="Optional text feedback")

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: int) -> int:
        if value not in (1, -1):
            raise ValueError("Rating must be either 1 (positive) or -1 (negative)")
        return value


class FeedbackResponse(BaseModel):
    """Response confirming feedback persistence."""

    status: str = Field(default="success", description="Status indicator")
    message: str = Field(
        default="Feedback recorded successfully",
        description="Friendly confirmation message",
    )

