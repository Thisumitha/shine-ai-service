from pydantic import BaseModel, Field

from app.api.schemas.chat import CitationSchema


class ChatMessageSchema(BaseModel):
    """A single dialogue turn within a chat session."""

    message_id: str = Field(..., description="Unique ID of the message")
    session_id: str = Field(..., description="ID of the parent session")
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")
    citations: list[CitationSchema] = Field(
        default_factory=list,
        description="Slide citations associated with this turn",
    )
    created_at: str | None = Field(None, description="ISO-8601 creation timestamp")


class SessionSummaryResponse(BaseModel):
    """Metadata summary of a chat session."""

    session_id: str = Field(..., description="Unique conversation session ID")
    user_id: str = Field(..., description="User ID of the session owner")
    course_id: str = Field(..., description="Course ID this session is scoped to")
    title: str = Field(..., description="Title of the session")
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")


class SessionDetailResponse(BaseModel):
    """Full detail of a chat session including message history."""

    session_id: str = Field(..., description="Unique conversation session ID")
    user_id: str = Field(..., description="User ID of the session owner")
    course_id: str = Field(..., description="Course ID this session is scoped to")
    title: str = Field(..., description="Title of the session")
    messages: list[ChatMessageSchema] = Field(
        default_factory=list,
        description="Chronological message history in this session",
    )

