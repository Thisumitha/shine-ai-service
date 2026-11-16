from pydantic import BaseModel, ConfigDict, Field


class InternalChatRequest(BaseModel):
    """Inbound request payload for the Spring-facing internal chat endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, description="Student's query or prompt")
    course_ids: list[int] = Field(
        ...,
        alias="courseIds",
        description="List of positive course IDs the student is enrolled in / allowed to search",
    )
    session_id: str | None = Field(
        default=None,
        alias="sessionId",
        description="Optional session ID for conversational memory across turns",
    )
    student_email: str | None = Field(
        default=None,
        alias="studentEmail",
        description="Email of the student, verified by Spring from the JWT. Used for personal-record questions",
    )


class InternalSourceSchema(BaseModel):
    """Source reference matching Spring backend integration expectations."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    course_id: int = Field(..., alias="courseId", description="Course ID of the material")
    item_id: int = Field(..., alias="itemId", description="Unique material item ID")
    material_title: str = Field(..., alias="materialTitle", description="Title of the course material/PDF")
    page: int = Field(..., description="1-based page number")
    snippet: str | None = Field(default=None, description="Excerpt snippet backing the claim")


class InternalChatResponse(BaseModel):
    """Outbound response payload returned to the Spring backend."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    answer: str = Field(..., description="Grounded answer produced by the agent")
    session_id: str = Field(..., alias="sessionId", description="Active conversation session ID")
    sources: list[InternalSourceSchema] = Field(
        default_factory=list,
        description="List of course slide citations backing the claims",
    )
    is_refusal: bool = Field(
        default=False,
        description="True if query could not be answered from course materials",
    )

