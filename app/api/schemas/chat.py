from pydantic import BaseModel, Field


class CitationSchema(BaseModel):
    """Citation metadata identifying source PDF, page number, and quote."""

    source_id: str = Field(default="", description="Marker or identifier of the source chunk, e.g. S1")
    file_name: str = Field(..., description="File name of the source PDF")
    page_number: int = Field(..., description="1-based slide or page number")
    snippet: str = Field(..., description="Quoted snippet from the course material")


class ChatRequest(BaseModel):
    """Inbound request payload for the AI chat endpoint."""

    session_id: str = Field(..., description="Unique conversation session ID")
    course_id: int = Field(..., description="Identifier of the course being studied")
    message: str = Field(..., min_length=1, description="Student's question or message")


class ChatResponse(BaseModel):
    """Outbound response payload returned by the AI chat endpoint."""

    session_id: str = Field(..., description="Unique conversation session ID")
    answer: str = Field(..., description="Grounded answer produced by the agent")
    citations: list[CitationSchema] = Field(
        default_factory=list,
        description="List of course slide citations backing the claims",
    )
    is_refusal: bool = Field(
        default=False,
        description="True if the query could not be answered from course materials",
    )
