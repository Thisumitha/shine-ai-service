from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A context chunk retrieved from vector or hybrid search."""

    chunk_id: str
    course_id: int | str
    file_name: str
    page_number: int
    content: str
    score: float = 0.0
    heading_path: str = ""

