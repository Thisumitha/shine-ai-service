from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One searchable passage of a course PDF. It never spans two pages, so the page citation is exact."""

    item_id: int
    course_id: int
    topic_id: int | None
    title: str
    page_number: int  # 1-based
    chunk_index: int  # position inside the item, from 0
    text: str
    is_table: bool = False

    @property
    def chunk_id(self) -> str:
        """Stable id, so re-indexing an item can replace its old chunks."""
        return f"{self.item_id}-{self.chunk_index}"


@dataclass(frozen=True)
class RetrievedChunk:
    """A context chunk retrieved from vector or hybrid search, scored for relevance."""

    chunk_id: str
    course_id: int | str
    file_name: str
    page_number: int
    content: str
    score: float = 0.0
    heading_path: str = ""

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float = 0.0, heading_path: str = "") -> "RetrievedChunk":
        """Convert a stored Chunk into a scored RetrievedChunk for answer generation."""
        return cls(
            chunk_id=chunk.chunk_id,
            course_id=chunk.course_id,
            file_name=chunk.title,
            page_number=chunk.page_number,
            content=chunk.text,
            score=score,
            heading_path=heading_path,
        )
