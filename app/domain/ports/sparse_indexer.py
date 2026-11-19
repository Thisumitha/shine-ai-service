from abc import ABC, abstractmethod

from app.domain.entities.chunk import Chunk


class SparseIndexer(ABC):
    """Interface for sparse document indexing and retrieval."""

    @abstractmethod
    def index_item(self, item_id: int, chunks: list[Chunk]) -> None:
        """Add or replace all chunks belonging to an item."""
        ...

    @abstractmethod
    def delete_by_item_id(self, item_id: int) -> None:
        """Remove all chunks belonging to an item."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        course_ids: set[int],
        top_k: int,
    ) -> list[dict]:
        """Retrieve relevant chunks using sparse search."""
        ...

    def score_chunk(self, query: str, chunk_id: str, course_ids: set[int] | None = None) -> dict | None:
        """Keyword score of one chunk (cosine, rank, ...), or None when the index cannot say."""
        return None
