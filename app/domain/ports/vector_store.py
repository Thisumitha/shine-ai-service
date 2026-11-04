from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Interface for persistent vector storage."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        ...

    @abstractmethod
    def delete_by_item_id(self, item_id: str) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        ...