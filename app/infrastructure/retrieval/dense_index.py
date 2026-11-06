from app.domain.ports.embedder import Embedder
from app.domain.ports.retriever import Retriever
from app.domain.ports.vector_store import VectorStore


class DenseRetriever(Retriever):
    """Retrieve relevant document chunks using dense vector search."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        course_ids: set[int],
        top_k: int,
    ) -> list[dict]:
        """Search only within the courses approved for the user."""

        if top_k <= 0:
            return []

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not course_ids:
            return []

        if any(
            not isinstance(course_id, int)
            or isinstance(course_id, bool)
            or course_id <= 0
            for course_id in course_ids
        ):
            raise ValueError("Course IDs must be positive integers.")

        # Convert the user's query into an embedding.
        query_embeddings = self.embedder.embed([query.strip()])

        if len(query_embeddings) != 1:
            raise RuntimeError(
                "Expected exactly one embedding for the search query."
            )

        query_embedding = query_embeddings[0]

        if not query_embedding:
            raise RuntimeError("The query embedding cannot be empty.")

        # Restrict ChromaDB search to server-approved course IDs.
        filters = {
            "course_id": {
                "$in": sorted(course_ids)
            }
        }

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )