
from app.domain.ports.retriever import Retriever
from app.domain.ports.sparse_indexer import SparseIndexer


class HybridRetriever(Retriever):
    """Combine dense and sparse search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_indexer: SparseIndexer,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("RRF constant must be positive.")

        self.dense_retriever = dense_retriever
        self.sparse_indexer = sparse_indexer
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        course_ids: set[int],
        top_k: int,
    ) -> list[dict]:
        """Combine dense and sparse results into one ranked list."""

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

        clean_query = query.strip()

        # Retrieve candidates from both search methods.
        dense_results = self.dense_retriever.search(
            query=clean_query,
            course_ids=course_ids,
            top_k=top_k,
        )

        sparse_results = self.sparse_indexer.search(
            query=clean_query,
            course_ids=course_ids,
            top_k=top_k,
        )

        combined = {}

        # Add dense ranking contributions.
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "score": 0.0,
                    "sources": [],
                    "dense_rank": None,
                    "sparse_rank": None,
                }

            combined[chunk_id]["score"] += (
                1 / (self.rrf_k + rank)
            )
            combined[chunk_id]["dense_rank"] = rank
            combined[chunk_id]["sources"].append("dense")

            if "distance" in result:
                combined[chunk_id]["dense_distance"] = result["distance"]

        # Add sparse ranking contributions.
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result["id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "score": 0.0,
                    "sources": [],
                    "dense_rank": None,
                    "sparse_rank": None,
                }

            combined[chunk_id]["score"] += (
                1 / (self.rrf_k + rank)
            )
            combined[chunk_id]["sparse_rank"] = rank
            combined[chunk_id]["sources"].append("sparse")

            if "score" in result:
                combined[chunk_id]["sparse_score"] = result["score"]

        # Return the highest RRF scores first.
        results = sorted(
            combined.values(),
            key=lambda result: (-result["score"], str(result["id"])),
        )

        return results[:top_k]
