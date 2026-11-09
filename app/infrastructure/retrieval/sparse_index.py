
import json
from dataclasses import asdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.domain.entities.chunk import Chunk
from app.domain.ports.sparse_indexer import SparseIndexer


class SparseIndex(SparseIndexer):
    """TF-IDF based sparse retrieval with persistent chunk loading."""

    def __init__(self, chunks_dir: str = "./data/chunks") -> None:
        self.chunks_dir = Path(chunks_dir)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
        )

        self._chunks: dict[str, dict] = {}
        self._records: list[dict] = []
        self._matrix = None

        self._load_chunks()

    def _load_chunks(self) -> None:
        """Load previously saved chunks from JSON files."""

        self._chunks.clear()

        if not self.chunks_dir.exists():
            return

        for file_path in self.chunks_dir.glob("*.json"):
            try:
                records = json.loads(
                    file_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(records, list):
                continue

            for record in records:
                if not isinstance(record, dict):
                    continue

                chunk_id = record.get("chunk_id")

                if not chunk_id:
                    continue

                self._chunks[str(chunk_id)] = record

        self._fit()

    def _fit(self) -> None:
        """Build the TF-IDF matrix from all indexed chunks."""

        records = list(self._chunks.values())

        self._records = [
            record
            for record in records
            if isinstance(record.get("text"), str)
            and record["text"].strip()
        ]

        if not self._records:
            self._matrix = None
            return

        texts = [
            record["text"]
            for record in self._records
        ]

        try:
            self._matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # Handle chunks that contain no valid TF-IDF terms.
            self._records = []
            self._matrix = None

    def index_item(
        self,
        item_id: int,
        chunks: list[Chunk],
    ) -> None:
        """Replace existing chunks and index the new chunks."""

        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("Item ID must be an integer.")

        if item_id <= 0:
            raise ValueError("Item ID must be positive.")

        if any(chunk.item_id != item_id for chunk in chunks):
            raise ValueError("All chunks must belong to the given item ID.")

        # Remove old chunks belonging to this item.
        self._chunks = {
            chunk_id: record
            for chunk_id, record in self._chunks.items()
            if record.get("item_id") != item_id
        }

        # Add the new chunks.
        for chunk in chunks:
            record = asdict(chunk)
            record["chunk_id"] = chunk.chunk_id
            self._chunks[chunk.chunk_id] = record

        self._fit()

    def delete_by_item_id(self, item_id: int) -> None:
        """Delete all indexed chunks belonging to an item."""

        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("Item ID must be an integer.")

        if item_id <= 0:
            raise ValueError("Item ID must be positive.")

        self._chunks = {
            chunk_id: record
            for chunk_id, record in self._chunks.items()
            if record.get("item_id") != item_id
        }

        self._fit()

    def search(
        self,
        query: str,
        course_ids: set[int],
        top_k: int,
    ) -> list[dict]:
        """Search for relevant chunks in approved courses only."""

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

        if self._matrix is None:
            return []

        try:
            query_vector = self.vectorizer.transform([query.strip()])
        except ValueError:
            return []

        similarities = cosine_similarity(
            query_vector,
            self._matrix,
        ).flatten()

        results = []

        for index, score in enumerate(similarities):
            record = self._records[index]

            # Enforce server-approved course access.
            if record.get("course_id") not in course_ids:
                continue

            if score <= 0:
                continue

            metadata = {
                key: record[key]
                for key in (
                    "item_id",
                    "course_id",
                    "topic_id",
                    "title",
                    "page_number",
                    "chunk_index",
                    "is_table",
                )
                if key in record
            }

            results.append({
                "id": record["chunk_id"],
                "text": record["text"],
                "metadata": metadata,
                "score": float(score),
            })

        results.sort(
            key=lambda result: (
                -result["score"],
                str(result["id"]),
            ),
        )

        return results[:top_k]
