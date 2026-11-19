
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
            # Keep the raw TF-IDF weights so vector magnitudes are visible.
            # Cosine is computed explicitly in search(): dot / (|q| * |d|).
            norm=None,
        )

        self._chunks: dict[str, dict] = {}
        self._records: list[dict] = []
        self._matrix = None
        self._doc_norms = None

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
            self._doc_norms = None
            return

        texts = [
            record["text"]
            for record in self._records
        ]

        try:
            self._matrix = self.vectorizer.fit_transform(texts)
            # |d| = sqrt(sum of squared TF-IDF weights) for every chunk.
            self._doc_norms = np.sqrt(
                np.asarray(self._matrix.multiply(self._matrix).sum(axis=1)).ravel()
            )
        except ValueError:
            # Handle chunks that contain no valid TF-IDF terms.
            self._records = []
            self._matrix = None
            self._doc_norms = None

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

        scored = self._similarities(query)
        if scored is None:
            return []
        dots, query_norm, similarities = scored

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
                "math": {
                    "dot_product": float(dots[index]),
                    "query_magnitude": query_norm,
                    "chunk_magnitude": float(self._doc_norms[index]),
                    "cosine": float(score),
                },
            })

        results.sort(
            key=lambda result: (
                -result["score"],
                str(result["id"]),
            ),
        )

        return results[:top_k]

    def _similarities(self, query: str):
        """Dot product, query magnitude and cosine of the query against every chunk (None if no usable words)."""
        try:
            query_vector = self.vectorizer.transform([query.strip()])
        except ValueError:
            return None

        # Dot product of the query with every chunk, then normalise by both magnitudes.
        dots = np.asarray((self._matrix @ query_vector.T).todense()).ravel()
        query_norm = float(np.sqrt(query_vector.multiply(query_vector).sum()))
        denominators = self._doc_norms * query_norm
        similarities = np.divide(
            dots,
            denominators,
            out=np.zeros_like(dots, dtype=float),
            where=denominators > 0,
        )
        return dots, query_norm, similarities

    def score_chunk(self, query: str, chunk_id: str, course_ids: set[int] | None = None) -> dict | None:
        """The keyword (TF-IDF) score of one chunk, even when it is not among the top results.

        `rank` is its position among the matching chunks of `course_ids` (default: its own course),
        or None when it shares no word with the query.
        """
        if self._matrix is None:
            return None
        index = next((i for i, record in enumerate(self._records) if record["chunk_id"] == chunk_id), None)
        scored = self._similarities(query)
        if index is None or scored is None:
            return None
        dots, query_norm, similarities = scored

        cosine = float(similarities[index])
        courses = course_ids or {self._records[index].get("course_id")}
        rank = None
        if cosine > 0:
            rank = 1 + sum(
                1
                for i, record in enumerate(self._records)
                if record.get("course_id") in courses and similarities[i] > cosine
            )
        return {
            "dot_product": float(dots[index]),
            "query_magnitude": query_norm,
            "chunk_magnitude": float(self._doc_norms[index]),
            "cosine": cosine,
            "rank": rank,
        }

    def explain_terms(self, query: str, chunk_id: str, top_n: int = 5) -> list[dict]:
        """Terms that contribute most to the dot product between a query and one chunk.

        Each term shows its IDF and its TF-IDF weight in the query and in the chunk;
        the dot product is the sum of the per-term products.
        """

        if self._matrix is None:
            return []

        index = next(
            (i for i, record in enumerate(self._records) if record["chunk_id"] == chunk_id),
            None,
        )
        if index is None:
            return []

        query_vector = self.vectorizer.transform([query.strip()])
        chunk_vector = self._matrix[index]
        shared = set(query_vector.indices) & set(chunk_vector.indices)
        terms = self.vectorizer.get_feature_names_out()
        query_row = dict(zip(query_vector.indices, query_vector.data))
        chunk_row = dict(zip(chunk_vector.indices, chunk_vector.data))

        rows = [
            {
                "term": str(terms[t]),
                "idf": float(self.vectorizer.idf_[t]),
                "query_weight": float(query_row[t]),
                "chunk_weight": float(chunk_row[t]),
                "product": float(query_row[t] * chunk_row[t]),
            }
            for t in shared
        ]
        rows.sort(key=lambda row: -row["product"])
        return rows[:top_n]
