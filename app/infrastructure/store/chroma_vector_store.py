
import chromadb

from app.domain.ports.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Persistent vector storage using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "document_chunks",
        client=None,
    ):
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        # Allow fake clients during unit testing.
        self.client = client or chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add or update document chunks."""

        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length.")

        if not ids:
            return

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts must be non-empty strings.")

        if any("item_id" not in metadata for metadata in metadatas):
            raise ValueError("Every metadata entry must contain item_id.")

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_item_id(self, item_id: str) -> None:
        """Delete all chunks belonging to an item."""

        if not item_id.strip():
            raise ValueError("Item ID cannot be empty.")

        self.collection.delete(
            where={"item_id": item_id}
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """Find the most similar document chunks."""

        if top_k <= 0:
            return []

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
        )

        matches = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append({
                "id": doc_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            })

        return matches
