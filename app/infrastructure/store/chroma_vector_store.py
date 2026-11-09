
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

        self.client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=persist_directory)
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

    def replace_item(
        self,
        item_id: int,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Replace an item's chunks and remove obsolete chunk IDs."""

        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("Item ID must be an integer.")

        if item_id <= 0:
            raise ValueError("Item ID must be positive.")

        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length.")

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts must be non-empty strings.")

        if any(metadata.get("item_id") != item_id for metadata in metadatas):
            raise ValueError("Every metadata entry must match the item ID.")

        existing = self.collection.get(
            where={"item_id": item_id},
            include=[],
        )

        existing_ids = set(existing["ids"])
        new_ids = set(ids)

        # Add new chunks before deleting obsolete chunks.
        if ids:
            self.add(ids, texts, embeddings, metadatas)

        obsolete_ids = sorted(existing_ids - new_ids)

        if obsolete_ids:
            self.collection.delete(ids=obsolete_ids)

    def delete_by_item_id(self, item_id: int) -> None:
        """Delete all chunks belonging to an item."""

        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError("Item ID must be an integer.")

        if item_id <= 0:
            raise ValueError("Item ID must be positive.")

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
            matches.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches
