
from unittest.mock import Mock

import pytest

from app.infrastructure.store.chroma_vector_store import ChromaVectorStore


@pytest.fixture
def mock_client():
    client = Mock()
    collection = Mock()

    client.get_or_create_collection.return_value = collection

    return client, collection


@pytest.fixture
def vector_store(mock_client):
    client, collection = mock_client

    store = ChromaVectorStore(client=client)

    return store, collection


def test_initializes_collection(mock_client):
    client, collection = mock_client

    ChromaVectorStore(client=client)

    client.get_or_create_collection.assert_called_once_with(
        name="document_chunks",
        metadata={"hnsw:space": "cosine"},
    )


def test_add_documents(vector_store):
    store, collection = vector_store

    store.add(
        ids=["chunk-1"],
        texts=["Introduction to Python"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"item_id": "item-1"}],
    )

    collection.upsert.assert_called_once_with(
        ids=["chunk-1"],
        documents=["Introduction to Python"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"item_id": "item-1"}],
    )


def test_add_empty_lists(vector_store):
    store, collection = vector_store

    store.add([], [], [], [])

    collection.upsert.assert_not_called()


def test_add_rejects_different_list_lengths(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.add(
            ids=["chunk-1"],
            texts=[],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"item_id": "item-1"}],
        )

    collection.upsert.assert_not_called()


def test_add_rejects_duplicate_ids(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.add(
            ids=["chunk-1", "chunk-1"],
            texts=["Text A", "Text B"],
            embeddings=[[0.1], [0.2]],
            metadatas=[
                {"item_id": "item-1"},
                {"item_id": "item-1"},
            ],
        )

    collection.upsert.assert_not_called()


def test_add_requires_item_id_metadata(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.add(
            ids=["chunk-1"],
            texts=["Some text"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"course_id": 1}],
        )

    collection.upsert.assert_not_called()


def test_delete_by_item_id(vector_store):
    store, collection = vector_store

    store.delete_by_item_id("item-1")

    collection.delete.assert_called_once_with(
        where={"item_id": "item-1"}
    )


def test_delete_rejects_empty_item_id(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.delete_by_item_id("")

    collection.delete.assert_not_called()


def test_search_returns_matches(vector_store):
    store, collection = vector_store

    collection.query.return_value = {
        "ids": [["chunk-1"]],
        "documents": [["Introduction to Python"]],
        "metadatas": [[{"item_id": "item-1"}]],
        "distances": [[0.15]],
    }

    result = store.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    assert result == [
        {
            "id": "chunk-1",
            "text": "Introduction to Python",
            "metadata": {"item_id": "item-1"},
            "distance": 0.15,
        }
    ]

    collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=5,
        where=None,
    )


def test_search_with_metadata_filter(vector_store):
    store, collection = vector_store

    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    store.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
        filters={"item_id": "item-1"},
    )

    collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=3,
        where={"item_id": "item-1"},
    )


def test_search_returns_empty_for_nonpositive_top_k(vector_store):
    store, collection = vector_store

    result = store.search(
        query_embedding=[0.1, 0.2],
        top_k=0,
    )

    assert result == []
    collection.query.assert_not_called()


def test_search_rejects_empty_embedding(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.search(
            query_embedding=[],
            top_k=5,
        )

    collection.query.assert_not_called()


def test_rejects_empty_collection_name(mock_client):
    client, _ = mock_client

    with pytest.raises(ValueError):
        ChromaVectorStore(
            client=client,
            collection_name=" ",
        )

    client.get_or_create_collection.assert_not_called()
