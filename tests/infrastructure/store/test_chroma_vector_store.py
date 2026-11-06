
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
        metadatas=[{"item_id": 1}],
    )

    collection.upsert.assert_called_once_with(
        ids=["chunk-1"],
        documents=["Introduction to Python"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"item_id": 1}],
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
            metadatas=[{"item_id": 1}],
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
                {"item_id": 1},
                {"item_id": 1},
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

    store.delete_by_item_id(1)

    collection.delete.assert_called_once_with(
        where={"item_id": 1}
    )


def test_delete_rejects_invalid_item_id(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.delete_by_item_id(0)

    collection.delete.assert_not_called()


def test_delete_rejects_string_item_id(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.delete_by_item_id("1")

    collection.delete.assert_not_called()


def test_search_returns_matches(vector_store):
    store, collection = vector_store

    collection.query.return_value = {
        "ids": [["chunk-1"]],
        "documents": [["Introduction to Python"]],
        "metadatas": [[{"item_id": 1}]],
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
            "metadata": {"item_id": 1},
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
        filters={"item_id": 1},
    )

    collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=3,
        where={"item_id": 1},
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


# --------------------------------------------------
# replace_item tests
# --------------------------------------------------


def test_replace_item_adds_new_chunks_and_deletes_obsolete_ids(
    vector_store,
):
    store, collection = vector_store

    collection.get.return_value = {
        "ids": ["12-0", "12-1", "12-2"]
    }

    store.replace_item(
        item_id=12,
        ids=["12-0", "12-1"],
        texts=["Updated chunk 0", "Updated chunk 1"],
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        metadatas=[
            {"item_id": 12},
            {"item_id": 12},
        ],
    )

    collection.get.assert_called_once_with(
        where={"item_id": 12},
        include=[],
    )

    collection.upsert.assert_called_once_with(
        ids=["12-0", "12-1"],
        documents=["Updated chunk 0", "Updated chunk 1"],
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        metadatas=[
            {"item_id": 12},
            {"item_id": 12},
        ],
    )

    collection.delete.assert_called_once_with(
        ids=["12-2"]
    )


def test_replace_item_with_empty_chunks_deletes_all_old_ids(
    vector_store,
):
    store, collection = vector_store

    collection.get.return_value = {
        "ids": ["12-0", "12-1"]
    }

    store.replace_item(
        item_id=12,
        ids=[],
        texts=[],
        embeddings=[],
        metadatas=[],
    )

    collection.get.assert_called_once_with(
        where={"item_id": 12},
        include=[],
    )

    collection.upsert.assert_not_called()

    collection.delete.assert_called_once_with(
        ids=["12-0", "12-1"]
    )


def test_replace_item_with_no_obsolete_ids_does_not_delete(
    vector_store,
):
    store, collection = vector_store

    collection.get.return_value = {
        "ids": ["12-0", "12-1"]
    }

    store.replace_item(
        item_id=12,
        ids=["12-0", "12-1"],
        texts=["Updated chunk 0", "Updated chunk 1"],
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        metadatas=[
            {"item_id": 12},
            {"item_id": 12},
        ],
    )

    collection.upsert.assert_called_once()

    collection.delete.assert_not_called()


def test_replace_item_rejects_wrong_item_metadata(vector_store):
    store, collection = vector_store

    with pytest.raises(
        ValueError,
        match="match the item ID",
    ):
        store.replace_item(
            item_id=12,
            ids=["12-0"],
            texts=["New chunk"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"item_id": 99}],
        )

    collection.get.assert_not_called()
    collection.upsert.assert_not_called()
    collection.delete.assert_not_called()


def test_replace_item_rejects_mismatched_list_lengths(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.replace_item(
            item_id=12,
            ids=["12-0"],
            texts=[],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"item_id": 12}],
        )

    collection.get.assert_not_called()
    collection.upsert.assert_not_called()
    collection.delete.assert_not_called()


def test_replace_item_rejects_invalid_item_id(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.replace_item(
            item_id=0,
            ids=[],
            texts=[],
            embeddings=[],
            metadatas=[],
        )

    collection.get.assert_not_called()
    collection.upsert.assert_not_called()
    collection.delete.assert_not_called()


def test_replace_item_preserves_old_ids_when_upsert_fails(
    vector_store,
):
    store, collection = vector_store

    collection.get.return_value = {
        "ids": ["12-0", "12-1"]
    }

    collection.upsert.side_effect = RuntimeError(
        "ChromaDB upsert failed"
    )

    with pytest.raises(
        RuntimeError,
        match="ChromaDB upsert failed",
    ):
        store.replace_item(
            item_id=12,
            ids=["12-0"],
            texts=["Updated chunk"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"item_id": 12}],
        )

    collection.delete.assert_not_called()


def test_rejects_boolean_item_id_in_replace(vector_store):
    store, collection = vector_store

    with pytest.raises(ValueError):
        store.replace_item(
            item_id=True,
            ids=[],
            texts=[],
            embeddings=[],
            metadatas=[],
        )

    collection.get.assert_not_called()
    collection.upsert.assert_not_called()
    collection.delete.assert_not_called()
