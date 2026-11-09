
import pytest

from app.infrastructure.retrieval.dense_index import DenseRetriever


class FakeEmbedder:
    def __init__(self, embeddings=None, error=None):
        self.embeddings = (
            [[0.1, 0.2, 0.3]]
            if embeddings is None
            else embeddings
        )
        self.error = error
        self.received_texts = []

    def embed(self, texts):
        self.received_texts = texts

        if self.error:
            raise self.error

        return self.embeddings


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.search_calls = []

    def search(self, query_embedding, top_k, filters=None):
        self.search_calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "filters": filters,
            }
        )

        return self.results


def make_retriever(embedder=None, vector_store=None):
    embedder = embedder or FakeEmbedder()
    vector_store = vector_store or FakeVectorStore()

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    return retriever, embedder, vector_store


def test_search_returns_relevant_chunks():
    expected_results = [
        {
            "id": "12-0",
            "text": "Introduction to algebra",
            "metadata": {
                "item_id": 12,
                "course_id": 3,
            },
            "distance": 0.15,
        }
    ]

    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(results=expected_results)

    retriever, _, _ = make_retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.search(
        query="What is algebra?",
        course_ids={3, 5},
        top_k=5,
    )

    assert results == expected_results

    assert embedder.received_texts == [
        "What is algebra?"
    ]

    assert vector_store.search_calls == [
        {
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 5,
            "filters": {
                "course_id": {
                    "$in": [3, 5]
                }
            },
        }
    ]


def test_search_only_uses_approved_course_ids():
    retriever, _, vector_store = make_retriever()

    retriever.search(
        query="Explain Python",
        course_ids={7},
        top_k=3,
    )

    filters = vector_store.search_calls[0]["filters"]

    assert filters == {
        "course_id": {
            "$in": [7]
        }
    }


def test_search_returns_empty_when_no_courses_are_approved():
    retriever, embedder, vector_store = make_retriever()

    results = retriever.search(
        query="Explain Python",
        course_ids=set(),
        top_k=5,
    )

    assert results == []
    assert embedder.received_texts == []
    assert vector_store.search_calls == []


def test_search_returns_empty_for_nonpositive_top_k():
    retriever, embedder, vector_store = make_retriever()

    results = retriever.search(
        query="Explain Python",
        course_ids={3},
        top_k=0,
    )

    assert results == []
    assert embedder.received_texts == []
    assert vector_store.search_calls == []


def test_search_rejects_empty_query():
    retriever, embedder, vector_store = make_retriever()

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        retriever.search(
            query=" ",
            course_ids={3},
            top_k=5,
        )

    assert embedder.received_texts == []
    assert vector_store.search_calls == []


def test_search_rejects_invalid_course_ids():
    retriever, embedder, vector_store = make_retriever()

    with pytest.raises(
        ValueError,
        match="Course IDs must be positive integers",
    ):
        retriever.search(
            query="Explain Python",
            course_ids={0, 3},
            top_k=5,
        )

    assert embedder.received_texts == []
    assert vector_store.search_calls == []


def test_search_rejects_boolean_course_id():
    retriever, embedder, vector_store = make_retriever()

    with pytest.raises(
        ValueError,
        match="Course IDs must be positive integers",
    ):
        retriever.search(
            query="Explain Python",
            course_ids={True},
            top_k=5,
        )

    assert embedder.received_texts == []
    assert vector_store.search_calls == []


def test_search_rejects_embedding_count_mismatch():
    embedder = FakeEmbedder(embeddings=[])
    retriever, _, vector_store = make_retriever(
        embedder=embedder,
    )

    with pytest.raises(
        RuntimeError,
        match="Expected exactly one embedding",
    ):
        retriever.search(
            query="Explain Python",
            course_ids={3},
            top_k=5,
        )

    assert vector_store.search_calls == []


def test_search_does_not_call_vector_store_when_embedding_fails():
    embedder = FakeEmbedder(
        error=RuntimeError("Embedding service failed")
    )

    retriever, _, vector_store = make_retriever(
        embedder=embedder,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service failed",
    ):
        retriever.search(
            query="Explain Python",
            course_ids={3},
            top_k=5,
        )

    assert vector_store.search_calls == []


def test_search_rejects_empty_query_embedding():
    embedder = FakeEmbedder(
        embeddings=[[]]
    )

    retriever, _, vector_store = make_retriever(
        embedder=embedder,
    )

    with pytest.raises(
        RuntimeError,
        match="The query embedding cannot be empty",
    ):
        retriever.search(
            query="Explain Python",
            course_ids={3},
            top_k=5,
        )

    assert vector_store.search_calls == []
