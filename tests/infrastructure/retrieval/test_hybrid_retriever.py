
import pytest

from app.infrastructure.retrieval.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, course_ids, top_k):
        self.calls.append({
            "query": query,
            "course_ids": course_ids,
            "top_k": top_k,
        })
        return self.results[:top_k]


class FakeSparseIndexer:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, query, course_ids, top_k):
        self.calls.append({
            "query": query,
            "course_ids": course_ids,
            "top_k": top_k,
        })
        return self.results[:top_k]


@pytest.fixture
def sample_results():
    dense = [
        {
            "id": "1-0",
            "text": "Python programming basics",
            "metadata": {"course_id": 10},
            "distance": 0.1,
        },
        {
            "id": "1-1",
            "text": "Machine learning concepts",
            "metadata": {"course_id": 10},
            "distance": 0.2,
        },
    ]

    sparse = [
        {
            "id": "1-1",
            "text": "Machine learning concepts",
            "metadata": {"course_id": 10},
            "score": 0.9,
        },
        {
            "id": "1-0",
            "text": "Python programming basics",
            "metadata": {"course_id": 10},
            "score": 0.7,
        },
    ]

    return dense, sparse


def test_hybrid_combines_dense_and_sparse_results(sample_results):
    dense_results, sparse_results = sample_results

    dense = FakeDenseRetriever(dense_results)
    sparse = FakeSparseIndexer(sparse_results)

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search(
        query="Python programming",
        course_ids={10},
        top_k=5,
    )

    assert len(results) == 2

    # Both retrieval methods contributed to each result.
    assert all(
        result["sources"] == ["dense", "sparse"]
        for result in results
    )

    # Both methods returned the same two chunks.
    assert {result["id"] for result in results} == {"1-0", "1-1"}


def test_hybrid_uses_rrf_ranking():
    dense = FakeDenseRetriever([
        {"id": "A", "text": "A", "metadata": {}},
        {"id": "B", "text": "B", "metadata": {}},
    ])

    sparse = FakeSparseIndexer([
        {"id": "B", "text": "B", "metadata": {}},
        {"id": "A", "text": "A", "metadata": {}},
    ])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("test", {10}, 2)

    # Both have equal combined RRF scores.
    assert results[0]["score"] == pytest.approx(
        results[1]["score"]
    )


def test_hybrid_returns_top_k_results():
    dense = FakeDenseRetriever([
        {"id": "A", "text": "A", "metadata": {}},
        {"id": "B", "text": "B", "metadata": {}},
        {"id": "C", "text": "C", "metadata": {}},
    ])

    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("test", {10}, 2)

    assert len(results) == 2


def test_hybrid_passes_approved_course_ids(sample_results):
    dense_results, sparse_results = sample_results

    dense = FakeDenseRetriever(dense_results)
    sparse = FakeSparseIndexer(sparse_results)

    retriever = HybridRetriever(dense, sparse)

    retriever.search("test", {10, 20}, 5)

    assert dense.calls[0]["course_ids"] == {10, 20}
    assert sparse.calls[0]["course_ids"] == {10, 20}


def test_hybrid_returns_empty_without_approved_courses():
    dense = FakeDenseRetriever()
    sparse = FakeSparseIndexer()

    retriever = HybridRetriever(dense, sparse)

    assert retriever.search("test", set(), 5) == []

    assert dense.calls == []
    assert sparse.calls == []


def test_hybrid_returns_empty_for_nonpositive_top_k():
    dense = FakeDenseRetriever()
    sparse = FakeSparseIndexer()

    retriever = HybridRetriever(dense, sparse)

    assert retriever.search("test", {10}, 0) == []

    assert dense.calls == []
    assert sparse.calls == []


def test_hybrid_rejects_empty_query():
    retriever = HybridRetriever(
        FakeDenseRetriever(),
        FakeSparseIndexer(),
    )

    with pytest.raises(ValueError, match="Search query"):
        retriever.search("", {10}, 5)


def test_hybrid_rejects_invalid_course_ids():
    retriever = HybridRetriever(
        FakeDenseRetriever(),
        FakeSparseIndexer(),
    )

    with pytest.raises(ValueError, match="Course IDs"):
        retriever.search("test", {0}, 5)


def test_hybrid_rejects_boolean_course_id():
    retriever = HybridRetriever(
        FakeDenseRetriever(),
        FakeSparseIndexer(),
    )

    with pytest.raises(ValueError, match="Course IDs"):
        retriever.search("test", {True}, 5)


def test_hybrid_rejects_invalid_rrf_constant():
    with pytest.raises(ValueError, match="RRF constant"):
        HybridRetriever(
            FakeDenseRetriever(),
            FakeSparseIndexer(),
            rrf_k=0,
        )


def test_hybrid_returns_empty_when_both_retrievers_return_empty():
    dense = FakeDenseRetriever([])
    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search(
        query="Python",
        course_ids={10},
        top_k=5,
    )

    assert results == []
    assert len(dense.calls) == 1
    assert len(sparse.calls) == 1


def test_hybrid_preserves_original_dense_and_sparse_scores():
    dense = FakeDenseRetriever([
        {
            "id": "A",
            "text": "Python basics",
            "metadata": {"course_id": 10},
            "distance": 0.15,
        }
    ])

    sparse = FakeSparseIndexer([
        {
            "id": "A",
            "text": "Python basics",
            "metadata": {"course_id": 10},
            "score": 0.85,
        }
    ])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search(
        query="Python",
        course_ids={10},
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["dense_distance"] == 0.15
    assert results[0]["sparse_score"] == 0.85
    assert results[0]["sources"] == ["dense", "sparse"]
