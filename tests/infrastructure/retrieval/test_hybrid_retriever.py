
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



def test_hybrid_does_not_count_duplicate_sparse_results_twice():
    dense = FakeDenseRetriever([])

    sparse = FakeSparseIndexer([
        {"id": "A", "text": "Python", "metadata": {}, "score": 0.9},
        {"id": "A", "text": "Python", "metadata": {}, "score": 0.8},
    ])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1 / 61)
    assert results[0]["sparse_rank"] == 1
    assert results[0]["sources"] == ["sparse"]


def test_hybrid_does_not_count_duplicate_dense_results_twice():
    dense = FakeDenseRetriever([
        {"id": "A", "text": "Python", "metadata": {}},
        {"id": "A", "text": "Python", "metadata": {}},
    ])

    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1 / 61)
    assert results[0]["dense_rank"] == 1
    assert results[0]["sources"] == ["dense"]

def test_hybrid_strips_query_whitespace():
    dense = FakeDenseRetriever([])
    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    retriever.search("  Python programming  ", {10}, 5)

    assert dense.calls[0]["query"] == "Python programming"
    assert sparse.calls[0]["query"] == "Python programming"


def test_hybrid_returns_dense_results_when_sparse_is_empty():
    dense = FakeDenseRetriever([
        {"id": "A", "text": "Python", "metadata": {}, "distance": 0.1}
    ])
    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)

    assert len(results) == 1
    assert results[0]["id"] == "A"
    assert results[0]["sources"] == ["dense"]
    assert results[0]["sparse_rank"] is None


def test_hybrid_returns_sparse_results_when_dense_is_empty():
    dense = FakeDenseRetriever([])
    sparse = FakeSparseIndexer([
        {"id": "B", "text": "Python basics", "metadata": {}, "score": 0.9}
    ])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)

    assert len(results) == 1
    assert results[0]["id"] == "B"
    assert results[0]["sources"] == ["sparse"]
    assert results[0]["dense_rank"] is None


def test_hybrid_rejects_non_integer_top_k():
    retriever = HybridRetriever(
        FakeDenseRetriever(),
        FakeSparseIndexer(),
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("Python", {10}, 2.5)


def test_hybrid_rejects_non_integer_rrf_constant():
    with pytest.raises(ValueError, match="RRF constant"):
        HybridRetriever(
            FakeDenseRetriever(),
            FakeSparseIndexer(),
            rrf_k=2.5,
        )

def test_hybrid_assigns_consecutive_dense_ranks_after_duplicates():
    dense = FakeDenseRetriever([
        {"id": "A", "text": "Python", "metadata": {}},
        {"id": "A", "text": "Python duplicate", "metadata": {}},
        {"id": "B", "text": "Python basics", "metadata": {}},
    ])
    sparse = FakeSparseIndexer([])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)
    results_by_id = {result["id"]: result for result in results}

    assert results_by_id["A"]["dense_rank"] == 1
    assert results_by_id["B"]["dense_rank"] == 2
    assert results_by_id["B"]["score"] == pytest.approx(1 / 62)


def test_hybrid_assigns_consecutive_sparse_ranks_after_duplicates():
    dense = FakeDenseRetriever([])
    sparse = FakeSparseIndexer([
        {"id": "A", "text": "Python", "metadata": {}, "score": 0.9},
        {"id": "A", "text": "Python duplicate", "metadata": {}, "score": 0.8},
        {"id": "B", "text": "Python basics", "metadata": {}, "score": 0.7},
    ])

    retriever = HybridRetriever(dense, sparse)

    results = retriever.search("Python", {10}, 5)
    results_by_id = {result["id"]: result for result in results}

    assert results_by_id["A"]["sparse_rank"] == 1
    assert results_by_id["B"]["sparse_rank"] == 2
    assert results_by_id["B"]["score"] == pytest.approx(1 / 62)