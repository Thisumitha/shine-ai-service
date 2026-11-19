import pytest

from app.domain.entities.chunk import Chunk
from app.infrastructure.retrieval.sparse_index import SparseIndex


@pytest.fixture
def sparse_index(tmp_path):
    return SparseIndex(chunks_dir=str(tmp_path))


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            item_id=1,
            course_id=10,
            topic_id=1,
            title="Python Basics",
            page_number=1,
            chunk_index=0,
            text="Python is a programming language used for data science.",
        ),
        Chunk(
            item_id=1,
            course_id=10,
            topic_id=1,
            title="Machine Learning",
            page_number=2,
            chunk_index=1,
            text="Machine learning uses algorithms to learn from data.",
        ),
        Chunk(
            item_id=2,
            course_id=20,
            topic_id=2,
            title="Database Systems",
            page_number=1,
            chunk_index=0,
            text="SQL databases store and manage structured information.",
        ),
    ]


def test_index_item_adds_chunks(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])

    assert len(sparse_index._chunks) == 2


def test_search_returns_relevant_chunks(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])

    results = sparse_index.search(
        query="Python programming",
        course_ids={10},
        top_k=5,
    )

    assert results
    assert results[0]["metadata"]["title"] == "Python Basics"


def test_search_only_returns_approved_courses(
    sparse_index,
    sample_chunks,
):
    sparse_index.index_item(1, sample_chunks[:2])
    sparse_index.index_item(2, sample_chunks[2:])

    results = sparse_index.search(
        query="database information",
        course_ids={10},
        top_k=5,
    )

    assert all(
        result["metadata"]["course_id"] == 10
        for result in results
    )


def test_search_returns_empty_without_approved_courses(
    sparse_index,
    sample_chunks,
):
    sparse_index.index_item(1, sample_chunks[:2])

    results = sparse_index.search(
        query="Python",
        course_ids=set(),
        top_k=5,
    )

    assert results == []


def test_search_returns_empty_for_nonpositive_top_k(
    sparse_index,
):
    assert sparse_index.search(
        "Python",
        {10},
        0,
    ) == []


def test_search_rejects_empty_query(sparse_index):
    with pytest.raises(ValueError):
        sparse_index.search("", {10}, 5)


def test_search_rejects_invalid_course_ids(sparse_index):
    with pytest.raises(ValueError):
        sparse_index.search("Python", {0}, 5)


def test_search_rejects_boolean_course_id(sparse_index):
    with pytest.raises(ValueError):
        sparse_index.search("Python", {True}, 5)


def test_delete_by_item_id(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])
    sparse_index.index_item(2, sample_chunks[2:])

    sparse_index.delete_by_item_id(1)

    assert len(sparse_index._chunks) == 1
    assert "2-0" in sparse_index._chunks


def test_reindex_replaces_old_chunks(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])

    new_chunk = Chunk(
        item_id=1,
        course_id=10,
        topic_id=1,
        title="Python Advanced",
        page_number=3,
        chunk_index=0,
        text="Advanced Python programming techniques.",
    )

    sparse_index.index_item(1, [new_chunk])

    assert len(sparse_index._chunks) == 1
    assert "1-1" not in sparse_index._chunks
    assert "1-0" in sparse_index._chunks


def test_search_returns_empty_for_empty_index(sparse_index):
    results = sparse_index.search(
        query="Python",
        course_ids={10},
        top_k=5,
    )

    assert results == []

def test_score_chunk_matches_the_search_score(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])
    hits = sparse_index.search("python data science", {10}, top_k=5)

    for hit in hits:
        scored = sparse_index.score_chunk("python data science", hit["id"])
        assert scored["cosine"] == pytest.approx(hit["score"])
        assert scored["dot_product"] == pytest.approx(hit["math"]["dot_product"])


def test_score_chunk_scores_a_chunk_outside_the_top_results(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])
    top_one = sparse_index.search("data", {10}, top_k=1)
    cut_off = next(h for h in sparse_index.search("data", {10}, top_k=5) if h["id"] != top_one[0]["id"])

    scored = sparse_index.score_chunk("data", cut_off["id"])

    assert scored["cosine"] == pytest.approx(cut_off["score"])
    assert scored["rank"] == 2  # it was cut by top_k=1, but the value is still available


def test_score_chunk_without_a_shared_word_has_no_rank(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])
    chunk_id = sparse_index.search("python", {10}, top_k=5)[0]["id"]

    scored = sparse_index.score_chunk("zebra", chunk_id)

    assert scored is None or (scored["cosine"] == 0 and scored["rank"] is None)


def test_score_chunk_unknown_id_is_none(sparse_index, sample_chunks):
    sparse_index.index_item(1, sample_chunks[:2])
    assert sparse_index.score_chunk("python", "no-such-chunk") is None
