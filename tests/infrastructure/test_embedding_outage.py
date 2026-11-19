import json
import logging
from unittest.mock import MagicMock

from google.genai import errors

from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder
from app.infrastructure.retrieval.hybrid_retriever import HybridRetriever


def api(code, message):
    return errors.APIError(code=code, response_json={"message": message})


DAILY = "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 2h30m1s."


def test_hybrid_search_uses_keywords_only_when_embedding_fails(caplog):
    dense = MagicMock()
    dense.search.side_effect = api(429, DAILY)
    sparse = MagicMock()
    sparse.search.return_value = [{"id": "c1", "text": "software crisis", "score": 2.0}]

    with caplog.at_level(logging.WARNING):
        hits = HybridRetriever(dense, sparse).search("software crisis", {1}, top_k=3)

    assert [h["id"] for h in hits] == ["c1"]
    assert hits[0]["dense_rank"] is None and hits[0]["sparse_rank"] == 1
    assert any("keyword search only" in r.getMessage() and "daily limit" in r.getMessage() for r in caplog.records)
    assert not any("{" in r.getMessage() for r in caplog.records)  # no raw API error text


def test_hybrid_search_does_not_hide_real_bugs():
    dense = MagicMock()
    dense.search.side_effect = KeyError("bug")
    retriever = HybridRetriever(dense, MagicMock())
    try:
        retriever.search("q", {1}, top_k=3)
    except KeyError:
        return
    raise AssertionError("a bug in the dense search must not be swallowed")


def test_embedder_does_not_retry_a_used_up_daily_quota(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = api(429, DAILY)
    embedder = GeminiEmbedder(client=client, model="m")
    monkeypatch.setattr(GeminiEmbedder._embed_batch.retry, "sleep", lambda _: None)

    try:
        embedder.embed(["hello"])
    except errors.APIError:
        pass
    assert client.models.embed_content.call_count == 1


def test_embedder_still_retries_a_per_minute_limit(monkeypatch):
    client = MagicMock()
    ok = MagicMock()
    ok.embeddings = [MagicMock(values=[0.1, 0.2])]
    client.models.embed_content.side_effect = [api(429, "EmbedContentRequestsPerMinute"), ok]
    monkeypatch.setattr(GeminiEmbedder._embed_batch.retry, "sleep", lambda _: None)

    assert GeminiEmbedder(client=client, model="m").embed(["hello"]) == [[0.1, 0.2]]
    assert client.models.embed_content.call_count == 2


def test_already_indexed_needs_registry_entry_chunks_file_and_same_course(tmp_path, monkeypatch):
    from app import cli
    from app.config import Settings

    (tmp_path / "7.json").write_text(json.dumps([{"course_id": 1}]), encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(chunks_dir=str(tmp_path)))

    assert cli._already_indexed({"a.pdf": 7}, "a.pdf", 1)
    assert not cli._already_indexed({"a.pdf": 7}, "a.pdf", 101)   # same file, different course
    assert not cli._already_indexed({"a.pdf": 7}, "b.pdf", 1)     # never registered
    assert not cli._already_indexed({"a.pdf": 8}, "a.pdf", 1)     # registered but no chunks file
