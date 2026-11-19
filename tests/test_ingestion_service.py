
import json

import pytest

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import PageContent, ParsedDocument
from app.services.ingestion_service import IngestionService


class FakePipeline:
    def parse(self, pdf_path):
        return ParsedDocument(
            filename="algebra.pdf",
            pages=[
                PageContent(
                    page_number=1,
                    text="Algebra notes for testing.",
                )
            ],
        )


class FakeEmbedder:
    def __init__(self, embeddings=None, error=None):
        self.embeddings = (
            [[0.1, 0.2]]
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
    def __init__(self, error=None):
        self.replace_calls = []
        self.error = error

    def replace_item(
        self,
        item_id,
        ids,
        texts,
        embeddings,
        metadatas,
    ):
        self.replace_calls.append(
            {
                "item_id": item_id,
                "ids": ids,
                "texts": texts,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

        if self.error:
            raise self.error


class FakeSparseIndexer:
    def __init__(self, error=None):
        self.deleted_item_ids = []
        self.index_calls = []
        self.error = error

    def delete_by_item_id(self, item_id):
        self.deleted_item_ids.append(item_id)

        if self.error:
            raise self.error

    def index_item(self, item_id, chunks):
        self.index_calls.append(
            {
                "item_id": item_id,
                "chunks": chunks,
            }
        )

        if self.error:
            raise self.error


def make_chunk():
    return Chunk(
        item_id=12,
        course_id=3,
        topic_id=2,
        title="Algebra",
        page_number=1,
        chunk_index=0,
        text="This is a test chunk about algebra.",
        is_table=False,
    )


def make_service(
    tmp_path,
    embedder=None,
    vector_store=None,
    sparse_indexer=None,
):
    embedder = embedder or FakeEmbedder()
    vector_store = vector_store or FakeVectorStore()
    sparse_indexer = sparse_indexer or FakeSparseIndexer()

    service = IngestionService(
        pipeline=FakePipeline(),
        extracted_dir=str(tmp_path / "extracted"),
        chunks_dir=str(tmp_path / "chunks"),
        embedder=embedder,
        vector_store=vector_store,
        sparse_indexer=sparse_indexer,
    )

    return (
        service,
        embedder,
        vector_store,
        sparse_indexer,
    )


def make_job():
    return {
        "item_id": 12,
        "course_id": 3,
        "topic_id": 2,
        "title": "Algebra",
        "pdf_path": "algebra.pdf",
    }


def test_process_embeds_and_stores_chunks(tmp_path, monkeypatch):
    chunk = make_chunk()

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [chunk],
    )

    service, embedder, vector_store, sparse_indexer = (
        make_service(tmp_path)
    )

    status = service.process(make_job())

    assert status == "INDEXED"

    assert embedder.received_texts == [chunk.text]

    # Dense indexing.
    assert len(vector_store.replace_calls) == 1

    replace_call = vector_store.replace_calls[0]

    assert replace_call["item_id"] == 12
    assert replace_call["ids"] == ["12-0"]
    assert replace_call["texts"] == [chunk.text]
    assert replace_call["embeddings"] == [[0.1, 0.2]]

    metadata = replace_call["metadatas"][0]

    assert metadata["item_id"] == 12
    assert metadata["course_id"] == 3
    assert metadata["topic_id"] == 2
    assert metadata["page_number"] == 1
    assert metadata["title"] == "Algebra"
    assert metadata["is_table"] is False

    # Sparse indexing.
    assert len(sparse_indexer.index_calls) == 1
    assert sparse_indexer.index_calls[0]["item_id"] == 12
    assert sparse_indexer.index_calls[0]["chunks"] == [chunk]

    # Extracted JSON file.
    extracted_file = tmp_path / "extracted" / "12.json"
    assert extracted_file.exists()

    extracted_data = json.loads(
        extracted_file.read_text(encoding="utf-8")
    )

    assert extracted_data["filename"] == "algebra.pdf"

    # Chunks JSON file.
    chunks_file = tmp_path / "chunks" / "12.json"
    assert chunks_file.exists()

    chunks_data = json.loads(
        chunks_file.read_text(encoding="utf-8")
    )

    assert len(chunks_data) == 1
    assert chunks_data[0]["chunk_id"] == "12-0"
    assert chunks_data[0]["text"] == chunk.text


def test_embedding_failure_does_not_update_indexes(
    tmp_path,
    monkeypatch,
):
    chunk = make_chunk()

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [chunk],
    )

    embedder = FakeEmbedder(
        error=RuntimeError("Embedding failed")
    )

    service, _, vector_store, sparse_indexer = make_service(
        tmp_path,
        embedder=embedder,
    )

    with pytest.raises(RuntimeError, match="Embedding failed"):
        service.process(make_job())

    assert vector_store.replace_calls == []
    assert sparse_indexer.index_calls == []
    assert sparse_indexer.deleted_item_ids == []

    assert not (tmp_path / "chunks" / "12.json").exists()


def test_embedding_count_mismatch_does_not_update_indexes(
    tmp_path,
    monkeypatch,
):
    chunk = make_chunk()

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [chunk],
    )

    embedder = FakeEmbedder(embeddings=[])

    service, _, vector_store, sparse_indexer = make_service(
        tmp_path,
        embedder=embedder,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding count does not match chunk count",
    ):
        service.process(make_job())

    assert vector_store.replace_calls == []
    assert sparse_indexer.index_calls == []
    assert sparse_indexer.deleted_item_ids == []


def test_empty_chunks_replace_old_indexes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [],
    )

    service, embedder, vector_store, sparse_indexer = (
        make_service(tmp_path)
    )

    status = service.process(make_job())

    assert status == "INDEXED"

    assert embedder.received_texts == []

    # Dense store receives an empty replacement.
    assert vector_store.replace_calls == [
        {
            "item_id": 12,
            "ids": [],
            "texts": [],
            "embeddings": [],
            "metadatas": [],
        }
    ]

    # Sparse entries are deleted.
    assert sparse_indexer.deleted_item_ids == [12]
    assert sparse_indexer.index_calls == []

    # Empty chunks file is saved.
    chunks_file = tmp_path / "chunks" / "12.json"

    assert json.loads(
        chunks_file.read_text(encoding="utf-8")
    ) == []


def test_dense_index_failure_does_not_update_sparse_index(
    tmp_path,
    monkeypatch,
):
    chunk = make_chunk()

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [chunk],
    )

    vector_store = FakeVectorStore(
        error=RuntimeError("Dense indexing failed")
    )

    service, _, _, sparse_indexer = make_service(
        tmp_path,
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Dense indexing failed",
    ):
        service.process(make_job())

    assert sparse_indexer.index_calls == []
    assert sparse_indexer.deleted_item_ids == []

    assert not (tmp_path / "chunks" / "12.json").exists()


def test_sparse_index_failure_does_not_save_chunks_json(
    tmp_path,
    monkeypatch,
):
    chunk = make_chunk()

    monkeypatch.setattr(
        "app.services.ingestion_service.chunk_document",
        lambda *args: [chunk],
    )

    sparse_indexer = FakeSparseIndexer(
        error=RuntimeError("Sparse indexing failed")
    )

    service, _, vector_store, _ = make_service(
        tmp_path,
        sparse_indexer=sparse_indexer,
    )

    with pytest.raises(
        RuntimeError,
        match="Sparse indexing failed",
    ):
        service.process(make_job())

    assert len(vector_store.replace_calls) == 1
    assert len(sparse_indexer.index_calls) == 1

    assert not (tmp_path / "chunks" / "12.json").exists()



def test_process_reports_each_stage_so_a_slow_step_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.ingestion_service.chunk_document", lambda *args: [make_chunk()])
    service, *_ = make_service(tmp_path)
    messages: list[str] = []

    service.process(make_job(), progress=messages.append)

    stages = [m.split()[0] for m in messages if m[0].isdigit()]
    assert stages == ["1/4", "2/4", "3/4", "4/4"]
    assert any("embedding 1 chunks with Gemini" in m for m in messages)


def test_process_works_without_a_progress_callback(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.ingestion_service.chunk_document", lambda *args: [make_chunk()])
    service, *_ = make_service(tmp_path)
    assert service.process(make_job()) == "INDEXED"  # the background worker calls it this way
