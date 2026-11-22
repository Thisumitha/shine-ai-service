
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import ParsedDocument
from app.domain.ports.embedder import Embedder
from app.domain.ports.sparse_indexer import SparseIndexer
from app.domain.ports.vector_store import VectorStore
from app.infrastructure.ingestion.chunker import chunk_document
from app.infrastructure.ingestion.pipeline import PdfPipeline


class IngestionService:
    """Extract, chunk, embed, and index one uploaded PDF."""

    def __init__(
        self,
        pipeline: PdfPipeline,
        extracted_dir: str,
        chunks_dir: str,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_indexer: SparseIndexer,
    ) -> None:
        self.pipeline = pipeline
        self.extracted_dir = Path(extracted_dir)
        self.chunks_dir = Path(chunks_dir)
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_indexer = sparse_indexer

    def parse(self, pdf_path: str) -> ParsedDocument:
        return self.pipeline.parse(pdf_path)

    def _save_chunks(self, item_id: int, chunks: list[Chunk]) -> None:
        """Save chunks as JSON for persistence and future index rebuilding."""

        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        chunks_file = self.chunks_dir / f"{item_id}.json"

        chunks_file.write_text(
            json.dumps(
                [
                    asdict(chunk) | {"chunk_id": chunk.chunk_id}
                    for chunk in chunks
                ],
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    def process(self, job: dict, progress: Callable[[str], None] | None = None) -> str:
        """Process a PDF and update both dense and sparse indexes.

        `progress` receives a short message at each stage (used by the CLI so a slow step is visible).
        """

        def say(message: str) -> None:
            if progress:
                progress(message)

        item_id = job["item_id"]

        # 1. Extract PDF content.
        say("1/4 reading the PDF (OCR on image pages can be slow) ...")
        started = time.monotonic()
        document = self.parse(job["pdf_path"])
        ocr_pages = [p.page_number for p in document.pages if p.used_ocr]
        say(
            f"1/4 read {len(document.pages)} pages, {sum(len(p.tables_markdown) for p in document.pages)} tables"
            + (f", OCR on pages {ocr_pages}" if ocr_pages else "")
            + f" ({time.monotonic() - started:.0f}s)"
        )

        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        extracted_file = self.extracted_dir / f"{item_id}.json"

        extracted_file.write_text(
            json.dumps(
                asdict(document),
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

        # 2. Split the extracted content into chunks.
        chunks = chunk_document(
            document,
            item_id,
            job.get("course_id"),
            job.get("topic_id"),
            job.get("title") or document.filename,
        )

        say(f"2/4 cut into {len(chunks)} chunks")

        # 3. Generate embeddings when chunks exist.
        texts = [chunk.text for chunk in chunks]

        say(f"3/4 embedding {len(chunks)} chunks with Gemini (waits on the network; a slow connection can take a minute) ...")
        started = time.monotonic()
        embeddings = self.embedder.embed(texts) if chunks else []
        say(f"    embeddings received ({time.monotonic() - started:.0f}s)")

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # 4. Prepare ChromaDB metadata.
        metadatas = []

        for chunk in chunks:
            metadata = {
                "item_id": chunk.item_id,
                "course_id": chunk.course_id,
                "page_number": chunk.page_number,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "is_table": chunk.is_table,
            }

            if chunk.topic_id is not None:
                metadata["topic_id"] = chunk.topic_id

            metadatas.append(metadata)

        say("4/4 saving to Chroma and the TF-IDF index ...")

        # 5. Replace the item's dense vectors.
        # New vectors are upserted before obsolete IDs are deleted.
        self.vector_store.replace_item(
            item_id=item_id,
            ids=[chunk.chunk_id for chunk in chunks],
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # 6. Replace the sparse index entries.
        if chunks:
            self.sparse_indexer.index_item(item_id, chunks)
        else:
            self.sparse_indexer.delete_by_item_id(item_id)

        # 7. Save chunks for persistence and future index rebuilding.
        self._save_chunks(item_id, chunks)
        say("4/4 stored in Chroma and the TF-IDF index")

        return "INDEXED"
