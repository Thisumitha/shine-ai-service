import json
from dataclasses import asdict
from pathlib import Path

from app.domain.entities.document import ParsedDocument
from app.infrastructure.ingestion.chunker import chunk_document
from app.infrastructure.ingestion.pipeline import PdfPipeline


class IngestionService:
    """Runs the extraction and chunking stages for one queued PDF and saves both results."""

    def __init__(self, pipeline: PdfPipeline, extracted_dir: str, chunks_dir: str) -> None:
        self.pipeline = pipeline
        self.extracted_dir = Path(extracted_dir)
        self.chunks_dir = Path(chunks_dir)

    def parse(self, pdf_path: str) -> ParsedDocument:
        return self.pipeline.parse(pdf_path)

    def process(self, job: dict) -> str:
        """Worker callback. Saves the parsed pages and their chunks as JSON and returns the job's final status."""
        document = self.parse(job["pdf_path"])
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        out = self.extracted_dir / f"{job['item_id']}.json"
        out.write_text(json.dumps(asdict(document), ensure_ascii=False, indent=1), encoding="utf-8")

        chunks = chunk_document(
            document, job["item_id"], job.get("course_id"), job.get("topic_id"), job.get("title") or document.filename
        )
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        out = self.chunks_dir / f"{job['item_id']}.json"
        out.write_text(json.dumps([asdict(c) | {"chunk_id": c.chunk_id} for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        return "EXTRACTED"  # becomes INDEXED once embedding exists
