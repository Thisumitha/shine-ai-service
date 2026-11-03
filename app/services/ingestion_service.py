import json
from dataclasses import asdict
from pathlib import Path

from app.domain.entities.document import ParsedDocument
from app.infrastructure.ingestion.pipeline import PdfPipeline


class IngestionService:
    """Runs the extraction stage for one queued PDF and keeps the result for the chunking stage."""

    def __init__(self, pipeline: PdfPipeline, extracted_dir: str) -> None:
        self.pipeline = pipeline
        self.extracted_dir = Path(extracted_dir)

    def parse(self, pdf_path: str) -> ParsedDocument:
        return self.pipeline.parse(pdf_path)

    def process(self, job: dict) -> str:
        """Worker callback. Saves the parsed pages as JSON and returns the job's final status."""
        document = self.parse(job["pdf_path"])
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        out = self.extracted_dir / f"{job['item_id']}.json"
        out.write_text(json.dumps(asdict(document), ensure_ascii=False, indent=1), encoding="utf-8")
        return "EXTRACTED"  # becomes INDEXED once chunking and embedding exist
