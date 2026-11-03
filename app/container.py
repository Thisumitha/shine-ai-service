from dataclasses import dataclass

from app.config import Settings, get_settings
from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
from app.infrastructure.ingestion.pipeline import PdfPipeline
from app.infrastructure.queue.job_store import JobStore
from app.infrastructure.queue.worker import Worker
from app.services.ingestion_service import IngestionService


@dataclass(frozen=True)
class Container:
    """Builds the long-lived objects once. Later phases add the embedder, LLM client, stores, etc."""

    settings: Settings
    job_store: JobStore
    ingestion: IngestionService
    worker: Worker


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    job_store = JobStore(settings.database_url)
    ocr = TesseractOcr(settings.ocr_language, settings.tesseract_cmd)
    pipeline = PdfPipeline(ocr, settings.ocr_max_pages)
    ingestion = IngestionService(pipeline, settings.extracted_dir)
    worker = Worker(job_store, process=ingestion.process)
    return Container(settings=settings, job_store=job_store, ingestion=ingestion, worker=worker)
