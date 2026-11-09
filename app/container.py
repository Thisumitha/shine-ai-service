from dataclasses import dataclass

from app.config import Settings, get_settings
from app.domain.ports.embedder import Embedder
from app.domain.ports.ocr_engine import OcrEngine
from app.domain.ports.retriever import Retriever
from app.domain.ports.sparse_indexer import SparseIndexer
from app.domain.ports.vector_store import VectorStore
from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
from app.infrastructure.ingestion.pipeline import PdfPipeline
from app.infrastructure.queue.job_store import JobStore
from app.infrastructure.queue.worker import Worker
from app.infrastructure.retrieval.dense_index import DenseRetriever
from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder
from app.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from app.infrastructure.retrieval.sparse_index import SparseIndex
from app.infrastructure.store.chroma_vector_store import ChromaVectorStore
from app.services.ingestion_service import IngestionService


@dataclass(frozen=True)
class Container:
    settings: Settings
    job_store: JobStore
    ocr: OcrEngine
    embedder: Embedder
    vector_store: VectorStore
    sparse_indexer: SparseIndexer
    retriever: Retriever
    ingestion: IngestionService
    worker: Worker


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()

    job_store = JobStore(settings.database_url)

    ocr = TesseractOcr(
        settings.ocr_language,
        settings.tesseract_cmd,
    )

    pipeline = PdfPipeline(ocr, settings.ocr_max_pages)

    embedder = GeminiEmbedder(api_key=settings.gemini_api_key)

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_dir
    )

    sparse_indexer = SparseIndex(
        chunks_dir=settings.chunks_dir
    )

    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_indexer=sparse_indexer,
    )

    ingestion = IngestionService(
        pipeline=pipeline,
        extracted_dir=settings.extracted_dir,
        chunks_dir=settings.chunks_dir,
        embedder=embedder,
        vector_store=vector_store,
        sparse_indexer=sparse_indexer,
    )

    worker = Worker(job_store, process=ingestion.process)

    return Container(
        settings=settings,
        job_store=job_store,
        ocr=ocr,
        embedder=embedder,
        vector_store=vector_store,
        sparse_indexer=sparse_indexer,
        retriever=retriever,
        ingestion=ingestion,
        worker=worker,
    )
