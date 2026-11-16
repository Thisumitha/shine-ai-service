import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.domain.ports.chat_store import ChatStore
from app.domain.ports.embedder import Embedder
from app.domain.ports.llm_client import LLMClient
from app.domain.ports.ocr_engine import OcrEngine
from app.domain.ports.retriever import Retriever
from app.domain.ports.sparse_indexer import SparseIndexer
from app.domain.ports.vector_store import VectorStore
from app.services.agent.agent import ChatAgent
from app.services.answer_service import AnswerService
from app.services.self_check_service import SelfCheckService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Container:
    settings: Settings
    job_store: Any
    ocr: OcrEngine | None
    embedder: Embedder | None
    vector_store: VectorStore | None
    sparse_indexer: SparseIndexer | None
    retriever: Retriever | None
    ingestion: Any
    worker: Any
    chat_store: ChatStore
    llm_client: LLMClient
    answer_service: AnswerService
    self_check_service: SelfCheckService
    chat_agent: ChatAgent | None


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()

    job_store = None
    try:
        from app.infrastructure.queue.job_store import JobStore
        job_store = JobStore(settings.database_url)
    except Exception as exc:
        logger.debug("JobStore could not be loaded: %s", exc)

    ocr = None
    try:
        from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
        ocr = TesseractOcr(
            settings.ocr_language,
            settings.tesseract_cmd,
        )
    except Exception as exc:
        logger.debug("TesseractOcr could not be loaded: %s", exc)

    pipeline = None
    try:
        from app.infrastructure.ingestion.pipeline import PdfPipeline
        if ocr is not None:
            pipeline = PdfPipeline(ocr, settings.ocr_max_pages)
    except Exception as exc:
        logger.debug("PdfPipeline could not be loaded: %s", exc)

    embedder = None
    try:
        from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder
        embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
    except Exception as exc:
        logger.debug("GeminiEmbedder could not be loaded: %s", exc)

    vector_store = None
    try:
        from app.infrastructure.store.chroma_vector_store import ChromaVectorStore
        vector_store = ChromaVectorStore(
            persist_directory=settings.chroma_dir
        )
    except Exception as exc:
        logger.debug("ChromaVectorStore could not be loaded: %s", exc)

    sparse_indexer = None
    try:
        from app.infrastructure.retrieval.sparse_index import SparseIndex
        sparse_indexer = SparseIndex(
            chunks_dir=settings.chunks_dir
        )
    except Exception as exc:
        logger.debug("SparseIndex could not be loaded: %s", exc)

    dense_retriever = None
    if embedder is not None and vector_store is not None:
        try:
            from app.infrastructure.retrieval.dense_index import DenseRetriever
            dense_retriever = DenseRetriever(
                embedder=embedder,
                vector_store=vector_store,
            )
        except Exception as exc:
            logger.debug("DenseRetriever could not be loaded: %s", exc)

    retriever = None
    if dense_retriever is not None and sparse_indexer is not None:
        try:
            from app.infrastructure.retrieval.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_indexer=sparse_indexer,
            )
        except Exception as exc:
            logger.debug("HybridRetriever could not be loaded: %s", exc)

    ingestion = None
    if pipeline and embedder and vector_store and sparse_indexer:
        try:
            from app.services.ingestion_service import IngestionService
            ingestion = IngestionService(
                pipeline=pipeline,
                extracted_dir=settings.extracted_dir,
                chunks_dir=settings.chunks_dir,
                embedder=embedder,
                vector_store=vector_store,
                sparse_indexer=sparse_indexer,
            )
        except Exception as exc:
            logger.debug("IngestionService could not be loaded: %s", exc)

    worker = None
    if job_store is not None and ingestion is not None:
        try:
            from app.infrastructure.queue.worker import Worker
            worker = Worker(job_store, process=ingestion.process)
        except Exception as exc:
            logger.debug("Worker could not be loaded: %s", exc)

    # Member 3: Memory, LLM Client, Answer Generator, and Agent
    from app.infrastructure.llm.gemini_client import GeminiClient
    from app.infrastructure.memory.sqlite_chat_store import SqliteChatStore

    chat_store = SqliteChatStore(db_path=settings.sqlite_path)
    llm_client = GeminiClient(settings=settings)
    answer_service = AnswerService(llm_client=llm_client)
    self_check_service = SelfCheckService(llm_client=llm_client, chat_store=chat_store)

    student_data_stage = None
    try:
        from app.infrastructure.clients.spring_student_data import SpringStudentDataClient
        from app.services.agent.student_data import StudentDataStage
        from app.services.student_tools import StudentToolService

        student_data_stage = StudentDataStage(
            llm_client=llm_client,
            tools=StudentToolService(
                SpringStudentDataClient(
                    base_url=settings.spring_base_url,
                    internal_secret=settings.internal_secret,
                    timeout_seconds=settings.student_data_timeout_seconds,
                )
            ),
        )
    except Exception as exc:
        logger.warning("Student data tools could not be loaded: %s", exc)

    chat_agent = None
    if retriever is not None:
        chat_agent = ChatAgent.create(
            chat_store=chat_store,
            llm_client=llm_client,
            retriever=retriever,
            answer_service=answer_service,
            self_check_service=self_check_service,
            student_data_stage=student_data_stage,
        )

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
        chat_store=chat_store,
        llm_client=llm_client,
        answer_service=answer_service,
        self_check_service=self_check_service,
        chat_agent=chat_agent,
    )
