import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import chat, feedback, health, internal, internal_chat, parse, sessions
from app.config import Settings
from app.container import Container, build_container

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
) -> FastAPI:
    if container is None:
        container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            container.job_store.create_table()
            if problem := container.ocr.problem():
                logger.warning("OCR not ready: %s See docs/PDF_EXTRACTION.md or run start.ps1.", problem)
            container.worker.start()
        except Exception as exc:
            logger.warning("Background worker / job_store skipped during startup: %s", exc)

        yield

        try:
            container.worker.stop()
        except Exception:
            pass

    app = FastAPI(title="Shine AI Service", lifespan=lifespan)
    app.state.container = container

    # Health & Utility
    app.include_router(health.router)
    app.include_router(internal.router)
    app.include_router(internal_chat.router)
    app.include_router(parse.router)

    # Member 3: AI Chat, Sessions & Feedback
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(feedback.router)

    return app


app = create_app()
