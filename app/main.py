import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import health, internal, parse
from app.config import Settings
from app.container import build_container

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        container.job_store.create_table()
        if problem := container.ocr.problem():
            logger.warning("OCR not ready: %s See docs/PDF_EXTRACTION.md or run start.ps1.", problem)
        container.worker.start()
        yield
        container.worker.stop()

    app = FastAPI(title="Shine AI Service", lifespan=lifespan)
    app.state.container = container
    app.include_router(health.router)
    app.include_router(internal.router)
    app.include_router(parse.router)
    return app


app = create_app()
