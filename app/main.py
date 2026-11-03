from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import health, internal
from app.config import Settings
from app.container import build_container


def create_app(settings: Settings | None = None) -> FastAPI:
    container = build_container(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        container.job_store.create_table()
        container.worker.start()
        yield
        container.worker.stop()

    app = FastAPI(title="Shine AI Service", lifespan=lifespan)
    app.state.container = container
    app.include_router(health.router)
    app.include_router(internal.router)
    return app


app = create_app()
