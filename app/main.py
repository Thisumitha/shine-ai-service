from fastapi import FastAPI

from app.api import health
from app.container import build_container


def create_app() -> FastAPI:
    app = FastAPI(title="Shine AI Service")
    app.state.container = build_container()
    app.include_router(health.router)
    return app


app = create_app()
