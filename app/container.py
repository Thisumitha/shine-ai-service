from dataclasses import dataclass

from app.config import Settings, get_settings
from app.infrastructure.queue.job_store import JobStore
from app.infrastructure.queue.worker import Worker


@dataclass(frozen=True)
class Container:
    """Builds the long-lived objects once. Later phases add the embedder, LLM client, stores, etc."""

    settings: Settings
    job_store: JobStore
    worker: Worker


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    job_store = JobStore(settings.database_url)
    return Container(settings=settings, job_store=job_store, worker=Worker(job_store))
