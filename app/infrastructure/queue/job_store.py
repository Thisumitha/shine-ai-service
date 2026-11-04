from datetime import datetime

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, delete, insert, select, update

metadata = MetaData()

jobs = Table(
    "index_jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("item_id", Integer, nullable=False),
    Column("course_id", Integer, nullable=False),
    Column("topic_id", Integer),
    Column("title", String(500)),
    Column("pdf_path", String(500)),
    Column("status", String(20)),  # QUEUED -> EXTRACTING -> INDEXED or FAILED
    Column("error", Text),
    Column("created_at", String(40)),
    Column("updated_at", String(40)),
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobStore:
    """The queue: one row per PDF in the index_jobs table (MySQL). Each job is a plain dict."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)

    def create_table(self) -> None:
        metadata.create_all(self.engine)

    def add(self, item_id, course_id, topic_id, title, pdf_path) -> dict:
        with self.engine.begin() as db:
            # a new upload of the same item replaces the one still waiting
            db.execute(delete(jobs).where(jobs.c.item_id == item_id, jobs.c.status == "QUEUED"))
            result = db.execute(
                insert(jobs).values(
                    item_id=item_id,
                    course_id=course_id,
                    topic_id=topic_id,
                    title=title,
                    pdf_path=pdf_path,
                    status="QUEUED",
                    created_at=now(),
                    updated_at=now(),
                )
            )
            return self._get(db, result.inserted_primary_key[0])

    def next_job(self) -> dict | None:
        """Take the oldest waiting job and mark it EXTRACTING."""
        with self.engine.begin() as db:
            row = db.execute(select(jobs).where(jobs.c.status == "QUEUED").order_by(jobs.c.id).limit(1)).first()
            if row is None:
                return None
            db.execute(update(jobs).where(jobs.c.id == row.id).values(status="EXTRACTING", updated_at=now()))
            return self._get(db, row.id)

    def set_status(self, job_id: int, status: str, error: str | None = None) -> None:
        with self.engine.begin() as db:
            db.execute(update(jobs).where(jobs.c.id == job_id).values(status=status, error=error, updated_at=now()))

    def latest(self, item_id: int) -> dict | None:
        with self.engine.begin() as db:
            row = db.execute(select(jobs).where(jobs.c.item_id == item_id).order_by(jobs.c.id.desc()).limit(1)).first()
            return row._asdict() if row else None

    def remove(self, item_id: int) -> list[str]:
        """Delete the item's jobs and return their PDF paths so the files can be deleted too."""
        with self.engine.begin() as db:
            paths = [r.pdf_path for r in db.execute(select(jobs.c.pdf_path).where(jobs.c.item_id == item_id))]
            db.execute(delete(jobs).where(jobs.c.item_id == item_id))
            return paths

    def _get(self, db, job_id: int) -> dict:
        return db.execute(select(jobs).where(jobs.c.id == job_id)).first()._asdict()
