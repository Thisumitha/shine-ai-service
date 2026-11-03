import logging
import threading

logger = logging.getLogger(__name__)


def not_ready(job: dict) -> None:
    """Placeholder until the PDF reading code exists: every job fails with this message."""
    raise NotImplementedError("PDF processing is not implemented yet")


class Worker:
    """A background thread: takes waiting jobs one by one and runs `process` on each."""

    def __init__(self, store, process=not_ready, wait_seconds: float = 2.0) -> None:
        self.store = store
        self.process = process
        self.wait_seconds = wait_seconds
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=5)

    def loop(self) -> None:
        while not self.stopping.is_set():
            job = self.store.next_job()
            if job is None:
                self.stopping.wait(self.wait_seconds)  # nothing to do, sleep a little
                continue
            try:
                final_status = self.process(job) or "INDEXED"
                self.store.set_status(job["id"], final_status)
            except Exception as error:  # a bad PDF must not stop the worker
                logger.warning("Job %s failed: %s", job["id"], error)
                self.store.set_status(job["id"], "FAILED", str(error))
