import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.infrastructure.queue.job_store import JobStore
from app.infrastructure.queue.worker import Worker
from app.main import create_app

SECRET = {"X-Internal-Secret": "s3cret"}
PDF = b"%PDF-1.4\n%fake pdf body\n"
FORM = {"itemId": "7", "courseId": "3", "topicId": "2", "title": "Algebra notes"}


def make_settings(tmp_path, secret="s3cret"):
    # tests use a throw-away SQLite file so no MySQL server is needed
    return Settings(
        internal_secret=secret,
        database_url=f"sqlite:///{tmp_path / 'test.sqlite'}",
        pdf_dir=str(tmp_path / "pdfs"),
        max_pdf_mb=1,
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


def upload(client, headers=SECRET, content=PDF, form=FORM):
    return client.post("/internal/index", headers=headers, data=form, files={"file": ("a.pdf", content, "application/pdf")})


def test_upload_is_accepted_and_saved(client, tmp_path):
    res = upload(client)
    assert res.status_code == 202
    job = res.json()
    assert (job["item_id"], job["course_id"], job["topic_id"], job["title"]) == (7, 3, 2, "Algebra notes")
    assert (tmp_path / "pdfs" / "7.pdf").read_bytes() == PDF


def test_wrong_or_missing_secret_is_rejected(client):
    assert upload(client, headers={}).status_code == 401
    assert upload(client, headers={"X-Internal-Secret": "wrong"}).status_code == 401


def test_empty_secret_setting_rejects_everything(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, secret=""))) as c:
        assert upload(c, headers={}).status_code == 401


def test_bad_files_and_missing_fields(client):
    assert upload(client, content=b"not a pdf").status_code == 400
    assert upload(client, content=b"%PDF" + b"0" * (1024 * 1024 + 1)).status_code == 413
    assert upload(client, form={"courseId": "3"}).status_code == 422


def test_job_fails_until_pipeline_exists(client):
    upload(client)
    for _ in range(50):
        job = client.get("/internal/index/7", headers=SECRET).json()
        if job["status"] in ("INDEXED", "FAILED"):
            break
        time.sleep(0.1)
    assert job["status"] == "FAILED"
    assert "not implemented" in job["error"]


def test_delete_removes_job_and_file(client, tmp_path):
    upload(client)
    assert client.delete("/internal/index/7", headers=SECRET).status_code == 204
    assert client.get("/internal/index/7", headers=SECRET).status_code == 404
    assert not (tmp_path / "pdfs" / "7.pdf").exists()


def test_reupload_replaces_waiting_job(tmp_path):
    store = JobStore(f"sqlite:///{tmp_path / 'q.sqlite'}")
    store.create_table()
    store.add(1, 1, None, "v1", "a.pdf")
    second = store.add(1, 1, None, "v2", "a.pdf")
    assert store.next_job()["id"] == second["id"]
    assert store.next_job() is None


def test_worker_runs_jobs_in_order_and_survives_a_failure(tmp_path):
    store = JobStore(f"sqlite:///{tmp_path / 'q.sqlite'}")
    store.create_table()
    done = []

    def process(job):
        done.append(job["item_id"])
        if job["item_id"] == 1:
            raise ValueError("bad pdf")

    for item in (1, 2, 3):
        store.add(item, 1, None, f"t{item}", "x.pdf")
    worker = Worker(store, process, wait_seconds=0.05)
    worker.start()
    for _ in range(50):
        if store.latest(3)["status"] == "INDEXED":
            break
        time.sleep(0.05)
    worker.stop()
    assert done == [1, 2, 3]
    assert store.latest(1)["status"] == "FAILED" and store.latest(1)["error"] == "bad pdf"
    assert store.latest(2)["status"] == "INDEXED"
