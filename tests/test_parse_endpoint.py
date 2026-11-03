import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.ingestion_service import IngestionService
from tests.pdf_samples import table_pdf, text_pdf

SECRET = {"X-Internal-Secret": "s3cret"}


def make_client(tmp_path):
    settings = Settings(
        internal_secret="s3cret",
        database_url=f"sqlite:///{tmp_path / 'test.sqlite'}",
        pdf_dir=str(tmp_path / "pdfs"),
        extracted_dir=str(tmp_path / "extracted"),
    )
    return TestClient(create_app(settings))


def test_parse_pdf_returns_pages_and_tables(tmp_path):
    pdf = table_pdf(tmp_path / "t.pdf")
    with make_client(tmp_path) as client:
        res = client.post(
            "/internal/parse-pdf", headers=SECRET, files={"file": ("t.pdf", open(pdf, "rb"), "application/pdf")}
        )
    assert res.status_code == 200
    body = res.json()
    assert body["page_count"] == 1
    assert "| Nimal | 80 | 75 |" in body["pages"][0]["tables"][0]


def test_parse_pdf_needs_the_secret_and_a_real_pdf(tmp_path):
    with make_client(tmp_path) as client:
        files = {"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")}
        assert client.post("/internal/parse-pdf", files=files).status_code == 401
        bad = {"file": ("a.pdf", b"not a pdf", "application/pdf")}
        assert client.post("/internal/parse-pdf", headers=SECRET, files=bad).status_code == 400
        broken = {"file": ("a.pdf", b"%PDF-1.4 broken", "application/pdf")}
        assert client.post("/internal/parse-pdf", headers=SECRET, files=broken).status_code == 422


def test_service_saves_parsed_pages_for_the_chunking_stage(tmp_path):
    with make_client(tmp_path) as client:
        service: IngestionService = client.app.state.container.ingestion
        status = service.process({"item_id": 5, "pdf_path": text_pdf(tmp_path / "n.pdf", ["Notes on algebra: solving linear equations step by step."])})
    saved = json.loads((tmp_path / "extracted" / "5.json").read_text(encoding="utf-8"))
    assert status == "EXTRACTED"
    assert "algebra" in saved["pages"][0]["text"]
