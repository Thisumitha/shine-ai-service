import logging

import pytesseract
from fastapi.testclient import TestClient

from app.config import Settings
from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
from app.main import create_app


def _not_found():
    raise pytesseract.TesseractNotFoundError()


def test_problem_when_tesseract_is_missing(monkeypatch):
    monkeypatch.setattr(pytesseract, "get_tesseract_version", _not_found)
    assert "not installed" in TesseractOcr().problem()


def test_problem_lists_missing_languages(monkeypatch):
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.3")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng", "osd"])
    assert "sin, tam" in TesseractOcr(language="eng+sin+tam").problem()


def test_no_problem_when_everything_is_installed(monkeypatch):
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.3")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng", "sin", "tam"])
    assert TesseractOcr(language="eng+sin").problem() is None


def test_startup_logs_a_warning_when_ocr_is_not_ready(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(TesseractOcr, "problem", lambda self: "Tesseract is not installed")
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.sqlite'}", pdf_dir=str(tmp_path / "pdfs"))
    with caplog.at_level(logging.WARNING), TestClient(create_app(settings)):
        pass
    assert "OCR not ready: Tesseract is not installed" in caplog.text
