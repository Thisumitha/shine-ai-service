import pytest

from app.domain.ports.ocr_engine import OcrEngine
from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
from app.infrastructure.ingestion.pdf_loader import PdfLoader
from app.infrastructure.ingestion.pipeline import PdfPipeline
from app.infrastructure.ingestion.table_extractor import table_to_markdown
from app.config import get_settings
from evaluation.table_fidelity import find_tesseract
from tests.pdf_samples import scanned_pdf, table_pdf, text_pdf

TESSERACT = find_tesseract(get_settings().tesseract_cmd)


class FakeOcr(OcrEngine):
    def __init__(self) -> None:
        self.calls = 0

    def read_page(self, image_png: bytes) -> str:
        assert image_png.startswith(b"\x89PNG")
        self.calls += 1
        return "text read by ocr"


def test_loader_reads_text_and_flags_scanned_pages(tmp_path):
    with PdfLoader(text_pdf(tmp_path / "t.pdf", ["Photosynthesis turns light into sugar."])) as loader:
        assert loader.page_count == 1
        assert "Photosynthesis" in loader.page_text(0)
        assert not loader.is_scanned(0)
    with PdfLoader(scanned_pdf(tmp_path / "s.pdf")) as loader:
        assert loader.is_scanned(0)
        assert loader.render_png(0).startswith(b"\x89PNG")


def test_text_pdf_does_not_call_ocr(tmp_path):
    ocr = FakeOcr()
    doc = PdfPipeline(ocr).parse(text_pdf(tmp_path / "t.pdf", ["First page has enough text to count as text.", "Second page has enough text to count as text."]))
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "Second page" in doc.pages[1].text
    assert not any(p.used_ocr for p in doc.pages)
    assert ocr.calls == 0


def test_scanned_page_goes_through_ocr(tmp_path):
    ocr = FakeOcr()
    doc = PdfPipeline(ocr).parse(scanned_pdf(tmp_path / "s.pdf"))
    assert doc.pages[0].used_ocr
    assert doc.pages[0].text == "text read by ocr"
    assert ocr.calls == 1


def test_ocr_page_cap_is_respected(tmp_path):
    ocr = FakeOcr()
    doc = PdfPipeline(ocr, ocr_max_pages=0).parse(scanned_pdf(tmp_path / "s.pdf"))
    assert ocr.calls == 0
    assert not doc.pages[0].used_ocr


def test_scanned_page_without_ocr_engine_is_empty_not_a_crash(tmp_path):
    doc = PdfPipeline(None).parse(scanned_pdf(tmp_path / "s.pdf"))
    assert doc.pages[0].text == ""


def test_table_is_returned_as_markdown(tmp_path):
    doc = PdfPipeline(FakeOcr()).parse(table_pdf(tmp_path / "tb.pdf"))
    assert len(doc.pages[0].tables_markdown) == 1
    table = doc.pages[0].tables_markdown[0]
    assert "| Name | Maths | Science |" in table
    assert "| Nimal | 80 | 75 |" in table
    assert "Marks table" in doc.text


def test_table_to_markdown_skips_non_tables():
    assert table_to_markdown([["only one cell"]]) == ""
    assert table_to_markdown([["a", "b"]]) == ""
    assert table_to_markdown([[None, ""], ["", None]]) == ""


@pytest.mark.skipif(TESSERACT is None, reason="Tesseract is not installed")
def test_real_tesseract_reads_a_scanned_page(tmp_path):
    doc = PdfPipeline(TesseractOcr(tesseract_cmd=TESSERACT)).parse(scanned_pdf(tmp_path / "s.pdf", "HELLO SHINE"))
    assert "HELLO" in doc.pages[0].text.upper()


class BrokenOcr(OcrEngine):
    def read_page(self, image_png: bytes) -> str:
        raise RuntimeError("tesseract is not installed")


def test_failing_ocr_leaves_the_page_empty_instead_of_crashing(tmp_path):
    doc = PdfPipeline(BrokenOcr()).parse(scanned_pdf(tmp_path / "s.pdf"))
    assert doc.pages[0].text == ""
    assert not doc.pages[0].used_ocr


def test_find_tesseract_prefers_a_valid_configured_path(tmp_path):
    from app.infrastructure.ingestion.ocr_tesseract import find_tesseract

    fake = tmp_path / "tesseract.exe"
    fake.write_text("x")
    assert find_tesseract(str(fake)) == str(fake)
    assert find_tesseract(str(tmp_path / "missing.exe")) != str(tmp_path / "missing.exe")
