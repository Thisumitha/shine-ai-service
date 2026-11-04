import logging
from pathlib import Path

from app.domain.entities.document import PageContent, ParsedDocument
from app.domain.ports.ocr_engine import OcrEngine
from app.infrastructure.ingestion.pdf_loader import PdfLoader
from app.infrastructure.ingestion.table_extractor import TableExtractor
from app.infrastructure.ingestion.text_cleaner import clean_page, remove_repeated_edges

logger = logging.getLogger(__name__)


class PdfPipeline:
    """PDF file -> ParsedDocument: text per page, OCR for scanned pages, tables as Markdown, cleaned."""

    def __init__(self, ocr: OcrEngine | None = None, ocr_max_pages: int = 200) -> None:
        self.ocr = ocr
        self.ocr_max_pages = ocr_max_pages

    def parse(self, path: str) -> ParsedDocument:
        texts, ocr_flags, tables = [], [], []
        ocr_used = 0
        with PdfLoader(path) as loader, TableExtractor(path) as extractor:
            for i in range(loader.page_count):
                text, used_ocr = loader.page_text(i), False
                if loader.is_scanned(i) and self.ocr and ocr_used < self.ocr_max_pages:
                    text, used_ocr = self.ocr.read_page(loader.render_png(i)), True
                    ocr_used += 1
                texts.append(clean_page(text))
                ocr_flags.append(used_ocr)
                tables.append(self._tables(extractor, i, used_ocr))
        texts = remove_repeated_edges(texts)
        pages = [PageContent(i + 1, t, tb, o) for i, (t, tb, o) in enumerate(zip(texts, tables, ocr_flags))]
        return ParsedDocument(filename=Path(path).name, pages=pages)

    @staticmethod
    def _tables(extractor: TableExtractor, index: int, used_ocr: bool) -> list[str]:
        if used_ocr:  # a scanned page has no ruled-line data for pdfplumber to read
            return []
        try:
            return extractor.tables(index)
        except Exception as error:  # a table failure should not lose the page text
            logger.warning("Table extraction failed on page %s: %s", index + 1, error)
            return []
