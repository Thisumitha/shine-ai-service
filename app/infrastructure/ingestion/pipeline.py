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

    def __init__(self, ocr: OcrEngine | None = None, ocr_max_pages: int = 200, render_dpi: int = 200) -> None:
        self.ocr = ocr
        self.ocr_max_pages = ocr_max_pages
        self.render_dpi = render_dpi

    def parse(self, path: str) -> ParsedDocument:
        texts, ocr_flags, tables = [], [], []
        ocr_used = 0
        with PdfLoader(path, self.render_dpi) as loader, TableExtractor(path) as extractor:
            for i in range(loader.page_count):
                text, used_ocr, ocr_tables = loader.page_text(i), False, []
                if loader.is_scanned(i) and self.ocr and ocr_used < self.ocr_max_pages:
                    try:
                        page = self.ocr.read_page_structured(loader.render_png(i))
                        text, ocr_tables, used_ocr = page.text, page.tables_markdown, True
                    except Exception as error:  # e.g. Tesseract not installed: keep the page empty, keep going
                        logger.warning("OCR failed on page %s, leaving it empty: %s", i + 1, error)
                    ocr_used += 1
                texts.append(clean_page(text))
                ocr_flags.append(used_ocr)
                tables.append(ocr_tables if used_ocr else self._tables(extractor, i))
        texts = remove_repeated_edges(texts)
        pages = [PageContent(i + 1, t, tb, o) for i, (t, tb, o) in enumerate(zip(texts, tables, ocr_flags))]
        return ParsedDocument(filename=Path(path).name, pages=pages)

    @staticmethod
    def _tables(extractor: TableExtractor, index: int) -> list[str]:
        try:
            return extractor.tables(index)
        except Exception as error:  # a table failure should not lose the page text
            logger.warning("Table extraction failed on page %s: %s", index + 1, error)
            return []
