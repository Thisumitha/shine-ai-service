import pymupdf

MIN_TEXT_CHARS = 25  # a page with fewer characters than this is treated as scanned


class PdfLoader:
    """Reads a PDF with PyMuPDF: page text, scanned-page check, and page images for OCR."""

    def __init__(self, path: str, render_dpi: int = 200) -> None:
        self.path = path
        self.render_dpi = render_dpi
        self.doc = pymupdf.open(path)

    def __enter__(self) -> "PdfLoader":
        return self

    def __exit__(self, *_) -> None:
        self.doc.close()

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def page_text(self, index: int) -> str:
        return self.doc[index].get_text("text")

    def is_scanned(self, index: int) -> bool:
        return len(self.page_text(index).strip()) < MIN_TEXT_CHARS

    def render_png(self, index: int) -> bytes:
        return self.doc[index].get_pixmap(dpi=self.render_dpi).tobytes("png")
