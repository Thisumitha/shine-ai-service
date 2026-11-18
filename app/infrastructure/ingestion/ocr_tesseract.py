import io
import shutil
from pathlib import Path

import pytesseract
from PIL import Image

from app.domain.ports.ocr_engine import OcrEngine, OcrPage
from app.infrastructure.ingestion.ocr_tables import Word, extract_tables, words_to_text


WINDOWS_TESSERACT = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")


def find_tesseract(configured: str = "") -> str | None:
    """Path to tesseract: the configured one, PATH, or the usual Windows install folder."""
    if configured and Path(configured).is_file():
        return configured
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    return str(WINDOWS_TESSERACT) if WINDOWS_TESSERACT.is_file() else None


class TesseractOcr(OcrEngine):
    def __init__(self, language: str = "eng", tesseract_cmd: str = "") -> None:
        self.language = language
        found = find_tesseract(tesseract_cmd)
        if found:
            pytesseract.pytesseract.tesseract_cmd = found

    def problem(self) -> str | None:
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            return "Tesseract is not installed or TESSERACT_CMD is wrong, so scanned pages will come back empty."
        try:
            installed = set(pytesseract.get_languages(config=""))
        except Exception as error:
            return f"Tesseract is installed but its language list could not be read: {error}"
        missing = [code for code in self.language.split("+") if code not in installed]
        if missing:
            return f"Tesseract language data missing for: {', '.join(missing)} (OCR_LANGUAGE={self.language})."
        return None

    def read_page(self, image_png: bytes) -> str:
        with Image.open(io.BytesIO(image_png)) as image:
            return pytesseract.image_to_string(image, lang=self.language)

    def read_page_structured(self, image_png: bytes) -> OcrPage:
        """Text plus tables: word boxes are regrouped into rows and columns (see ocr_tables.py)."""
        with Image.open(io.BytesIO(image_png)) as image:
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        words = [
            Word(
                text=data["text"][i].strip(),
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                order=i,
                paragraph=(data["block_num"][i], data["par_num"][i]),
                line=data["line_num"][i],
            )
            for i in range(len(data["text"]))
            if data["level"][i] == 5 and data["text"][i].strip() and float(data["conf"][i]) >= 0
        ]
        tables, rest = extract_tables(words)
        return OcrPage(text=words_to_text(rest), tables_markdown=tables)
