import io

import pytesseract
from PIL import Image

from app.domain.ports.ocr_engine import OcrEngine


class TesseractOcr(OcrEngine):
    def __init__(self, language: str = "eng", tesseract_cmd: str = "") -> None:
        self.language = language
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

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
