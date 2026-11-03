import io

import pytesseract
from PIL import Image

from app.domain.ports.ocr_engine import OcrEngine


class TesseractOcr(OcrEngine):
    def __init__(self, language: str = "eng", tesseract_cmd: str = "") -> None:
        self.language = language
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def read_page(self, image_png: bytes) -> str:
        with Image.open(io.BytesIO(image_png)) as image:
            return pytesseract.image_to_string(image, lang=self.language)
