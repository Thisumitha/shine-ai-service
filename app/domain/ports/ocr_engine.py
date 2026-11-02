from abc import ABC, abstractmethod


class OcrEngine(ABC):
    @abstractmethod
    def read_page(self, image_png: bytes) -> str: ...
