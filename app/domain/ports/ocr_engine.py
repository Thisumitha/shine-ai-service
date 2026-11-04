from abc import ABC, abstractmethod


class OcrEngine(ABC):
    @abstractmethod
    def read_page(self, image_png: bytes) -> str: ...

    def problem(self) -> str | None:
        """A readable reason this engine cannot work right now, or None when it is ready."""
        return None
