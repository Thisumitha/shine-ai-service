from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OcrPage:
    """What OCR read from one page image: running text, plus any tables rebuilt as Markdown."""

    text: str
    tables_markdown: list[str] = field(default_factory=list)


class OcrEngine(ABC):
    @abstractmethod
    def read_page(self, image_png: bytes) -> str: ...

    def read_page_structured(self, image_png: bytes) -> OcrPage:
        """Text and tables of a page image. Engines that cannot find tables return plain text only."""
        return OcrPage(text=self.read_page(image_png))

    def problem(self) -> str | None:
        """A readable reason this engine cannot work right now, or None when it is ready."""
        return None
