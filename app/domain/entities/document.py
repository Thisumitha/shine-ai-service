from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageContent:
    page_number: int  # 1-based, as printed in the PDF viewer
    text: str
    tables_markdown: list[str] = field(default_factory=list)
    used_ocr: bool = False

    @property
    def full_text(self) -> str:
        """Page text followed by its tables, ready for chunking later."""
        return "\n\n".join(part for part in [self.text, *self.tables_markdown] if part)


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    pages: list[PageContent]

    @property
    def text(self) -> str:
        return "\n\n".join(page.full_text for page in self.pages if page.full_text)
