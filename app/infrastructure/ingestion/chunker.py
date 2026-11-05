import re

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import PageContent, ParsedDocument

TARGET = 900  # pack text up to about this many characters
MAX_SIZE = 1200  # a chunk is never longer than this
OVERLAP = 150  # the next chunk repeats up to this much of the previous one
MIN_SIZE = 150  # a smaller leftover is merged into the chunk before it

_SENTENCE_END = re.compile(r"(?<=[.?!।])\s+|\n")  # "।" ends sentences in Sinhala and Hindi


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for one very long sentence: cut at spaces, never inside a word."""
    parts = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _sentences(text: str, limit: int) -> list[str]:
    result = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if sentence:
            result.extend(_hard_split(sentence, limit) if len(sentence) > limit else [sentence])
    return result


def _units(text: str, limit: int) -> list[str]:
    """Paragraphs that fit as they are, otherwise their sentences."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend([paragraph] if len(paragraph) <= limit else _sentences(paragraph, limit))
    return units


def _tail(text: str, overlap: int) -> str:
    """The last whole sentences of a chunk that fit in `overlap` characters."""
    tail = ""
    for sentence in reversed(_sentences(text, overlap)):
        candidate = f"{sentence} {tail}".strip()
        if len(candidate) > overlap:
            break
        tail = candidate
    return tail


def _text_chunks(text: str, target: int, max_size: int, overlap: int, min_size: int) -> list[str]:
    chunks: list[list[str]] = []  # [full text, new text without the overlap]
    buffer, fresh = "", ""

    def flush() -> None:
        nonlocal buffer, fresh
        if not fresh:
            return
        if len(fresh) < min_size and chunks and len(chunks[-1][0]) + 2 + len(fresh) <= max_size:
            chunks[-1][0] += "\n\n" + fresh
            chunks[-1][1] += "\n\n" + fresh
        else:
            chunks.append([buffer, fresh])
        buffer, fresh = "", ""

    for unit in _units(text, max_size):
        if fresh and len(buffer) + 2 + len(unit) > target:
            previous = buffer
            flush()
            tail = _tail(previous, overlap)
            buffer = tail if tail and len(tail) + 2 + len(unit) <= max_size else ""
        buffer = f"{buffer}\n\n{unit}" if buffer else unit
        fresh = f"{fresh}\n\n{unit}" if fresh else unit
    flush()
    return [full for full, _ in chunks]


def _table_chunks(markdown: str, max_size: int) -> list[str]:
    """A table is one chunk. A long one is split by rows with the header repeated in each part."""
    if len(markdown) <= max_size:
        return [markdown]
    lines = markdown.split("\n")
    header, rows = "\n".join(lines[:2]), lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len(header) + len("\n".join(current)) + 1 + len(row) + 1 > max_size:
            parts.append(header + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current:
        parts.append(header + "\n" + "\n".join(current))
    return parts


def chunk_page(
    page: PageContent,
    target: int = TARGET,
    max_size: int = MAX_SIZE,
    overlap: int = OVERLAP,
    min_size: int = MIN_SIZE,
) -> list[tuple[str, bool]]:
    """Chunks of one page as (text, is_table). Text chunks come first, then the tables."""
    result = [(text, False) for text in _text_chunks(page.text, target, max_size, overlap, min_size)]
    for table in page.tables_markdown:
        result.extend((part, True) for part in _table_chunks(table, max_size))
    return result


def chunk_document(
    document: ParsedDocument,
    item_id: int,
    course_id: int,
    topic_id: int | None,
    title: str,
    **sizes: int,
) -> list[Chunk]:
    """Cut a parsed PDF into chunks. A chunk never spans two pages."""
    chunks: list[Chunk] = []
    for page in document.pages:
        for text, is_table in chunk_page(page, **sizes):
            chunks.append(
                Chunk(
                    item_id=item_id,
                    course_id=course_id,
                    topic_id=topic_id,
                    title=title,
                    page_number=page.page_number,
                    chunk_index=len(chunks),
                    text=text,
                    is_table=is_table,
                )
            )
    return chunks
