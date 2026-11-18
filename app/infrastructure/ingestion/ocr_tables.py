"""Rebuild tables from OCR word boxes.

Tesseract reads a table column by column, so "Stand-alone applications ... Description ..." loses the link
between a row's cells. Here the word boxes are grouped back into rows, split into cells at wide gaps, and a
run of rows that line up in columns becomes a Markdown table. Words outside tables are returned untouched,
in Tesseract's own reading order.

Pure functions on `Word` boxes (no Tesseract needed), so they can be unit-tested with made-up coordinates.
"""

from dataclasses import dataclass
from statistics import median

from app.infrastructure.ingestion.table_extractor import table_to_markdown

ROW_TOLERANCE = 0.6  # a word joins a row when its centre is within this many word heights of the row
CELL_GAP = 1.0  # a horizontal gap wider than this many word heights starts a new cell
MAX_ROW_GAP = 2.5  # a vertical gap wider than this many word heights ends a table
MAX_LABEL_ROWS = 2  # single-cell rows allowed in a row (a wrapped header line, a section label)
MIN_TABLE_ROWS = 3
MIN_MULTI_CELL_ROWS = 2


@dataclass(frozen=True)
class Word:
    text: str
    left: int
    top: int
    width: int
    height: int
    order: int  # position in Tesseract's reading order
    paragraph: tuple[int, int] = (0, 0)  # (block, paragraph) it belongs to
    line: int = 0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def centre_y(self) -> float:
        return self.top + self.height / 2


@dataclass(frozen=True)
class Cell:
    text: str
    left: int
    right: int
    words: tuple[Word, ...]


def group_rows(words: list[Word]) -> list[list[Word]]:
    """Words on the same visual line, left to right, top to bottom."""
    if not words:
        return []
    tolerance = ROW_TOLERANCE * median(w.height for w in words)
    rows: list[list[Word]] = []
    for word in sorted(words, key=lambda w: w.centre_y):
        if rows and abs(word.centre_y - sum(w.centre_y for w in rows[-1]) / len(rows[-1])) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda w: w.left) for row in rows]


def split_cells(row: list[Word], gap: float) -> list[Cell]:
    """Cut a row into cells wherever the space between two words is wider than `gap`."""
    cells: list[list[Word]] = [[row[0]]]
    for word in row[1:]:
        if word.left - cells[-1][-1].right > gap:
            cells.append([word])
        else:
            cells[-1].append(word)
    return [Cell(" ".join(w.text for w in c), c[0].left, c[-1].right, tuple(c)) for c in cells]


def _row_top(row: list[Word]) -> float:
    return min(w.top for w in row)


def _row_bottom(row: list[Word]) -> float:
    return max(w.top + w.height for w in row)


def find_blocks(rows: list[list[Cell]], word_height: float) -> list[list[int]]:
    """Index runs of rows that look like one table: multi-cell rows, with single-cell labels between them."""
    blocks: list[list[int]] = []
    current: list[int] = []

    def close() -> None:
        # a trailing single-cell row is a caption or label, not part of the table
        while current and len(rows[current[-1]]) < 2:
            current.pop()
        multi = sum(1 for i in current if len(rows[i]) >= 2)
        if len(current) >= MIN_TABLE_ROWS and multi >= MIN_MULTI_CELL_ROWS:
            blocks.append(list(current))
        current.clear()

    for index, cells in enumerate(rows):
        if current:
            previous = rows[current[-1]]
            vertical_gap = _row_top([w for c in cells for w in c.words]) - _row_bottom(
                [w for c in previous for w in c.words]
            )
            if vertical_gap > MAX_ROW_GAP * word_height:
                close()
        if len(cells) >= 2:
            current.append(index)
        elif current and any(len(r) >= 2 for r in rows[index + 1:index + 1 + MAX_LABEL_ROWS]):
            current.append(index)  # a label row inside the table, e.g. "Current assets" or a wrapped "(inches)"
        else:
            close()
    close()
    return blocks


def _columns(block: list[list[Cell]]) -> list[tuple[int, int]]:
    """Column x-ranges: the merged extents of the cells in the rows that have the most cells."""
    widest = max(len(r) for r in block)
    intervals = sorted((c.left, c.right) for r in block if len(r) == widest for c in r)
    columns: list[list[int]] = []
    for left, right in intervals:
        if columns and left <= columns[-1][1]:
            columns[-1][1] = max(columns[-1][1], right)
        else:
            columns.append([left, right])
    return [(a, b) for a, b in columns]


def _column_of(cell: Cell, columns: list[tuple[int, int]]) -> int:
    overlaps = [max(0, min(cell.right, b) - max(cell.left, a)) for a, b in columns]
    if max(overlaps) > 0:
        return overlaps.index(max(overlaps))
    centre = (cell.left + cell.right) / 2
    return min(range(len(columns)), key=lambda i: abs(centre - (columns[i][0] + columns[i][1]) / 2))


def block_to_markdown(block: list[list[Cell]]) -> str:
    columns = _columns(block)
    table = []
    for cells in block:
        row = [""] * len(columns)
        for cell in cells:
            index = _column_of(cell, columns)
            row[index] = f"{row[index]} {cell.text}".strip()
        table.append(row)
    return table_to_markdown(table)


def extract_tables(words: list[Word]) -> tuple[list[str], list[Word]]:
    """(tables as Markdown, the words that are not part of any table)."""
    if len(words) < 6:
        return [], words
    word_height = median(w.height for w in words)
    rows = [split_cells(r, CELL_GAP * word_height) for r in group_rows(words)]
    tables: list[str] = []
    used: set[int] = set()
    for indexes in find_blocks(rows, word_height):
        markdown = block_to_markdown([rows[i] for i in indexes])
        if not markdown:
            continue
        tables.append(markdown)
        used.update(w.order for i in indexes for c in rows[i] for w in c.words)
    return tables, [w for w in words if w.order not in used]


def words_to_text(words: list[Word]) -> str:
    """Lines in Tesseract's reading order, with a blank line between paragraphs."""
    lines: list[str] = []
    last_paragraph: tuple[int, int] | None = None
    last_line: tuple | None = None
    for word in sorted(words, key=lambda w: w.order):
        key = (word.paragraph, word.line)
        if key != last_line:
            if last_paragraph is not None and word.paragraph != last_paragraph:
                lines.append("")
            lines.append(word.text)
        else:
            lines[-1] += " " + word.text
        last_paragraph, last_line = word.paragraph, key
    return "\n".join(lines).strip()
