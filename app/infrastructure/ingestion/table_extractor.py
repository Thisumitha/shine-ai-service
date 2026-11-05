import pdfplumber


def _cell(value) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def table_to_markdown(rows: list[list]) -> str:
    """Turn rows (first row = header) into a Markdown table. Empty string if it is not a real table."""
    rows = [[_cell(c) for c in row] for row in rows if row]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class TableExtractor:
    """Finds ruled tables on each page with pdfplumber and returns them as Markdown."""

    def __init__(self, path: str) -> None:
        self.pdf = pdfplumber.open(path)

    def __enter__(self) -> "TableExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.pdf.close()

    def tables(self, index: int) -> list[str]:
        found = self.pdf.pages[index].extract_tables()
        return [md for md in (table_to_markdown(t) for t in found) if md]
