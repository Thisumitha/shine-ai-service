import re
from collections import Counter


def clean_page(text: str) -> str:
    """Fix words split across lines, collapse spaces and extra blank lines."""
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "informa-\ntion" -> "information"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _edge_lines(text: str, count: int = 2) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) <= 2 * count:  # too short to tell a header from the body
        return []
    return lines[:count] + lines[-count:]


def _normalise(line: str) -> str:
    return re.sub(r"\d+", "#", line)  # "Page 3" and "Page 4" count as the same footer


def remove_repeated_edges(pages: list[str], min_pages: int = 3, share: float = 0.6) -> list[str]:
    """Drop lines that sit at the top or bottom of most pages (running headers and footers)."""
    if len(pages) < min_pages:
        return pages
    seen = Counter(_normalise(line) for text in pages for line in set(_edge_lines(text)))
    repeated = {line for line, n in seen.items() if n >= max(min_pages, share * len(pages))}
    if not repeated:
        return pages
    result = []
    for text in pages:
        edges = set(_edge_lines(text))
        drop = {line for line in edges if _normalise(line) in repeated}
        kept = [line for line in text.split("\n") if line.strip() not in drop]
        result.append(clean_page("\n".join(kept)))
    return result
