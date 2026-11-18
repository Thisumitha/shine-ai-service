"""Checks how faithfully a PDF's tables survive extraction (native text layer, or OCR of page images).

Two scores per table, measured against hand-written ground truth (`sample_tables_truth.json`):
  cell recall   - share of expected cells found anywhere in the extracted page (text + Markdown tables)
  row integrity - share of expected rows whose cells all sit on ONE extracted row (a Markdown table row
                  or a single line of text). This is what breaks when OCR reads a table column by column.

    python -m evaluation.table_fidelity                 # native + OCR on sample/sample-tables.pdf
    python -m evaluation.table_fidelity --mode native
"""

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.domain.entities.document import PageContent
from app.infrastructure.ingestion.chunker import chunk_page
from app.infrastructure.ingestion.ocr_tesseract import find_tesseract  # also imported by the tests

ROOT = Path(__file__).resolve().parents[1]
TRUTH_PATH = ROOT / "evaluation" / "sample_tables_truth.json"
DEFAULT_PDF = ROOT / "sample" / "sample-tables.pdf"

def normalise(text: str) -> str:
    """Lower case, one space between words, and drop the bars Markdown puts between cells."""
    return " ".join(text.replace("|", " ").lower().split())


def load_truth(path: Path = TRUTH_PATH) -> dict[int, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(page): tables for page, tables in data["pages"].items()}


def rasterise(pdf_path: Path, pages: list[int], out_path: Path, dpi: int = 200) -> Path:
    """An image-only copy of the chosen pages (1-based), so the only way to read them is OCR."""
    source = pymupdf.open(pdf_path)
    target = pymupdf.open()
    for number in pages:
        page = source[number - 1]
        png = page.get_pixmap(dpi=dpi).tobytes("png")
        target.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, stream=png)
    target.save(out_path)
    return out_path


def extracted_rows(page: PageContent) -> list[str]:
    """Every row a reader could find on the page: Markdown table rows, then lines of plain text."""
    rows = [normalise(line) for table in page.tables_markdown for line in table.split("\n") if "---" not in line]
    rows += [normalise(line) for line in page.text.split("\n")]
    return [row for row in rows if row]


@dataclass
class TableScore:
    page: int
    name: str
    cell_recall: float
    row_integrity: float
    missing_cells: list[str]
    split_rows: list[list[str]]


def score_table(page_no: int, table: dict, page: PageContent) -> TableScore:
    page_text = normalise(page.text + "\n" + "\n".join(page.tables_markdown))
    rows = extracted_rows(page)
    cells = [c for row in table["rows"] for c in row]
    missing = [c for c in cells if normalise(c) not in page_text]
    split = [row for row in table["rows"] if not any(all(normalise(c) in found for c in row) for found in rows)]
    return TableScore(
        page=page_no,
        name=table["name"],
        cell_recall=1 - len(missing) / len(cells),
        row_integrity=1 - len(split) / len(table["rows"]),
        missing_cells=missing,
        split_rows=split,
    )


def chunk_row_integrity(table: dict, page: PageContent) -> float:
    """Share of expected rows that end up whole inside a single chunk once the page is chunked."""
    chunks = [normalise(text) for text, _ in chunk_page(page)]
    whole = [r for r in table["rows"] if any(all(normalise(c) in chunk for c in r) for chunk in chunks)]
    return len(whole) / len(table["rows"])


def read_pages(pdf_path: Path, mode: str, pages: list[int], tesseract_cmd: str = "") -> dict[int, PageContent]:
    """Parse the chosen pages with the real pipeline. mode is 'native' or 'ocr' (rasterised first)."""
    from app.infrastructure.ingestion.ocr_tesseract import TesseractOcr
    from app.infrastructure.ingestion.pipeline import PdfPipeline

    with tempfile.TemporaryDirectory() as tmp:
        if mode == "ocr":
            pdf_path = rasterise(pdf_path, pages, Path(tmp) / "scanned.pdf")
            document = PdfPipeline(TesseractOcr(tesseract_cmd=tesseract_cmd)).parse(str(pdf_path))
            return {number: page for number, page in zip(pages, document.pages)}
        document = PdfPipeline(None).parse(str(pdf_path))
        return {number: document.pages[number - 1] for number in pages}


def evaluate(mode: str, pdf_path: Path = DEFAULT_PDF, tesseract_cmd: str = "") -> list[tuple[TableScore, float]]:
    truth = load_truth()
    parsed = read_pages(pdf_path, mode, sorted(truth), tesseract_cmd)
    results = []
    for page_no, tables in truth.items():
        for table in tables:
            results.append((score_table(page_no, table, parsed[page_no]), chunk_row_integrity(table, parsed[page_no])))
    return results


def _fit(cell: str, limit: int) -> str:
    cell = " ".join(str(cell).split())
    return cell if len(cell) <= limit else cell[: limit - 3] + "..."


def grid(rows: list[list[str]], max_cell: int = 44) -> str:
    """Rows as an aligned text grid with ruled lines, readable in any terminal (long cells end in ...)."""
    rows = [[_fit(c, max_cell) for c in row] for row in rows if row]
    if not rows:
        return "(empty)"
    columns = max(len(r) for r in rows)
    rows = [r + [""] * (columns - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(columns)]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([rule, body[0], rule, *body[1:], rule])


def markdown_rows(markdown: str) -> list[list[str]]:
    """Cells of a Markdown table, without the --- separator row."""
    rows = []
    for line in markdown.split("\n"):
        cells = [c.strip().replace("\\|", "|") for c in line.strip().strip("|").split(" | ")]
        if line.strip() and not set(line.replace("|", "").replace(" ", "")) <= {"-"}:
            rows.append(cells)
    return rows


def show_tables(mode: str, pdf_path: Path = DEFAULT_PDF, tesseract_cmd: str = "") -> str:
    """The tables exactly as the pipeline extracted them, drawn as text grids, next to the answer key."""
    truth = load_truth()
    parsed = read_pages(pdf_path, mode, sorted(truth), tesseract_cmd)
    out = [f"EXTRACTED TABLES ({mode})"]
    for page_no in sorted(truth):
        page = parsed[page_no]
        out.append(f"\n=== page {page_no}: {len(page.tables_markdown)} table(s) found, {len(truth[page_no])} expected ===")
        for i, markdown in enumerate(page.tables_markdown, 1):
            out.append(f"\n  extracted table {i}:\n" + "\n".join("  " + line for line in grid(markdown_rows(markdown)).split("\n")))
        for table in truth[page_no]:
            out.append(f"\n  expected: {table['name']}\n" + "\n".join("  " + line for line in grid(table["rows"]).split("\n")))
    return "\n".join(out)


def report(mode: str, results: list[tuple[TableScore, float]]) -> str:
    """Aligned plain-text scores for the terminal (report_markdown gives the Markdown version)."""
    width = max(len(s.name) for s, _ in results)
    head = f" {'Pg':>2}  {'Table':<{width}}  {'Cells':>6}  {'Rows':>6}  {'Chunked':>8}"
    lines = [f"TABLE FIDELITY: {mode}", "=" * len(head), head, "-" * len(head)]
    for s, chunked in results:
        lines.append(f" {s.page:>2}  {s.name:<{width}}  {s.cell_recall:>6.0%}  {s.row_integrity:>6.0%}  {chunked:>8.0%}")
    n = len(results)
    lines.append("-" * len(head))
    lines.append(
        f" {'':>2}  {'MEAN':<{width}}  {sum(s.cell_recall for s, _ in results) / n:>6.0%}  "
        f"{sum(s.row_integrity for s, _ in results) / n:>6.0%}  {sum(c for _, c in results) / n:>8.0%}"
    )
    lines.append("=" * len(head))
    lines.append(" Cells = expected cells found | Rows = rows kept together | Chunked = rows whole after chunking")
    problems = [
        (s, f"missing cells {s.missing_cells}" if s.missing_cells else "", f"split rows {s.split_rows[:3]}" if s.split_rows else "")
        for s, _ in results
        if s.missing_cells or s.split_rows
    ]
    if problems:
        lines.append("\n Problems found:")
        for s, missing, split in problems:
            lines += [f"  - page {s.page}, {s.name}"] + [f"      {m}" for m in (missing, split) if m]
    return "\n".join(lines)


def report_markdown(mode: str, results: list[tuple[TableScore, float]]) -> str:
    lines = [
        f"### Table fidelity: {mode}",
        "",
        "| Page | Table | Cell recall | Row integrity | Rows intact after chunking |",
        "| :---: | :--- | :---: | :---: | :---: |",
    ]
    for s, chunked in results:
        lines.append(f"| {s.page} | {s.name} | {s.cell_recall:.0%} | {s.row_integrity:.0%} | {chunked:.0%} |")
    n = len(results)
    lines.append(
        f"| | **mean** | **{sum(s.cell_recall for s, _ in results) / n:.0%}** | "
        f"**{sum(s.row_integrity for s, _ in results) / n:.0%}** | **{sum(c for _, c in results) / n:.0%}** |"
    )
    for s, _ in results:
        if s.missing_cells:
            lines.append(f"\n- p{s.page} {s.name}: missing cells {s.missing_cells}")
        if s.split_rows:
            lines.append(f"- p{s.page} {s.name}: rows split across lines {s.split_rows[:3]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score table extraction against ground truth")
    parser.add_argument("--mode", choices=["native", "ocr", "both"], default="both")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--show-tables", action="store_true", help="draw each extracted table next to the expected one")
    parser.add_argument("--markdown", action="store_true", help="print the scores as a Markdown table (for documents)")
    args = parser.parse_args()

    from app.config import get_settings

    tesseract = find_tesseract(get_settings().tesseract_cmd)
    modes = ["native", "ocr"] if args.mode == "both" else [args.mode]
    if "ocr" in modes and tesseract is None:
        sys.exit("Tesseract not found. Install it and set TESSERACT_CMD in .env (see docs/PDF_EXTRACTION.md).")
    for mode in modes:
        if args.show_tables:
            print(show_tables(mode, Path(args.pdf), tesseract or "") + "\n")
        render = report_markdown if args.markdown else report
        print(render(mode, evaluate(mode, Path(args.pdf), tesseract or "")) + "\n")


if __name__ == "__main__":
    main()
