"""Small PDFs generated on the fly so the tests need no files or network."""

import pymupdf


def text_pdf(path, pages: list[str]) -> str:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), body, fontsize=12)
    doc.save(path)
    return str(path)


def scanned_pdf(path, text: str = "SCANNED PAGE") -> str:
    """A PDF whose only page is a picture, with no text layer."""
    source = pymupdf.open()
    page = source.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 300), text, fontsize=40)
    png = page.get_pixmap(dpi=100).tobytes("png")
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png)
    doc.save(path)
    return str(path)


def table_pdf(path) -> str:
    """One page with a ruled 3x3 table."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 30, 550, 80), "Marks table", fontsize=14)
    x0, y0, cw, rh = 50, 100, 120, 30
    for r in range(4):
        page.draw_line((x0, y0 + r * rh), (x0 + 3 * cw, y0 + r * rh))
    for c in range(4):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + 3 * rh))
    cells = [["Name", "Maths", "Science"], ["Nimal", "80", "75"], ["Kamal", "65", "90"]]
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            page.insert_text((x0 + c * cw + 8, y0 + r * rh + 20), value, fontsize=11)
    doc.save(path)
    return str(path)
