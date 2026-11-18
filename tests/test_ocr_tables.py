"""The row/column rebuilding logic, tested with made-up word boxes (no Tesseract needed)."""

from app.infrastructure.ingestion.ocr_tables import Word, extract_tables, group_rows, split_cells, words_to_text

H = 20  # word height in pixels


def word(text, left, row, order, width=None):
    return Word(text, left, top=100 + row * 40, width=width or 12 * len(text), height=H, order=order)


def grid(rows):
    """rows of (text, left) cells -> words, with consecutive reading order numbers."""
    words, order = [], 0
    for r, cells in enumerate(rows):
        for text, left in cells:
            for k, part in enumerate(text.split()):
                words.append(word(part, left + k * 90, r, order, width=80))
                order += 1
    return words


TABLE = [
    [("Term", 100), ("Meaning", 500)],
    [("Stand alone", 100), ("Runs locally", 500)],
    [("Embedded", 100), ("Inside hardware", 500)],
    [("Batch", 100), ("Scheduled jobs", 500)],
]


def test_rows_are_grouped_by_vertical_position():
    rows = group_rows(grid(TABLE))
    assert len(rows) == 4
    assert [w.text for w in rows[1]] == ["Stand", "alone", "Runs", "locally"]


def test_wide_gaps_split_cells_and_small_gaps_do_not():
    cells = split_cells(group_rows(grid(TABLE))[1], gap=H)
    assert [c.text for c in cells] == ["Stand alone", "Runs locally"]


def test_aligned_rows_become_a_markdown_table_that_keeps_rows_together():
    tables, rest = extract_tables(grid(TABLE))
    assert rest == []
    assert tables == [
        "| Term | Meaning |\n| --- | --- |\n"
        "| Stand alone | Runs locally |\n| Embedded | Inside hardware |\n| Batch | Scheduled jobs |"
    ]


def test_section_label_rows_stay_inside_the_table():
    rows = [
        [("Item", 100), ("2010", 400), ("2009", 600)],
        [("Assets", 100)],
        [("Property", 100), ("345", 400), ("445", 600)],
        [("Cash", 100), ("524", 400), ("123", 600)],
    ]
    table = extract_tables(grid(rows))[0][0]
    assert "| Assets |  |  |" in table
    assert "| Property | 345 | 445 |" in table


def test_ordinary_prose_is_not_turned_into_a_table():
    lines = [[("This is one ordinary sentence of text", 100)] for _ in range(5)]
    tables, rest = extract_tables(grid(lines))
    assert tables == [] and len(rest) == len(grid(lines))


def test_text_outside_the_table_is_kept_in_reading_order():
    title = [("Marks", 100)]
    words = grid([title, *TABLE])
    tables, rest = extract_tables(words)
    assert len(tables) == 1
    assert words_to_text(rest) == "Marks"
