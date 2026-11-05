from app.infrastructure.ingestion.text_cleaner import clean_page, remove_repeated_edges


def test_joins_hyphenated_line_breaks():
    assert clean_page("informa-\ntion is key") == "information is key"


def test_collapses_spaces_and_blank_lines():
    assert clean_page("a   b\n\n\n\nc  \n") == "a b\n\nc"


def test_drops_headers_and_footers_that_repeat_on_most_pages():
    topics = ["alpha", "bravo", "charlie", "delta", "echo"]
    pages = [f"Shine Institute\nIntro of {t}\nMiddle of {t}\nEnd of {t}\nPage {i}" for i, t in enumerate(topics, 1)]
    cleaned = remove_repeated_edges(pages)
    assert cleaned == [f"Intro of {t}\nMiddle of {t}\nEnd of {t}" for t in topics]


def test_keeps_text_when_there_are_too_few_pages_to_judge():
    pages = ["Header\nOne", "Header\nTwo"]
    assert remove_repeated_edges(pages) == pages
