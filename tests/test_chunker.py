from app.domain.entities.document import PageContent, ParsedDocument
from app.infrastructure.ingestion.chunker import chunk_document, chunk_page


def sentence(n: int) -> str:
    return f"Sentence number {n} explains one small idea about the lesson."


def paragraph(start: int, count: int) -> str:
    return " ".join(sentence(n) for n in range(start, start + count))


def doc(*pages: PageContent) -> ParsedDocument:
    return ParsedDocument(filename="x.pdf", pages=list(pages))


def chunks_of(document: ParsedDocument):
    return chunk_document(document, item_id=7, course_id=3, topic_id=2, title="Algebra")


def test_short_page_is_one_chunk():
    result = chunks_of(doc(PageContent(1, "A short page.")))
    assert [c.text for c in result] == ["A short page."]
    assert result[0].page_number == 1 and result[0].course_id == 3 and result[0].title == "Algebra"
    assert result[0].chunk_id == "7-0"


def test_empty_page_gives_nothing():
    assert chunks_of(doc(PageContent(1, ""), PageContent(2, "  "))) == []


def test_long_text_is_split_within_the_size_limits():
    page = PageContent(1, "\n\n".join(paragraph(i * 6, 6) for i in range(10)))
    result = chunks_of(doc(page))
    assert len(result) > 1
    assert all(len(c.text) <= 1200 for c in result)
    assert all(len(c.text) >= 150 for c in result)


def test_chunks_never_cross_a_page():
    result = chunks_of(doc(PageContent(1, paragraph(0, 30)), PageContent(2, paragraph(100, 30))))
    assert {c.page_number for c in result} == {1, 2}
    page_one = " ".join(c.text for c in result if c.page_number == 1)
    assert "Sentence number 100" not in page_one
    assert [c.chunk_index for c in result] == list(range(len(result)))


def test_next_chunk_repeats_the_end_of_the_previous_one():
    result = chunks_of(doc(PageContent(1, "\n\n".join(paragraph(i * 6, 6) for i in range(10)))))
    for before, after in zip(result, result[1:]):
        last_sentence = before.text.split(". ")[-1].strip()
        assert last_sentence in after.text


def test_words_are_never_cut():
    words = " ".join(f"word{n}" for n in range(1000))  # one huge "sentence" with no full stop
    result = chunks_of(doc(PageContent(1, words)))
    assert len(result) > 1
    assert all(len(c.text) <= 1200 for c in result)
    joined = " ".join(c.text for c in result)
    assert all(f"word{n}" in joined.split() for n in range(0, 1000, 37))


def test_sinhala_sentences_split_on_the_danda():
    text = " ".join("මෙය පාඩමේ කුඩා අදහසක් පැහැදිලි කරයි।" for _ in range(80))
    result = chunks_of(doc(PageContent(1, text)))
    assert len(result) > 1
    assert all(c.text.rstrip().endswith("।") for c in result)


def test_tiny_leftover_is_merged_into_the_previous_chunk():
    # just over one chunk worth of text, then a very short last paragraph
    page = PageContent(1, paragraph(0, 14) + "\n\n" + paragraph(14, 1))
    result = chunks_of(doc(page))
    assert all(len(c.text) >= 150 for c in result)


def test_table_is_its_own_chunk():
    table = "| Name | Marks |\n| --- | --- |\n| Nimal | 80 |"
    result = chunks_of(doc(PageContent(2, "Marks of the class.", [table])))
    assert [(c.text, c.is_table) for c in result] == [("Marks of the class.", False), (table, True)]


def test_long_table_is_split_by_rows_with_the_header_repeated():
    header = "| Name | Marks |\n| --- | --- |"
    rows = "\n".join(f"| Student {n} with a long enough name | {n} |" for n in range(80))
    parts = chunk_page(PageContent(1, "", [header + "\n" + rows]))
    assert len(parts) > 1
    assert all(is_table and text.startswith(header) and len(text) <= 1200 for text, is_table in parts)
    assert sum(text.count("Student") for text, _ in parts) == 80
