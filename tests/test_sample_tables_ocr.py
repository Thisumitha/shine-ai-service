"""Does the pipeline return the tables in sample/sample-tables.pdf correctly?

native: the PDF's own text layer (also proves the hand-written ground truth is right).
ocr   : the same pages turned into pictures, then read with the real Tesseract.

Run:  pytest tests/test_sample_tables_ocr.py -v
"""

import pytest

from app.config import get_settings
from evaluation.table_fidelity import DEFAULT_PDF, evaluate, find_tesseract, report

TESSERACT = find_tesseract(get_settings().tesseract_cmd)

pytestmark = pytest.mark.skipif(not DEFAULT_PDF.is_file(), reason="sample/sample-tables.pdf is missing")


def _mean(results, attr):
    return sum(getattr(s, attr) for s, _ in results) / len(results)


def test_native_extraction_matches_the_ground_truth():
    results = evaluate("native")
    print("\n" + report("native", results))
    assert _mean(results, "cell_recall") >= 0.98  # a wrong ground-truth cell would show up here
    assert _mean(results, "row_integrity") >= 0.90


@pytest.mark.skipif(TESSERACT is None, reason="Tesseract is not installed (set TESSERACT_CMD)")
def test_ocr_reads_every_table_cell():
    results = evaluate("ocr", tesseract_cmd=TESSERACT)
    print("\n" + report("ocr", results))
    for score, _ in results:
        assert score.cell_recall >= 0.90, f"{score.name}: missing {score.missing_cells}"


@pytest.mark.skipif(TESSERACT is None, reason="Tesseract is not installed (set TESSERACT_CMD)")
def test_ocr_keeps_each_table_row_together():
    results = evaluate("ocr", tesseract_cmd=TESSERACT)
    assert _mean(results, "row_integrity") >= 0.85, report("ocr", results)


@pytest.mark.skipif(TESSERACT is None, reason="Tesseract is not installed (set TESSERACT_CMD)")
def test_ocr_table_rows_survive_chunking_whole():
    results = evaluate("ocr", tesseract_cmd=TESSERACT)
    assert sum(chunked for _, chunked in results) / len(results) >= 0.85, report("ocr", results)
