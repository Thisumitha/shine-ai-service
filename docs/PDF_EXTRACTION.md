# PDF extraction: what is done and how it works

This is the first half of "reading a PDF". It turns an uploaded PDF into clean text per page. Chunking, embeddings and search come later and are not built yet.

## What is done

| Part | File | What it does |
|---|---|---|
| Page and document objects | `app/domain/entities/document.py` | `PageContent` (page number, text, tables, whether OCR was used) and `ParsedDocument` |
| PDF reader | `app/infrastructure/ingestion/pdf_loader.py` | Reads each page's text with PyMuPDF, detects scanned pages, renders a page to a PNG image |
| OCR | `app/infrastructure/ingestion/ocr_tesseract.py` | Reads a page image with Tesseract (implements the `OcrEngine` port) |
| Tables | `app/infrastructure/ingestion/table_extractor.py` | Finds ruled tables with pdfplumber and writes them as Markdown |
| Cleaner | `app/infrastructure/ingestion/text_cleaner.py` | Fixes split words, extra spaces, and removes repeated headers and footers |
| Pipeline | `app/infrastructure/ingestion/pipeline.py` | Runs the steps above for a whole PDF |
| Service | `app/services/ingestion_service.py` | Runs the pipeline for a queued job and saves the result |
| Test endpoint | `app/api/routers/parse.py` | `POST /internal/parse-pdf`, returns pages and tables as JSON |
| Wiring | `app/container.py`, `app/main.py`, `app/config.py` | Builds the objects once and registers the route |

## How it works

```
Spring uploads PDF -> POST /internal/index -> job QUEUED (MySQL)
                                                 |
                      background worker takes the job (status EXTRACTING)
                                                 |
                    IngestionService.process -> PdfPipeline.parse
                                                 |
        for every page:
          1. read text                      (PdfLoader)
          2. text almost empty? -> scanned  -> render PNG -> Tesseract
          3. text pages only: find tables   (TableExtractor) -> Markdown
          4. clean the text                 (clean_page)
        after all pages:
          5. remove headers/footers that repeat on most pages
                                                 |
              save data/extracted/<itemId>.json, job status = EXTRACTED
```

Details:
- **Scanned page:** fewer than 25 characters of text. If no OCR engine is available the page stays empty instead of failing.
- **OCR cap:** at most `OCR_MAX_PAGES` (default 200) pages per PDF are OCR'd, to limit time.
- **Tables:** a table needs at least 2 rows and 2 columns. The first row becomes the header. Pages that were OCR'd skip table detection.
- **Headers/footers:** a top or bottom line that appears (digits ignored, so "Page 3" = "Page 4") on at least 60% of pages, and at least 3 pages, is removed. Pages with 4 or fewer lines are left alone.
- **Bad PDF:** the worker marks the job `FAILED` with the error text and keeps running. A table error on one page does not lose that page's text.
- **Output JSON:** `{"filename": ..., "pages": [{"page_number", "text", "tables_markdown", "used_ocr"}, ...]}`. The next stage (chunking) will read this.

## Settings (`.env`)

| Setting | Meaning |
|---|---|
| `TESSERACT_CMD` | Path to `tesseract.exe`. Leave empty if it is on PATH |
| `OCR_LANGUAGE` | Languages joined with `+`, e.g. `eng+sin+tam` (each needs its language file installed) |
| `OCR_MAX_PAGES` | Maximum pages OCR'd per PDF |
| `EXTRACTED_DIR` | Where the JSON results are saved (default `./data/extracted`) |

## Install Tesseract (Windows)

Install the UB Mannheim build, add the Sinhala and Tamil language data if needed, then set `TESSERACT_CMD`, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`. Without it, text PDFs work and scanned pages come back empty.

## Try it

1. `pip install -r requirements.txt`
2. `uvicorn app.main:app --reload --port 8000`
3. Open `http://localhost:8000/docs`, use `POST /internal/parse-pdf`, add the `X-Internal-Secret` header and choose a PDF.
4. Or run the tests: `pytest` (no network or API key needed; the Tesseract test is skipped when Tesseract is missing).

## Tests

- `tests/test_pdf_pipeline.py`: text, scanned (fake OCR), OCR page cap, tables, missing OCR engine, real Tesseract (skipped if not installed)
- `tests/test_text_cleaner.py`: cleaning and header/footer removal
- `tests/test_parse_endpoint.py`: the endpoint (secret, bad file, broken PDF) and the saved JSON
- `tests/pdf_samples.py`: builds small sample PDFs on the fly

## Not done yet

Chunking, embeddings, ChromaDB, search, chat, Gemini OCR fallback, Docker. Jobs end as `EXTRACTED`; they will end as `INDEXED` once indexing exists.
