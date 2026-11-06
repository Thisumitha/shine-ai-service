# Chunking: what is done and how it works

Chunking cuts the text of a PDF into small passages. Search and the AI work on passages, not on a whole PDF, and each passage keeps its page number so an answer can cite "page 12". Embeddings and search come next and are not built yet.

## What is done

| Part | File | What it does |
|---|---|---|
| Chunk object | `app/domain/entities/chunk.py` | `Chunk` (item, course, topic, title, page, index, text, is_table) with a stable `chunk_id` = `<itemId>-<index>` |
| Chunker | `app/infrastructure/ingestion/chunker.py` | `chunk_document` cuts a parsed PDF into chunks |
| Service | `app/services/ingestion_service.py` | After parsing, builds the chunks and saves `data/chunks/<itemId>.json` |
| Setting | `app/config.py` | `CHUNKS_DIR` (default `./data/chunks`) |

## How it works

```
data/extracted/<itemId>.json  (pages: text + tables)
        |
   for every page (a chunk never crosses a page):
     1. split the text into paragraphs (blank lines)
     2. a paragraph longer than 1200 characters -> split into sentences
        (. ? ! । and line ends); a sentence longer than that -> cut at a space
     3. pack the pieces together up to about 900 characters
     4. start the next chunk with the last whole sentences (up to 150 characters)
        of the previous one, so an idea cut at the edge is still found
     5. a leftover under 150 characters is merged into the chunk before it
     6. every table is its own chunk (a table over 1200 characters is split
        by rows, with the header row repeated in each part)
        |
data/chunks/<itemId>.json
```

Sizes are in characters, so Sinhala, Tamil and English are treated the same and no tokenizer is needed. They are constants at the top of `chunker.py` and can be changed there.

## Output

`data/chunks/<itemId>.json` is a list of:

```json
{"item_id": 6, "course_id": 3, "topic_id": 1, "title": "Algebra", "page_number": 2,
 "chunk_index": 1, "text": "...", "is_table": false, "chunk_id": "6-1"}
```

## Try it

Upload a PDF through `POST /internal/index` (or run the service on a queued job) and open `data/chunks/<itemId>.json`. Check that each chunk has a page number, sizes are within limits, and a table is one chunk. `pytest tests/test_chunker.py` covers the rules without any PDF.

## Not done yet

Embeddings, ChromaDB, search, chat. Jobs still end as `EXTRACTED`; they will end as `INDEXED` once chunks are embedded and stored.
