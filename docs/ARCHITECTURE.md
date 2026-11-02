# Architecture

The service uses four layers. Dependencies point inward only: `api -> services -> domain`, and `infrastructure -> domain`. Nothing in `domain` imports FastAPI, httpx, Chroma or Gemini.

```
app/
  main.py, config.py        app factory and settings
  container.py              composition root: the only place that wires concrete classes to ports
  api/                      PRESENTATION  HTTP only: routers, request/response schemas, auth dependencies
    routers/                health, chat, study, reports, internal_index
  services/                 APPLICATION   use cases: chat, ingestion, quiz, flashcards, summary, progress, reports
  domain/                   DOMAIN        plain Python, no framework
    entities/               CurrentUser, Chunk, Document, Message ...
    ports/                  interfaces: CourseAccess, Embedder, LLMClient, OcrEngine, Retriever, ChatStore
  infrastructure/           ADAPTERS      implement the ports
    auth/                   JWT verifier, access guard
    clients/                Spring backend client, game service client
    ingestion/              pdf loader, text cleaner, chunker, OCR engines
    retrieval/              dense + sparse index, hybrid retriever, reranker
    store/                  Chroma vector store, SQLite document registry
    llm/                    Gemini client
    memory/                 SQLite chat store
evaluation/                 ground-truth set and accuracy runs
scripts/                    backfill and maintenance scripts
tests/                      mirrors app/; ports are replaced by fakes (no network, no API key)
```

## Rules

- A router calls one service method and maps the result to a response. No business logic in routers.
- A service depends on ports (abstract classes), never on a concrete adapter.
- `container.py` builds the adapters and injects them into the services. Tests build the same services with fakes.
- The caller identity (`CurrentUser`) comes from the verified token. Course access is asked from the backend using that email.

## Where the build steps land

| Step | Layer(s) |
|---|---|
| JWT auth | infrastructure/auth + api deps |
| Spring client | infrastructure/clients (implements `CourseAccess`) |
| Internal index API | api/routers + services |
| PDF ingestion, retrieval, LLM, chat, study, reports | later phases |
