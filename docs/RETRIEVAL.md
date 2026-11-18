# Retrieval Architecture & Hybrid Indexing

This document describes the retrieval engine implemented in `app/infrastructure/retrieval/` and `app/infrastructure/store/` (Member 2).

---

## 1. Overview

The retrieval engine provides relevant course context chunks for user queries. Because academic lecture slides and course notes contain both semantic concepts (requiring neural vector search) and specific keywords/acronyms/formulas (requiring exact lexical search), the service implements **Hybrid Retrieval** combining:

1. **Dense Retrieval**: Semantic vector search with Gemini embeddings + ChromaDB.
2. **Sparse Retrieval**: Lexical term matching with TF-IDF.
3. **Reciprocal Rank Fusion (RRF)**: Re-ranks and blends both result sets into a single ordered candidate list.

```
                  User Query
                      |
        +-------------+-------------+
        |                           |
        v                           v
  DenseRetriever              SparseIndex
(GeminiEmbedder + Chroma)     (TF-IDF Lexical)
        |                           |
   Top-K Dense                 Top-K Sparse
        |                           |
        +-------------+-------------+
                      |
                      v
            Reciprocal Rank Fusion (RRF)
                      |
                      v
             Top-K Fused Chunks
```

---

## 2. Components

### A. Embedder (`GeminiEmbedder`)
- **Location**: `app/infrastructure/retrieval/embedder_gemini.py`
- **Port Implemented**: `app.domain.ports.embedder.Embedder`
- **Model**: `gemini-embedding-001` by default (via `google-genai` SDK); set `GEMINI_EMBEDDING_MODEL` in `.env` to change it, then re-ingest.
- Generates 768-dimensional dense vectors for both document chunks and incoming user queries.
- Batched embedding generation during PDF indexing.

### B. Dense Vector Store (`ChromaVectorStore` & `DenseRetriever`)
- **Location**:
  - `app/infrastructure/store/chroma_vector_store.py`
  - `app/infrastructure/retrieval/dense_index.py`
- **Storage**: Persistent SQLite-backed ChromaDB stored in `./data/chroma`.
- **Metadata Stored**:
  - `chunk_id`
  - `item_id` / `course_id`
  - `source_doc` (PDF filename)
  - `page_number`
  - `heading_path`
- **Filtering**: Enforces strict course isolation by filtering against `course_id` metadata.

### C. Sparse Keyword Index (`SparseIndex`)
- **Location**: `app/infrastructure/retrieval/sparse_index.py`
- **Port Implemented**: `app.domain.ports.sparse_indexer.SparseIndexer`
- Uses `scikit-learn`'s `TfidfVectorizer` to index tokens.
- Captures exact terminology, acronyms (e.g., "ACID", "REST", "FIFO"), function names, and codes that embeddings might generalize away.

### D. Hybrid Retriever (`HybridRetriever`)
- **Location**: `app/infrastructure/retrieval/hybrid_retriever.py`
- **Port Implemented**: `app.domain.ports.retriever.Retriever`
- Executes parallel/sequential queries to both Dense and Sparse indices.
- Fuses rankings using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$
  *(Default constant $k = 60$)*
- Returns deduplicated, ranked chunks adhering to the requesting student's authorized `course_ids`.

---

## 3. Data Flow

1. **Indexing Flow**:
   - `IngestionService` cuts extracted text into `Chunk` objects.
   - Chunks are sent to `GeminiEmbedder` to generate vectors.
   - `ChromaVectorStore` persists vectors and metadata.
   - `SparseIndex` indexes chunk text for lexical search.

2. **Query Flow**:
   - `ActStage` calls `retriever.search(query, course_ids, top_k=5)`.
   - `DenseRetriever` queries Chroma with semantic embedding.
   - `SparseIndex` queries TF-IDF index with raw query tokens.
   - `HybridRetriever` computes RRF scores, sorts chunks descending, and returns top $K$ `RetrievedChunk` instances.

