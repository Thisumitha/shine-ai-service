# Evaluation Framework & Benchmark

Code: `evaluation/` (`metrics.py`, `live_judges.py`, `run_eval.py`). Run it with `python -m app.cli eval`.

## 1. Datasets

| File | Rows | Course | PDF it needs |
| :--- | :---: | :---: | :--- |
| `evaluation/sample_dataset.jsonl` (default) | 16 (14 answerable, 2 not) | 1 | `sample/Introduction_to_Software_Engineering.pdf` (`python -m app.cli load-sample`) |
| `evaluation/dataset.jsonl` | 40 (35 answerable, 5 not) | 101 | three marketing PDFs that are **not** in the repo; ingest them with `--course 101` first, otherwise every answer is a refusal |

Each row is a Question / Answer / Context triple:

```json
{
  "id": "se-qa-001", "course_id": 1,
  "question": "What is software engineering?",
  "ground_truth_answer": "Software engineering is the disciplined and organized approach to ...",
  "ground_truth_context": "the passage on the page that states it",
  "expected_source_doc": "Introduction_to_Software_Engineering.pdf",
  "expected_page_numbers": [5],
  "is_answerable": true
}
```

`ground_truth_context` in `sample_dataset.jsonl` was picked automatically (the chunk on the expected page that best matches the reference answer). Review it by hand before quoting it as a gold label.
Unanswerable rows (`is_answerable: false`) have no context; the system must refuse them.

## 2. Metrics

Unanswerable questions are scored only for refusal accuracy. The RAG metrics are computed on answerable questions.

| Metric | How it is computed |
| :--- | :--- |
| **Context Relevance** | mean cosine similarity between the question embedding and each retrieved chunk embedding, `dot(a, b) / (|a||b|)` written out in `live_judges.cosine` |
| **Answer Relevance** | cosine similarity between the question embedding and the answer embedding. A false refusal of an answerable question is scored too, so it lowers the average |
| **Groundedness (Faithfulness)** | an LLM judge (its own prompt, separate from the agent's self-check) lists the answer's claims and marks each supported or not by the retrieved chunks; score = supported / total, computed in code. Refusals make no claims and are not scored |
| **Context Recall** | share of the distinct words of `ground_truth_context` found in the retrieved chunks (no LLM) |
| **Hit@1/3/5, MRR** | whether the expected page is among the first k retrieved chunks (retrieval uses `top_k = 5`) |
| **Refusal accuracy / false refusal rate** | unanswerable questions refused; answerable questions refused |
| **Answer F1** | word overlap with the ground-truth answer |

## 3. Ablation

```powershell
python -m app.cli eval --config dense_only
python -m app.cli eval --config sparse_only
python -m app.cli eval --config hybrid
python -m app.cli eval --config hybrid_with_self_check
python -m app.cli eval --config all --output report.md     # all four, one table
```

| Configuration | Retriever | Generation |
| :--- | :--- | :--- |
| `dense_only` | Gemini embeddings + Chroma | answer from top 4 chunks, no self-check |
| `sparse_only` | TF-IDF | answer from top 4 chunks, no self-check |
| `hybrid` | dense + TF-IDF fused with RRF | answer from top 4 chunks, no self-check |
| `hybrid_with_self_check` | hybrid | the full agent: plan, retrieve, answer, self-check, bounded retry |

`--mock` produces synthetic numbers for a smoke test only. Reports say so, and the numbers must never be quoted as results.
Free Gemini keys are rate limited and the API returns 503 under load; use `--limit N` and re-run if a run is interrupted. A transient API failure can lower a score, so re-run before drawing conclusions from a single bad row.
