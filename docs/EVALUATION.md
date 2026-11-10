# Evaluation Framework & Benchmark

This document describes the evaluation dataset, metrics, and ablation runner located in `evaluation/`.

---

## 1. Overview

To scientifically measure and report the system's accuracy and reliability, we benchmark both:
1. **Retrieval Performance**: How effectively the Hybrid Retriever finds the right slides.
2. **Generation Quality**: How grounded and accurate the answers are, and whether hallucinations are caught.

---

## 2. Benchmark Dataset (`evaluation/dataset.jsonl`)

The dataset contains **30–50 curated Question-Answer pairs** extracted from actual course lecture slides and PDFs.

### Structure of each JSON line:
```json
{
  "id": "qa-001",
  "course_id": 101,
  "question": "What is the worst-case time complexity of merge sort and why?",
  "ground_truth_answer": "Merge sort has an O(n log n) worst-case time complexity because it always divides the array in half (log n levels) and performs linear time merging (n operations per level).",
  "expected_source_doc": "Algorithms_101.pdf",
  "expected_page_numbers": [18],
  "is_answerable": true,
  "language": "en"
}
```

### Unanswerable Test Cases (~5 Questions):
Includes deliberately out-of-syllabus or non-existent questions (e.g., *"What is quantum entanglement according to Chapter 1?"*) with `"is_answerable": false` to verify that the `SelfCheckService` refuses to answer and scores high on refusal recall.

---

## 3. Metrics Computed

### A. Retrieval Metrics

1. **Hit@K (K=1, 3, 5)**:
   - Evaluates whether the expected page number appears within the top $K$ retrieved chunks.
   $$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{target page} \in \text{Top-}K)$$

2. **Mean Reciprocal Rank (MRR)**:
   - Measures the rank position where the first relevant slide chunk was retrieved.
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

### B. Generation & Safety Metrics

1. **Groundedness / Faithfulness (NLI Entailment)**:
   - Sentence-level verification checking whether every claim made in the generated answer is directly supported by the retrieved course chunks.
   - Evaluated by the `SelfCheckService` on a scale of $0.00$ to $1.00$.

2. **Answer Relevance**:
   - Assesses whether the response directly addresses the student's question without extraneous off-topic information.

3. **Refusal Accuracy**:
   - Precision and recall for correctly identifying unanswerable questions and responding with the standard fallback message.

---

## 4. Running Ablation Studies

The ablation CLI compares different configurations of the pipeline:

```powershell
.venv\Scripts\python.exe -m evaluation.run_eval --config dense_only
.venv\Scripts\python.exe -m evaluation.run_eval --config sparse_only
.venv\Scripts\python.exe -m evaluation.run_eval --config hybrid
.venv\Scripts\python.exe -m evaluation.run_eval --config hybrid_with_self_check
```

### Ablation Matrix:
| Configuration | Hit@3 | Hit@5 | MRR | Groundedness | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense Only (Gemini Embeddings) | -- | -- | -- | -- | -- |
| Sparse Only (TF-IDF) | -- | -- | -- | -- | -- |
| **Hybrid (Dense + Sparse RRF)** | **Target $\ge 80\%$** | **Target $\ge 90\%$** | **Target $\ge 0.75$** | -- | -- |
| **Hybrid + Self-Check Verifier** | **Target $\ge 80\%$** | **Target $\ge 90\%$** | **Target $\ge 0.75$** | **Target $\ge 90\%$** | **$< 5\%$** |

