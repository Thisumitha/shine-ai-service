# Shine AI Service: Explainable Agentic RAG

Ask questions about PDF files and get answers that **show their working**: the chunks that were retrieved, the similarity maths behind them (TF-IDF dot product, vector magnitudes, cosine, Reciprocal Rank Fusion) and automatic scores for Context Relevance, Groundedness and Answer Relevance.

It runs **on its own**: no Spring backend, no MySQL, no login. All you need is below.

## 0. What you need

| Need | Check |
| :--- | :--- |
| Python 3.10 or newer | `python --version` (on macOS/Linux try `python3 --version`) |
| A Google Gemini API key (free) | https://aistudio.google.com/apikey |
| Internet | the key is used for embeddings and answers |
| Optional: Tesseract OCR | only for scanned PDFs; normal PDFs work without it |

> Do not look for a ready-made `venv` folder in this project. A venv is created on your own computer in step 1.

## 1. Set up (about 5 minutes, once)

Open a terminal **inside this folder** (the one that contains `README.md`).

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
```

What you should see:
- After `activate`, your prompt starts with `(venv)`.
- `pip install` prints a lot of lines and takes a few minutes. It ends without the word `ERROR`.
- In the `.env` file, change the line `GEMINI_API_KEY=` to `GEMINI_API_KEY=your_key_here`, save and close. Leave every other line as it is.

If PowerShell says *running scripts is disabled* when you activate, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Every new terminal window needs `venv\Scripts\activate` (or `source venv/bin/activate`) again.

## 2. Quick test with the included sample PDF

The sample is `sample/Introduction_to_Software_Engineering.pdf`.

```bash
python -m app.cli load-sample
python -m app.cli ask "What is the software crisis?" --show-math
python -m app.cli eval --limit 5
```

What you should see:
1. `load-sample` prints `[INGEST] ...` and a line like `22 chunks (pages 1-18) embedded into Chroma and indexed with TF-IDF`.
2. `ask` prints these blocks in order:
   - `[OBSERVE]` your question
   - `[REASON]` the intent the system decided on, and `[PLAN]` when a multi-part question was split into several searches
   - `[RETRIEVE]` the chunks each search returned **before** any answer, with RRF, dense and TF-IDF scores
   - `[LOOP]` / `[CHECK]` retries (a rewritten query) and the self-check verdict
   - `[ANSWER]` with the source pages
   - `[PERSIST]` the conversation was saved
3. `eval` shows each question as it runs (pages retrieved versus expected, a short answer, and its scores), then a summary table with Context Relevance, Answer Relevance and Groundedness. Library log lines are hidden; set `SHINE_DEBUG=1` to see them. Remove `--limit 5` to run all 16 questions. `python -m app.cli eval --config all` runs the four ablation configurations (dense only, TF-IDF only, hybrid, hybrid + self-check), each with its own retriever.

## 3. Test with your own PDFs

1. Put your PDF files in one folder, for example `C:\my_pdfs`.
2. Start clean (this deletes the earlier index in `data/`) and ingest them:

```bash
python -m app.cli reset
python -m app.cli ingest "C:\my_pdfs"
```
You can also ingest one file: `python -m app.cli ingest "C:\my_pdfs\notes.pdf"`.
**Put the path in double quotes**, especially if it has spaces.

3. Ask questions. **Put the question in double quotes.**

```bash
python -m app.cli ask "Your question about the PDF here" --show-math
```

### How to read the output (the numbers below are only an example)

```
#1 notes.pdf p.5  RRF=0.0325  dense_rank=1 sparse_rank=2
    dense distance = 0.2310
    TF-IDF dot=3.4120  |q|=2.1000  |d|=6.8000  cosine = dot/(|q||d|) = 0.2389
```
- `#1`: best chunk. `p.5`: its page in the PDF.
- `dense_rank` / `sparse_rank`: its position in the embedding search and in the TF-IDF keyword search.
- `RRF`: the two ranks fused as `1/(60+rank)` for each list, summed. Higher is better.
- `dot`, `|q|`, `|d|`, `cosine`: the TF-IDF maths. `cosine = dot / (|q| x |d|)`.
- The **worked example** below the chunks lists each shared word with its idf, query weight, chunk weight and their product. The products add up to the dot product.

## 4. Test with your own questions and answers (evaluation)

1. Copy `evaluation/sample_dataset.jsonl` to a new file, e.g. `my_questions.jsonl`.
2. Edit it. One line is one question:

```json
{"id": "q1", "course_id": 1, "question": "What is X?", "ground_truth_answer": "X is ...", "ground_truth_context": "the passage that states it", "expected_source_doc": "notes.pdf", "expected_page_numbers": [3], "is_answerable": true}
```
- `course_id` must be `1`, the same course the `ingest` command used by default. If you ingest with `--course 7`, use `7` here and in `ask --course 7`.
- `expected_page_numbers`: the page(s) where the answer is. Used for Hit@k and MRR.
- For a question the PDF cannot answer, use `"is_answerable": false`, `"expected_page_numbers": []` and an empty `ground_truth_answer`. The system should refuse it.

3. Run it and save the report:
```bash
python -m app.cli eval --dataset my_questions.jsonl --output report.md
```

| Score | Meaning |
| :--- | :--- |
| Context Relevance | how close the retrieved chunks are to the question (cosine of embeddings, 0 to 1) |
| Answer Relevance | how close the answer is to the question (cosine of embeddings, 0 to 1) |
| Groundedness | an LLM judge splits the answer into claims and marks each supported or not by the retrieved chunks; score = supported / total (refusals make no claims, so they are reported as the false refusal rate instead) |
| Context Recall | share of the ground-truth passage (`ground_truth_context`) found in the retrieved chunks |
| Hit@k, MRR | whether the expected page was in the top k chunks |
| Refusal accuracy | unanswerable questions refused, answerable ones answered |
| Answer F1 | word overlap with your ground-truth answer |

`--mock` gives made-up numbers for a quick smoke test. The report says so. Never use it for results.

## 5. Check that tables (and OCR) are read correctly

`sample/sample-tables.pdf` has 9 tables (pages 1-3) with known contents, written by hand in `evaluation/sample_tables_truth.json`. The check reads them two ways and scores both against that ground truth:

| Mode | What happens |
| :--- | :--- |
| `native` | the PDF's own text layer, tables found by pdfplumber (also proves the ground truth is right) |
| `ocr` | the same pages are turned into **pictures** (no text layer), then read with Tesseract OCR |

**Needs:** the venv from step 1 and Tesseract installed (Windows: the UB Mannheim build). If `tesseract` is not on PATH, set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`. The Windows install folder is also tried automatically. No Gemini key or internet is needed.

**Run the automatic test**
```bash
pytest tests/test_sample_tables_ocr.py -v -s
```
`-s` prints the score tables. The OCR tests are skipped (with a reason) when Tesseract cannot be found.

**Run the report on any PDF**
```bash
python -m evaluation.table_fidelity                      # native + OCR on sample/sample-tables.pdf
python -m evaluation.table_fidelity --mode ocr           # OCR only
python -m evaluation.table_fidelity --pdf "C:\my_pdfs\other.pdf"   # your own PDF (edit the truth file to match)
```

**How to read the scores**

| Column | Meaning |
| :--- | :--- |
| Cell recall | share of the expected cells found on the page (a misread number lowers it) |
| Row integrity | share of expected rows whose cells all sit on **one** row (a table row, or one line of text). This is the one that fails when OCR reads a table column by column |
| Rows intact after chunking | share of expected rows that are whole inside a single chunk after chunking |

The test passes when OCR cell recall is at least 90% for every table and mean row integrity is at least 85%.

**Result on the sample (Tesseract 5.4, 200 dpi)**

| | Cell recall | Row integrity |
| :--- | :---: | :---: |
| native | 100% | 100% |
| OCR, plain Tesseract text (before the table reader) | 96% | 62% |
| OCR with the table reader (`ocr_tables.py`) | 100% | 98% |

The one split row is an OCR character error (`11` read as `el`), not a layout error. OCR is not perfect: a different scan resolution changes which digits are misread (300 dpi was worse than 200 on this sample), so check numbers in important tables.

**Test your own table:** copy a table into `evaluation/sample_tables_truth.json` (page number, name, rows of cells), then run the command above with `--pdf`.

**Limits:** the table reader (`app/infrastructure/ingestion/ocr_tables.py`) rebuilds tables from word positions. It handles ruled and well-aligned tables with at least 3 rows. Merged cells, rotated tables and text laid out in two columns can still be misread. Only English OCR data is installed by default.

## 6. Student questions via MCP (mock data)

Questions about a student's timetable, invoices and attendance go to student-data tools (`app/services/student_tools.py`), also available as an MCP server. In standalone mode they read **mock data** from `sample/student_data.json` for the student `demo.student@shine.lk`. Nothing is sent to any other system.

```bash
python -m app.cli ask "When is my next class?"
python -m app.cli ask "Do I have any pending invoices?"
python -m app.cli ask "How is my attendance?"
python -m app.cli mcp
```
The last command starts the MCP server (it waits for an MCP client such as the MCP Inspector; press Ctrl+C to stop).

## 7. Command summary

| Command | What it does |
| :--- | :--- |
| `python -m app.cli load-sample` | index the included sample PDF |
| `python -m app.cli ingest "<file or folder>" [--course N]` | read, chunk, embed and index PDFs |
| `python -m app.cli ask "<question>" [--show-math] [--course N]` | answer with retrieval details |
| `python -m app.cli eval [--dataset file] [--limit N] [--output file]` | score the system |
| `python -m app.cli mcp` | start the student-data MCP server |
| `pytest tests/test_sample_tables_ocr.py -v -s` | check OCR and table extraction on `sample/sample-tables.pdf` |
| `python -m evaluation.table_fidelity [--mode native\|ocr] [--pdf file]` | table fidelity report |
| `python -m app.cli reset` | delete everything in `data/` |

## 8. Troubleshooting

| Message or problem | Fix |
| :--- | :--- |
| `GEMINI_API_KEY is empty` | edit `.env` and set the key (step 1), run from the project folder |
| `Chat is not available. Ingest a PDF first` | run `load-sample` or `ingest` first |
| `No PDF found at ...` | check the path, use double quotes, the file must end in `.pdf` |
| 429 / quota / rate limit errors | wait a minute and retry, or use `--limit 5` for `eval`; free Gemini keys have limits |
| Answer is "not found in your course material" | the PDFs do not contain it, or the PDF is scanned (needs Tesseract, set `TESSERACT_CMD` in `.env`) |
| OCR table test says `Tesseract is not installed` | install Tesseract and set `TESSERACT_CMD` in `.env` |
| `python` not found | on macOS/Linux use `python3`; on Windows reinstall Python and tick "Add to PATH" |
| `No module named ...` | the venv is not active: run the activate command again |
| Old results in the answers | `python -m app.cli reset`, then ingest again |

## 9. How it works

```
PDF -> extract (PyMuPDF text, Tesseract OCR for scans, pdfplumber tables)
    -> clean -> page-aware chunks
    -> Gemini embeddings -> Chroma (persistent, data/chroma)
    -> chunks JSON (data/chunks) -> TF-IDF index (rebuilt at start-up)

question -> Observe -> Reason -> Act -> Persist
                          |        |
                          |        +-> hybrid retriever = dense (Chroma) + sparse (TF-IDF), fused by RRF
                          |        +-> answer with citations -> self-check that refuses unsupported claims
                          +-> student-data intent -> tools (timetable, invoices, attendance)
```

- `app/domain`: interfaces (ports). `app/infrastructure`: Chroma, Gemini, TF-IDF, OCR. `app/services`: ingestion, answer, self-check and the agent stages. `app/container.py`: dependency injection. `evaluation/`: metrics and runner. More detail in `docs/`.
- Sparse search uses `TfidfVectorizer(norm=None)`; the dot product, magnitudes and cosine are computed explicitly in `app/infrastructure/retrieval/sparse_index.py`.
- The agent loop is Observe -> Reason -> Act -> Persist, where Act is a bounded loop (`MAX_ATTEMPTS = 2` in `app/services/agent/act.py`). Reason can split a multi-part question into up to 3 searches. If nothing is retrieved, or the self-check rejects the answer, the agent rewrites the query and tries once more, then refuses instead of guessing. Every turn returns an `AgentTrace` (`app/services/agent/state.py`) with the searches, scores, retries and self-check verdict.

## 10. Optional: Spring backend

Set `SPRING_BASE_URL` and `INTERNAL_SECRET` in `.env` to read real student records and to use the web API (`uvicorn app.main:app`). See `docs/API_REFERENCE.md` and `docs/STUDENT_MCP.md`. Not needed for any step above.

## 11. For the submission

Send: the source code, `requirements.txt`, `.env.example`, this README, the GitHub link, the report and the video link.
**Do not send** your `venv` folder (huge and tied to your computer), your real `.env` (contains your private key), `data/`, `__pycache__/` or `.git/`.
