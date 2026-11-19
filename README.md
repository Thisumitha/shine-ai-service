# Shine AI Service: Explainable Agentic RAG

Ask questions about PDF files and get answers that **show their working**: the chunks that were retrieved, the similarity maths behind them (TF-IDF dot product, vector magnitudes, cosine, Reciprocal Rank Fusion) and automatic scores for Context Relevance, Groundedness and Answer Relevance.

It runs **on its own**: no Spring backend, no MySQL, no login. You only need Python and a free Gemini key.

## Quick reference

| Task | Action | Section |
| :--- | :--- | :---: |
| Install it for the first time | create the venv, install, put the key in `.env` | [2](#2-set-up) |
| Run a first example | `load-sample`, then `ask` | [3](#3-try-it-with-the-included-pdfs) |
| Use my own PDFs | `reset`, `ingest`, `ask` | [4](#4-use-your-own-pdfs) |
| Measure how good the answers are | `eval` with a dataset | [6](#6-evaluation-how-good-are-the-answers) |
| Check the code is not broken | `pytest` (279 tests, no key needed) | [7](#7-automatic-tests) |
| Check tables and scanned pages are read | `table_fidelity` | [8](#8-tables-and-ocr-check) |
| Keep answering when Gemini is busy | automatic fallback, Ollama | [9](#9-when-gemini-is-busy-or-over-quota) |
| Find a command | cheat sheet | [10](#10-command-cheat-sheet) |
| Fix an error message | troubleshooting | [11](#11-troubleshooting) |

**Contents:** Part 1 Get started (1-3) · Part 2 Use it (4-5) · Part 3 Check quality (6-8) · Part 4 Availability (9) · Part 5 Reference (10-14)

---

# Part 1. Get started

## 1. What you need

| Need | Check |
| :--- | :--- |
| Python 3.10 or newer | `python --version` (on macOS/Linux try `python3 --version`) |
| A Google Gemini API key (free) | https://aistudio.google.com/apikey |
| Internet | the key is used for embeddings and answers |
| Optional: Tesseract OCR | only for scanned PDFs; normal PDFs work without it |
| Optional: Ollama | a free local model used as the last fallback (section 9) |

> Do not look for a ready-made `venv` folder in this project. A venv is created on your own computer in section 2.

## 2. Set up

This is done once. Open a terminal **inside this folder** (the one that contains `README.md`).

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
- `pip install` prints many lines and ends without the word `ERROR`.
- In `.env`, change `GEMINI_API_KEY=` to `GEMINI_API_KEY=your_key_here`, save and close. Leave every other line as it is.

If PowerShell says *running scripts is disabled* when you activate, run the following once and try again:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Every new terminal window needs `venv\Scripts\activate` (or `source venv/bin/activate`) again.

> **Note:** `copy .env.example .env` **overwrites** an existing `.env`, including the key. Run it only during the first setup.

## 3. Try it with the included PDFs

The `sample/` folder has two ready-made sets. Each one is a separate **course**, so their answers never mix.

| Set | Folder | Course | Load it with | Matching questions |
| :--- | :--- | :---: | :--- | :--- |
| Software engineering | `sample/` (`Introduction_to_Software_Engineering.pdf`, `sample-tables.pdf`) | `1` | `python -m app.cli load-sample` | `evaluation/sample_dataset.jsonl` (16) |
| Business studies | `sample/business/` (`The_Market.pdf`, `Marketing_Objectives_and_Strategy.pdf`, `Product_and_Service_Design.pdf`) | `101` | `python -m app.cli load-sample --set business` | `evaluation/dataset.jsonl` (40) |

`sample/business/Preparing_Questionnaire.docx` is only kept for reference. Ingestion reads **PDF files only**, so it is skipped.

**Software set**
```bash
python -m app.cli load-sample
python -m app.cli ask "What is the software crisis?" --show-math
python -m app.cli eval --limit 5
```

**Business set** (note `--course 101`)
```bash
python -m app.cli load-sample --set business
python -m app.cli ask "What are the three core elements of the Design Mix?" --show-math --course 101
python -m app.cli eval --dataset evaluation/dataset.jsonl --limit 5
```

`ask` tests one question. `eval --limit 5` scores 5 questions of the matching dataset: the first answerable ones **plus 1 question the PDFs cannot answer**, so the hallucination check (the system must refuse) is part of even a small run. Remove `--limit 5` to run the whole dataset (16 questions for the software set, 40 for the business set).

What `load-sample` prints: `[INGEST] ...` per file and a line like `22 chunks (pages 1-18) embedded into Chroma and indexed with TF-IDF`.

What `eval` prints: each question as it runs (pages retrieved against the expected page, a short answer and its scores), then a summary table with Context Relevance, Answer Relevance and Groundedness. See section 6 for how to read it.

What `ask` prints, in order:

| Block | Meaning |
| :--- | :--- |
| `[OBSERVE]` | your question |
| `[REASON]` | the intent the system decided on; `[PLAN]` appears when a multi-part question is split into several searches |
| `[RETRIEVE]` | the chunks each search returned **before** any answer, with RRF, dense and TF-IDF scores |
| `[LOOP]` / `[CHECK]` | retries (a rewritten query) and the self-check verdict |
| `[ANSWER]` | the answer with its source pages |
| `[PERSIST]` | the conversation was saved |

---

# Part 2. Use it

## 4. Use your own PDFs

1. Put your PDF files in one folder, for example `C:\my_pdfs`.
2. Start clean (this deletes the earlier index in `data/`) and ingest them:
```bash
python -m app.cli reset
python -m app.cli ingest "C:\my_pdfs"
```
You can also ingest one file: `python -m app.cli ingest "C:\my_pdfs\notes.pdf"`. **Put paths in double quotes**, especially if they have spaces.

3. Ask. **Put the question in double quotes.**
```bash
python -m app.cli ask "Your question about the PDF here" --show-math
```
Use `--course N` in both `ingest` and `ask` if you keep several courses apart (default is course `1`).

### How to read the retrieval output (numbers are only an example)

`--show-math` prints the cosine similarity of every chunk, with its workings:

```
#1 notes.pdf p.5  RRF=0.0325  dense_rank=1 sparse_rank=2
      dense: cosine distance = 0.2310, cosine similarity = 1 - distance = 0.7690
      TF-IDF dot=3.4120  |q|=2.1000  |d|=6.8000  cosine = dot/(|q||d|) = 0.2389
#2 notes.pdf p.9  RRF=0.0161  dense_rank=2 sparse_rank=None
      dense: cosine distance = 0.2880, cosine similarity = 1 - distance = 0.7120
      TF-IDF dot=1.9000  |q|=2.1000  |d|=5.5000  cosine = dot/(|q||d|) = 0.1645 (keyword rank 5, outside the top 4)
```

| Part | Meaning |
| :--- | :--- |
| `#1`, `p.5` | the best chunk and its page in the PDF |
| `dense_rank`, `sparse_rank` | its position in the meaning-based (embedding) search and in the TF-IDF keyword search. `None` means the chunk was not among that search's top 4 |
| `RRF` | the two ranks fused as `1/(60+rank)` for each list, summed. Higher is better |
| `dense: cosine similarity` | how close the question and chunk are in meaning (0 to 1): `1 - cosine distance`. Shown when the meaning-based search returned the chunk |
| `TF-IDF cosine` | how well the question's words match the chunk (0 to 1): `dot / (|q| x |d|)`, where `dot` is the dot product and `|q|`, `|d|` are the lengths of the question and chunk vectors |
| `(keyword rank 5, outside the top 4)` | the chunk was cut off by the keyword search, so it earns no RRF points from it. Its TF-IDF cosine is still calculated and shown, so every chunk has one |
| worked example | printed below the chunks: every shared word with its idf, query weight, chunk weight and product. The products add up to the dot product |

Use `--show-scores` instead for a short version, one line per chunk with only the two cosine similarity values:

```
#1 notes.pdf p.5  RRF=0.0325  dense_rank=1 sparse_rank=2
      dense cosine similarity 0.7690 | TF-IDF cosine similarity 0.2389
```

## 5. Student questions via MCP (mock data)

Questions about a student's timetable, invoices and attendance go to student-data tools (`app/services/student_tools.py`), also available as an MCP server. In standalone mode they read **mock data** from `sample/student_data.json` for the student `demo.student@shine.lk`. Nothing is sent to any other system.

```bash
python -m app.cli ask "When is my next class?"
python -m app.cli ask "Do I have any pending invoices?"
python -m app.cli ask "How is my attendance?"
python -m app.cli mcp
```
The last command starts the MCP server (it waits for an MCP client such as the MCP Inspector; press Ctrl+C to stop).

---

# Part 3. Check quality

## 6. Evaluation: how good are the answers?

Evaluation asks a list of questions whose answers you already know, then scores what the system did.

### Step A. Pick a dataset

| File | Questions | Course | Load its PDFs with |
| :--- | :---: | :---: | :--- |
| `evaluation/sample_dataset.jsonl` | 16 | `1` | `python -m app.cli load-sample` |
| `evaluation/dataset.jsonl` | 40 (35 answerable + 5 not) | `101` | `python -m app.cli load-sample --set business` |

`dataset.jsonl` covers the three business PDFs (18 questions on Marketing Objectives and Strategy, 9 on The Market, 8 on Product and Service Design) and adds a `topic` to each line. Its 5 unanswerable questions have the topic `Out of Syllabus` (FIFA World Cup, Maxwell equations, Apache Kafka ...); the system should **refuse** them. It has no `ground_truth_context`, so **Context Recall shows `n/a`** for it; every other score works.

The dataset's `course_id` must equal the course you ingested, otherwise every answer is "not found".

### Step B. Run it

```bash
python -m app.cli eval --limit 5                                            # 4 answerable + 1 refusal question
python -m app.cli eval --dataset evaluation/dataset.jsonl --limit 5         # 4 answerable + 1 refusal question
python -m app.cli eval --dataset evaluation/dataset.jsonl --output biz_report.md   # all 40, report saved
python -m app.cli eval --config all                                         # compare 4 setups (below)
python -m app.cli eval --mock                                               # fake numbers, no key (smoke test only)
```

While it runs, each question shows the pages retrieved against the expected page, a short answer and its scores. The end is a summary table. `--config all` runs the four ablations: dense only, TF-IDF only, hybrid, hybrid + self-check, each with its own retriever. `--mock` numbers are made up and the report says so. Never use them as results.

### Step C. Read the result

```
 Retrieval   Hit@1 100%   Hit@3 100%   Hit@5 100%   MRR 1.00
 RAG quality Context relevance 0.71   Groundedness 100%   Answer relevance 0.79   Context recall 1.00
 Safety      Refusal accuracy 100%   False refusals 0%
 Overlap     Answer F1 vs ground truth 56%
```

| Score | Meaning | Good value |
| :--- | :--- | :--- |
| Hit@k | the expected page was among the top k chunks | high |
| MRR | `1.00` = the right page was always first | close to 1 |
| Context Relevance | how close the retrieved chunks are to the question (cosine of embeddings, 0 to 1) | higher |
| Answer Relevance | how close the answer is to the question (cosine of embeddings, 0 to 1) | higher |
| **Groundedness** | an LLM judge splits the answer into claims and marks each supported or not by the retrieved chunks. This is the "no invented facts" score | near 100% |
| Context Recall | share of your `ground_truth_context` found in the retrieved chunks | higher |
| Refusal accuracy | unanswerable questions refused, answerable ones answered | 100% |
| False refusals | answerable questions it wrongly refused | 0% |
| Answer F1 | word overlap with your ground-truth answer. A correct answer worded differently scores low | trust the others first |

**Order of importance:** Groundedness and Hit@k first, then Refusal accuracy. Answer F1 last.

**What was checked in this project**

| Run | Result |
| :--- | :--- |
| Sample set, 3 questions (live) | Hit@1 100%, Groundedness 100%, Context recall 1.00, no false refusals |
| Business set, 3 answerable (live) | right page found, Groundedness 1.00 for all 3 |
| Business set, 2 out-of-syllabus (live) | both refused correctly (refusal accuracy 100%) |
| `dataset.jsonl` file check | 40 rows, unique ids, every answerable row has pages, no unanswerable row has pages |

These are small spot checks. Run the whole dataset for real numbers.

### Check a dataset file without running anything

```bash
python -c "import json;r=[json.loads(l) for l in open('evaluation/dataset.jsonl',encoding='utf-8') if l.strip()];print(len(r),'rows,',sum(x['is_answerable'] for x in r),'answerable'); assert len({x['id'] for x in r})==len(r),'duplicate ids'"
```
Expected: `40 rows, 35 answerable`.

### Write your own questions

1. Copy `evaluation/sample_dataset.jsonl` to a new file, for example `my_questions.jsonl`.
2. One line is one question:
```json
{"id": "q1", "course_id": 1, "question": "What is X?", "ground_truth_answer": "X is ...", "ground_truth_context": "the passage that states it", "expected_source_doc": "notes.pdf", "expected_page_numbers": [3], "is_answerable": true}
```
3. For a question the PDF cannot answer use `"is_answerable": false`, `"expected_page_numbers": []` and an empty `ground_truth_answer`.
4. Run: `python -m app.cli eval --dataset my_questions.jsonl --output report.md`

### Usage limits (read before running the full set)

- One question makes about **4 to 6 model calls** (classify, answer, self-check, groundedness judge) plus several embedding calls, so the 40 set needs roughly 160 to 240 model calls.
- A free Gemini key has low quotas, for example a daily request limit on the main model and a rate limit on embedding requests. Start with `--limit 5`.
- When one model is out of quota the service switches to the next one by itself (section 9) and the run continues, only slower.
- **Context Relevance** and **Answer Relevance** are computed with Gemini embeddings. When the embedding quota is used up, these two scores show `n/a` (one warning is printed) and the run continues: retrieval, answers, Groundedness and the other scores are still produced, and the model fallback (including Ollama) applies. Run again later to get the two missing scores.
- Avoid ingesting and evaluating at the same time, because both use the embedding quota.
- If only the local Ollama model answers, the judge uses that same small model, so Groundedness is less reliable. For the final report, run when Gemini works.

## 7. Automatic tests

279 unit tests. They need **no Gemini key, no internet and no Ollama** because the model calls are replaced by fakes. Run them from the project folder with the venv active.

```bash
pytest                                  # all tests; expect "279 passed"
pytest -q                               # same, shorter output
pytest -x                               # stop at the first failure
pytest --collect-only -q                # list the tests without running them
pytest tests/infrastructure -v          # one folder, one line per test
pytest tests/infrastructure/test_gemini_client.py -v          # one file
pytest tests/infrastructure/test_gemini_client.py::test_falls_back_to_next_model_in_order   # one test
pytest -k "fallback or ollama" -v       # every test whose name matches
```

| Folder or file | What it checks |
| :--- | :--- |
| `tests/infrastructure/test_gemini_client.py` | Gemini calls, retries, daily-quota skip, model fallback order |
| `tests/infrastructure/test_ollama_client.py` | the local Ollama client (against a fake Ollama server) |
| `tests/infrastructure/test_fallback_client.py` | Gemini first, Ollama as the last resort, cool-down after a failure |
| `tests/infrastructure/test_claude_chain.py` | the model chain, immediate switching, cool-downs, the Claude Code client (against a fake `claude` command) |
| `tests/infrastructure/test_error_messages.py` | plain-language error messages (no raw API errors) |
| `tests/infrastructure/test_embedding_outage.py` | keyword-only search when embeddings fail, no retry of a used-up daily quota, skipping of indexed files |
| `tests/infrastructure/test_network_retries.py` | network retries |
| `tests/services`, `tests/api`, `tests/domain` | the agent stages, the web API, the entities |
| `tests/test_*.py` | chunking, ingestion, OCR, evaluation, MCP server, end-to-end chat |

The tests work in temporary folders with a fake embedder (`tests/conftest.py`), so running them never changes `data/` (your indexed PDFs) and never uses Gemini quota.

Some tests are **skipped** (not failed) when Tesseract is not installed. If a test fails, run it alone with `pytest <file>::<test> -v` to see why. The tests read your `.env`, so tests that depend on a setting pass it explicitly.

## 8. Tables and OCR check

`sample/sample-tables.pdf` has 9 tables (pages 1-3) with known contents, written by hand in `evaluation/sample_tables_truth.json`. The check reads them two ways and scores both against that ground truth:

| Mode | What happens |
| :--- | :--- |
| `native` | the PDF's own text layer, tables found by pdfplumber (also proves the ground truth is right) |
| `ocr` | the same pages are turned into **pictures** (no text layer), then read with Tesseract OCR |

**Needs:** the venv from section 2 and Tesseract installed (Windows: the UB Mannheim build). If `tesseract` is not on PATH, set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`. The Windows install folder is also tried automatically. No Gemini key or internet is needed.

```bash
pytest tests/test_sample_tables_ocr.py -v -s                       # automatic test; -s prints the score tables
python -m evaluation.table_fidelity                                # native + OCR on sample/sample-tables.pdf
python -m evaluation.table_fidelity --mode ocr                     # OCR only
python -m evaluation.table_fidelity --pdf "C:\my_pdfs\other.pdf"   # your own PDF (edit the truth file to match)
```
The OCR tests are skipped (with a reason) when Tesseract cannot be found.

| Column | Meaning |
| :--- | :--- |
| Cell recall | share of the expected cells found on the page (a misread number lowers it) |
| Row integrity | share of expected rows whose cells all sit on **one** row. This is the one that fails when OCR reads a table column by column |
| Rows intact after chunking | share of expected rows that are whole inside a single chunk after chunking |

The test passes when OCR cell recall is at least 90% for every table and mean row integrity is at least 85%.

**Result on the sample (Tesseract 5.4, 200 dpi)**

| | Cell recall | Row integrity |
| :--- | :---: | :---: |
| native | 100% | 100% |
| OCR, plain Tesseract text (before the table reader) | 96% | 62% |
| OCR with the table reader (`ocr_tables.py`) | 100% | 98% |

The one split row is an OCR character error (`11` read as `el`), not a layout error. A different scan resolution changes which digits are misread (300 dpi was worse than 200 on this sample), so check numbers in important tables.

**Test your own table:** copy a table into `evaluation/sample_tables_truth.json` (page number, name, rows of cells), then run the command above with `--pdf`.

**Limits:** the table reader (`app/infrastructure/ingestion/ocr_tables.py`) rebuilds tables from word positions. It handles ruled and well-aligned tables with at least 3 rows. Merged cells, rotated tables and text laid out in two columns can still be misread. Only English OCR data is installed by default.

---

# Part 4. Availability

## 9. When Gemini is busy or over quota

Free Gemini keys have low quotas. The service switches models automatically and prints one plain line when it does, for example:
`gemini-3.8-flash: the free daily limit is used up, resets in about 3h. Switching to claude-code.`

**One switch: `USE_GEMINI`.** Set it in `.env` to choose who writes the answers:

| `USE_GEMINI` | Answers, question classification, self-check and the evaluation judge run on |
| :--- | :--- |
| `true` | **Gemini only**: `GEMINI_MODEL`, then the models in `GEMINI_FALLBACK_MODELS`. No Claude. |
| `false` | **Claude Code only**: the `claude` command, signed in with your Claude account. No Gemini model writes answers. |
| `auto` (default) | the model chain below, for example Gemini first and Claude Code if it fails |

Embeddings (indexing PDFs and the meaning-based half of the search) use Gemini in **every** mode, because Claude has no embedding model. With `false`, a question still makes one Gemini embedding call; if that is unavailable the search continues with keyword search only. Restart the command after changing `.env`. `false` is slower: a question makes several Claude calls, and a rejected self-check adds more.

**The model chain.** `LLM_CHAIN` in `.env` lists the answer models in order. The first one that works answers; a model that fails is skipped for a short cool-down (longer for a used-up daily quota) so it does not slow every question.

```
LLM_CHAIN=gemini:gemini-3.8-flash,claude-code,gemini:gemini-3.5-flash-lite,ollama
```

| Entry | What it is | Notes |
| :--- | :--- | :--- |
| `gemini:<model>` | a Gemini model | an entry with another one behind it gives up after `LLM_FAST_SWITCH_ATTEMPTS` (default 1), so the switch is immediate |
| `claude-code` | the `claude` command-line tool, signed in with your own Claude account | text only: all tools are turned off, it cannot read files or run commands |
| `ollama` | the local model (set up below) | free, no quota |

If `LLM_CHAIN` is empty, the older settings apply: `GEMINI_MODEL`, then `GEMINI_FALLBACK_MODELS`, then `LLM_FALLBACK_PROVIDER=ollama`.

**Claude Code as a fallback.** Install Claude Code and sign in once (`claude`, then `/login`); check with `claude --version`. Settings: `CLAUDE_CODE_MODEL` (default `haiku`, the fastest), `CLAUDE_CODE_TIMEOUT_SECONDS` and `CLAUDE_CODE_EXTRA_ARGS` (default `--restricted --effort low`). Know before you enable it:
- The question and the retrieved course text are **sent to Anthropic**, and each call is **billed to the Claude account** that is signed in (a few cents per question on a pay-as-you-go account, or counted against a subscription's usage).
- A call takes a few seconds, and one question makes several calls (classification, answer, self-check).
- It is meant for development and demonstrations. A production service should use an API key with its own limits, not a personal sign-in.
- To turn it off, remove `claude-code` from `LLM_CHAIN`.

How it behaves:
- A busy server (503/504) is retried with growing waits (`GEMINI_BACKOFF_SECONDS`, `GEMINI_MAX_RETRIES`). A chain entry with another one behind it gives up after `LLM_FAST_SWITCH_ATTEMPTS`; with the older settings it is tried twice.
- A **daily** quota is not retried, because it stays exhausted. That model is skipped on later calls.
- A call that hangs is stopped after `GEMINI_TIMEOUT_SECONDS`.
- After Gemini fails, Ollama answers without trying Gemini again for a cool-down period (`LLM_FALLBACK_COOLDOWN_SECONDS`).
- A bad request (400) does not fall back, because another model would fail the same way.
- **Embeddings always use Gemini**, even when Ollama answers. A local model cannot replace them automatically, because vectors from different models are not comparable and the index would have to be rebuilt.
- If embeddings are unavailable when a **question** is asked (quota or outage), the search continues with **keyword (TF-IDF) search only** and the answer is still produced. One plain line reports this.
- If embeddings are unavailable while **indexing** a PDF, the command stops with one plain message and saves nothing for that file. Files that are already indexed are skipped on the next run.

**Set up Ollama (free, local, no limits)**
```bash
ollama serve                 # skip if the Ollama app is already running
ollama pull qwen2.5:3b       # small model, runs on a standard laptop
ollama list                  # qwen2.5:3b should be listed
```
`LLM_PROVIDER=ollama` makes Ollama answer **every** question instead of Gemini. A 3B model is slower on a CPU and less accurate than Gemini; try `OLLAMA_MODEL=llama3.1:8b` if you have 16 GB of RAM or more.

Set `SHINE_DEBUG=1` to see the technical error text that is normally hidden.

---

# Part 5. Reference

## 10. Command cheat sheet

**Load documents**

| Command | What it does |
| :--- | :--- |
| `python -m app.cli load-sample [--force]` | index the software sample PDFs (course 1); files already indexed are skipped |
| `python -m app.cli load-sample --set business` | index the three business PDFs (course 101) |
| `python -m app.cli ingest "<file or folder>" [--course N] [--force]` | read, chunk, embed and index PDFs; files already indexed are skipped unless `--force` is given |
| `python -m app.cli reset` | delete everything in `data/` |

**Ask**

| Command | What it does |
| :--- | :--- |
| `python -m app.cli ask "<question>" [--show-math] [--show-scores] [--course N]` | answer with retrieval details; `--show-math` prints the cosine similarity workings of every chunk, `--show-scores` only the values |
| `python -m app.cli mcp` | start the student-data MCP server |

**Evaluate and test**

| Command | What it does |
| :--- | :--- |
| `python -m app.cli eval [--dataset file] [--limit N] [--output file] [--config all] [--mock]` | score the system (default dataset: `evaluation/sample_dataset.jsonl`) |
| `python -m app.cli eval --dataset evaluation/dataset.jsonl --limit 5` | score the business set (load it first) |
| `pytest` | run all 279 unit tests (no key or internet needed) |
| `pytest -k "fallback or ollama" -v` | run only the tests whose name matches |
| `pytest tests/test_sample_tables_ocr.py -v -s` | check OCR and table extraction |
| `python -m evaluation.table_fidelity [--mode native\|ocr] [--pdf file]` | table fidelity report |

**Ollama**

| Command | What it does |
| :--- | :--- |
| `ollama serve` | start the local model server |
| `ollama pull qwen2.5:3b` | download the small model |
| `ollama list` | show installed models |

## 11. Troubleshooting

| Message or problem | Fix |
| :--- | :--- |
| `GEMINI_API_KEY is empty` | edit `.env` and set the key (section 2), run from the project folder |
| `Chat is not available. Ingest a PDF first` | run `load-sample` or `ingest` first |
| `No PDF found at ...` | check the path, use double quotes, the file must end in `.pdf` (a `.docx` is not read) |
| `Could not index <file>: the free daily limit is used up` | the embedding quota is exhausted. Nothing was saved for that file; run the same command again later (indexed files are skipped) |
| Every answer is "not found" for the business questions | you loaded the wrong course: use `load-sample --set business` and `--course 101` |
| `the free daily limit is used up` | handled automatically: the next model is used (section 9). The quota resets daily |
| `Google's servers are busy right now` | handled automatically: retry, then the next model. If every model fails, try again later |
| `Context relevance n/a` / `Answer relevance n/a` in the report | Gemini embeddings were unavailable (quota). The other scores are valid; run again later for these two |
| `Claude Code is not installed or not on PATH` / `Claude Code could not answer` | install Claude Code and sign in (`claude`, then `/login`), or set `USE_GEMINI=true` / remove `claude-code` from `LLM_CHAIN` |
| `USE_GEMINI must be true, false or auto` | fix the value in `.env` |
| `Cannot reach Ollama` | start it with `ollama serve`, then `ollama pull qwen2.5:3b` |
| `Could not get an answer: ...` | the message says why in plain words; set `SHINE_DEBUG=1` for the technical detail |
| Answer is "not found in your course material" | the PDFs do not contain it, or the PDF is scanned (needs Tesseract, set `TESSERACT_CMD` in `.env`) |
| OCR table test says `Tesseract is not installed` | install Tesseract and set `TESSERACT_CMD` in `.env` |
| `python` not found | on macOS/Linux use `python3`; on Windows reinstall Python and tick "Add to PATH" |
| `No module named ...` | the venv is not active: run the activate command again |
| Old results in the answers | `python -m app.cli reset`, then ingest again |
| Your key disappeared from `.env` | `copy .env.example .env` overwrites it; put the key back |

## 12. How it works

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

- `app/domain`: interfaces (ports). `app/infrastructure`: Chroma, Gemini, Ollama, TF-IDF, OCR. `app/services`: ingestion, answer, self-check and the agent stages. `app/container.py`: dependency injection. `evaluation/`: metrics and runner. More detail in `docs/`.
- Sparse search uses `TfidfVectorizer(norm=None)`; the dot product, magnitudes and cosine are computed explicitly in `app/infrastructure/retrieval/sparse_index.py`.
- The agent loop is Observe -> Reason -> Act -> Persist, where Act is a bounded loop (`MAX_ATTEMPTS = 2` in `app/services/agent/act.py`). Reason can split a multi-part question into up to 3 searches. If nothing is retrieved, or the self-check rejects the answer, the agent rewrites the query and tries once more, then refuses instead of guessing. Every turn returns an `AgentTrace` (`app/services/agent/state.py`) with the searches, scores, retries and self-check verdict.
- The model behind the answers is an `LLMClient` (`app/domain/ports/llm_client.py`). `GeminiClient` (with model fallback), `OllamaClient` and `FallbackLLMClient` (Gemini first, Ollama last) implement it.

## 13. Optional: Spring backend

Everything above runs without Spring. To use the service from the Spring Boot backend (students ask questions through the web app), connect the two as follows.

**What Spring does:** it checks the student's login and enrollment, then calls this service. This service never talks to students directly. Spring sends PDFs when they are uploaded and sends questions with the list of courses the student may search.

| Direction | Call | Purpose |
| :--- | :--- | :--- |
| Spring to this service | `POST /internal/chat` | ask a question: `{question, courseIds, sessionId?, studentEmail?}` |
| Spring to this service | `POST /internal/index` | send a PDF to be read and indexed (answers with `202` and works in the background) |
| Spring to this service | `GET` / `DELETE /internal/index/{itemId}` | check the index status of a PDF, or remove it |
| This service to Spring | `GET /internal/student/timetable`, `/invoices`, `/attendance` | personal questions ("When is my next class?") |

Every one of these calls carries the header `X-Internal-Secret`. The reply of `/internal/chat` is `{answer, sessionId, sources, is_refusal}`.

**Settings that must match**

| Where | Setting | Value |
| :--- | :--- | :--- |
| this service, `.env` | `INTERNAL_SECRET` | the same secret on both sides. If it is empty, every call from Spring is refused with 401 |
| this service, `.env` | `SPRING_BASE_URL` | the address of Spring, for example `http://localhost:8080`. If it is empty, personal questions use the mock data in `sample/student_data.json` |
| Spring | `AI_SERVICE_SECRET` | the same value as `INTERNAL_SECRET` |
| Spring | `AI_SERVICE_URL` | the address of this service, for example `http://localhost:8000` |
| Spring | `AI_SERVICE_TIMEOUT` | seconds Spring waits for an answer (default 180). An answer written by Claude Code or Ollama can take over a minute, so do not set it low |

Use a long random secret outside local development.

**Start both**
```bash
# this service (from this folder, venv active)
uvicorn app.main:app --port 8000

# Spring (from the backend folder), with the same secret
AI_SERVICE_SECRET=<the same secret> AI_SERVICE_URL=http://localhost:8000 mvn spring-boot:run
```

**Check the connection without Spring**
```bash
curl http://localhost:8000/health                                     # {"status":"ok",...}
curl -X POST http://localhost:8000/internal/chat -H "Content-Type: application/json" \
     -d '{"question":"What is the software crisis?","courseIds":[1]}'  # 401 without the secret header
curl -X POST http://localhost:8000/internal/chat -H "Content-Type: application/json" \
     -H "X-Internal-Secret: <secret>" -d '{"question":"What is the software crisis?","courseIds":[1]}'
```
`courseIds` must be courses whose PDFs were indexed (`1` for the software sample, `101` for the business sample). A question the PDFs cannot answer comes back with `"is_refusal": true`.

Also see `docs/API_REFERENCE.md` and `docs/STUDENT_MCP.md`.

## 14. For the submission

Send: the source code, `requirements.txt`, `.env.example`, this README, the GitHub link, the report and the video link.
**Do not send** your `venv` folder (huge and tied to your computer), your real `.env` (contains your private key), `data/`, `__pycache__/` or `.git/`.
