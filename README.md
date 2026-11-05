# shine-ai-service

AI microservice for Shine Institute (Python, FastAPI). It will read the PDFs that teachers upload to course modules and let students ask questions about the courses they have paid for, with page citations.

The scope and build order are described in `Shine-AI-Chatbot-Guide.pdf` (Course Study Assistant proposal).

## Status

Project foundation only: repository, folder architecture and the connection to the Spring backend (JWT check, enrollment lookup, internal index hook). The AI features (PDF reading, search, chat, quizzes, reports) are built in later phases.

## PDF extraction

An uploaded PDF is queued, then read page by page: text (PyMuPDF), OCR for scanned pages (Tesseract), tables as Markdown (pdfplumber), then cleaned. The result is saved as `data/extracted/<itemId>.json` and the job ends as `EXTRACTED`. The text is then cut into page-aware chunks saved as `data/chunks/<itemId>.json` (see `docs/CHUNKING.md`). Embedding comes next.

- Install Tesseract (Windows: the UB Mannheim installer) and set `TESSERACT_CMD` and `OCR_LANGUAGE` in `.env`. Without it, text PDFs still work; scanned pages come back empty.
- Try a PDF by hand: `POST /internal/parse-pdf` in `/docs` (needs the `X-Internal-Secret` header).

## Branches

- `main`: stable, tagged releases
- `develop`: integration branch
- `feature/<name>`: one branch per step, merged into `develop`

## For the team

- `docs/TEAM_GUIDE.md`: setup, what to write in each folder and file, git workflow
- `docs/ARCHITECTURE.md`: the layer rules in short
