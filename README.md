# shine-ai-service

AI microservice for Shine Institute (Python, FastAPI). It will read the PDFs that teachers upload to course modules and let students ask questions about the courses they have paid for, with page citations.

The scope and build order are described in `Shine-AI-Chatbot-Guide.pdf` (Course Study Assistant proposal).

## Status

Project foundation only: repository, folder architecture and the connection to the Spring backend (JWT check, enrollment lookup, internal index hook). The AI features (PDF reading, search, chat, quizzes, reports) are built in later phases.

## Branches

- `main`: stable, tagged releases
- `develop`: integration branch
- `feature/<name>`: one branch per step, merged into `develop`
