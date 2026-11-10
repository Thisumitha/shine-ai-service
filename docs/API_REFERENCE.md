# REST API Reference

This document specifies the HTTP REST API exposed by `shine-ai-service`.

---

## 1. Base URL & Ports

- **Base URL**: `http://localhost:8000` (configurable via `PORT` in `.env`)
- **Interactive Documentation**:
  - Swagger UI: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`

---

## 2. Authentication

Requests use two authentication mechanisms depending on the route:

1. **Student / Client Routes (`/api/v1/ai/*`)**:
   - Header: `Authorization: Bearer <jwt_token>`
   - Authenticated by `api.deps.get_current_user` using `JWT_SECRET`.
   - Extracts student identity (`email`, `user_id`, `role`).

2. **Internal Server Routes (`/internal/*`)**:
   - Header: `X-Internal-Secret: <internal_secret>`
   - Used for private service-to-service communication with Spring Boot backend (e.g., triggering indexing).

---

## 3. Endpoints

### A. Health & Diagnostics

#### `GET /health`
Returns service availability status.

- **Response `200 OK`**:
```json
{
  "status": "ok"
}
```

---

### B. Chat & AI Assistant

#### `POST /api/v1/ai/chat`
Send a question to the 4-stage AI Chat Agent.

- **Request Body**:
```json
{
  "session_id": "sess-456",
  "course_id": 101,
  "message": "What is binary search and what is its time complexity?"
}
```

- **Response `200 OK`**:
```json
{
  "session_id": "sess-456",
  "answer": "Binary search is an algorithm that searches a sorted array in O(log n) time by repeatedly dividing the search space in half.",
  "citations": [
    {
      "source_doc": "Algorithms_101.pdf",
      "page_number": 14,
      "quote": "Binary search divides the sorted array in half each step in O(log n)."
    }
  ],
  "is_refusal": false
}
```

---

### C. Chat History & Sessions

#### `GET /api/v1/ai/sessions`
List all active and past chat sessions for the authenticated student.

- **Query Parameters**:
  - `course_id` *(optional int)*: Filter sessions by course.
- **Response `200 OK`**:
```json
[
  {
    "session_id": "sess-456",
    "course_id": 101,
    "title": "Binary search inquiry",
    "created_at": "2026-11-09T18:30:00Z",
    "updated_at": "2026-11-09T18:35:00Z"
  }
]
```

#### `GET /api/v1/ai/sessions/{session_id}`
Retrieve the full message history and citations for a specific chat session.

- **Response `200 OK`**:
```json
{
  "session_id": "sess-456",
  "course_id": 101,
  "messages": [
    {
      "id": 1,
      "sender": "user",
      "text": "What is binary search?",
      "citations": [],
      "created_at": "2026-11-09T18:30:00Z"
    },
    {
      "id": 2,
      "sender": "assistant",
      "text": "Binary search is an efficient search algorithm...",
      "citations": [
        {
          "source_doc": "Algorithms_101.pdf",
          "page_number": 14,
          "quote": "Binary search operates in O(log n) time."
        }
      ],
      "created_at": "2026-11-09T18:30:02Z"
    }
  ]
}
```

---

### D. User Feedback

#### `POST /api/v1/ai/feedback`
Submit student feedback (+1 thumbs-up or -1 thumbs-down) for an assistant's response.

- **Request Body**:
```json
{
  "message_id": 2,
  "score": 1,
  "comment": "Very clear explanation and accurate slide citation."
}
```

- **Response `200 OK`**:
```json
{
  "status": "success",
  "feedback_id": 1
}
```

---

### E. Spring-Facing Internal Chat

#### `POST /internal/chat`
Called by the Spring Boot backend when an enrolled student asks a question. Spring handles authentication, verifies student enrollment, and passes authorized course IDs.

- **Headers**: `X-Internal-Secret: <secret>`
- **Request Body**:
```json
{
  "question": "What is photosynthesis?",
  "courseIds": [12, 15],
  "sessionId": "optional-session-id"
}
```

- **Response `200 OK`**:
```json
{
  "answer": "Photosynthesis is the process by which green plants convert light into chemical energy.",
  "sessionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "sources": [
    {
      "courseId": 12,
      "itemId": 345,
      "materialTitle": "Biology Notes",
      "page": 4,
      "snippet": "Photosynthesis is the process..."
    }
  ],
  "is_refusal": false
}
```

- **Notes**:
  - `courseIds`: Searched in a single call across all courses using the hybrid retriever (`course_ids: set[int]`).
  - `sessionId`: If omitted in request, a new UUID is generated and returned so Spring can pass it back on subsequent turns.
  - If no relevant course materials are found, returns HTTP `200 OK` with refusal text in `answer` and an empty `sources: []` list.

---

### F. Ingestion & Indexing (Internal)

#### `POST /internal/index-pdf`
Trigger asynchronous background PDF parsing, OCR, chunking, and embedding.

- **Headers**: `X-Internal-Secret: <secret>`
- **Request Body**:
```json
{
  "item_id": 201,
  "course_id": 101,
  "file_path": "data/pdfs/Algorithms_101.pdf"
}
```

- **Response `202 Accepted`**:
```json
{
  "job_id": 55,
  "status": "QUEUED"
}
```

