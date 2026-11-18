# Chat Agent & Grounded Generation

This document describes the 4-Stage AI Chat Agent, Answer Generator, Self-Check Entailment Verifier, and SQLite Memory store implemented in `app/services/agent/`, `app/services/`, and `app/infrastructure/` (Member 3).

---

## 1. Architecture: The 4-Stage Agent Pipeline

The Chat Agent follows an explicit 4-stage sequential loop: **Observe -> Reason -> Act -> Persist**.

```
User Message
     |
     v
[ 1. OBSERVE ]  ---> Reads recent conversation history (last N turns) from ChatStore
     |
     v
[ 2. REASON ]   ---> Classifies intent using LLM (course_question | small_talk | out_of_scope)
     |
     v
[ 3. ACT ]      ---> If course_question:
     |                 a. HybridRetriever retrieves top course chunks
     |                 b. AnswerService generates grounded answer with [S1] markers
     |                 c. SelfCheckService checks claim entailment (threshold 0.70)
     |               If small_talk / out_of_scope:
     |                 Direct polite response without retrieval
     v
[ 4. PERSIST ]  ---> Saves user message, assistant response, citations to ChatStore
     |
     v
Final Response
```

---

## 2. Core Stages & Services

### Stage 1: Observe (`ObserveStage`)
- **Location**: `app/services/agent/observe.py`
- Fetches the last $N$ messages (default 6 messages / 3 dialogue turns) from `ChatStore` for the active `session_id`.
- Formats past exchanges into conversation context so multi-turn follow-up questions ("explain that in simpler terms") have proper context.

### Stage 2: Reason (`ReasonStage`)
- **Location**: `app/services/agent/reason.py`
- **Prompt**: `app/services/prompts/reason_prompts.py`
- Calls `LLMClient.generate_json()` using structured JSON mode to categorize the student's query into:
  - `course_question`: Academic inquiry requiring slide retrieval.
  - `small_talk`: Greetings, thanks, polite conversational remarks.
  - `out_of_scope`: General knowledge questions unrelated to university courses (e.g., celebrity gossip, weather).

### Stage 3: Act (`ActStage`)
- **Location**: `app/services/agent/act.py`
- **Sub-service: AnswerService (`app/services/answer_service.py`)**:
  - Injects retrieved chunks labeled as `[S1]`, `[S2]`, etc.
  - Generates strictly grounded responses adhering to source materials.
  - Translates `[S1]` inline markers into explicit `Citation` objects containing `source_doc`, `page_number`, and quoted text.
  - Supports trilingual replies (English, Sinhala, Tamil) matching the student's query language.
- **Sub-service: SelfCheckService (`app/services/self_check_service.py`)**:
  - Performs sentence-level Natural Language Inference (NLI) claim entailment.
  - Scores whether every claim in the answer is backed by the retrieved context.
  - **Threshold**: Requires entailment score $\ge 0.70$.
  - **Refusal Fallback**: If score $< 0.70$ or no context is available, replaces the answer with:
    *"I couldn't find this information in your course materials."*
  - Logs unanswerable questions to `ChatStore.log_unanswered()` for lecturer review.

### Stage 4: Persist (`PersistStage`)
- **Location**: `app/services/agent/persist.py`
- Saves both the user's prompt and the assistant's grounded response to `ChatStore`.
- Stores extracted citations as serialized JSON linked to the assistant's message.

---

## 3. SQLite Chat Memory (`SqliteChatStore`)

- **Location**: `app/infrastructure/memory/sqlite_chat_store.py`
- **Port Implemented**: `app.domain.ports.chat_store.ChatStore`
- **Storage**: Lightweight SQLite database located at `./data/service.sqlite` (or `:memory:` for fast testing).
- **Schema Tables**:
  - `chat_sessions`: `session_id`, `user_id`, `course_id`, `created_at`, `updated_at`.
  - `chat_messages`: `id`, `session_id`, `sender` (`user`/`assistant`), `text`, `citations_json`, `created_at`.
  - `chat_feedback`: `id`, `message_id`, `score` (`+1`/`-1`), `comment`, `created_at`.
  - `unanswered_queries`: `id`, `query_text`, `course_id`, `reason`, `created_at`.

---

## 4. LLM Infrastructure (`GeminiClient`)

- **Location**: `app/infrastructure/llm/gemini_client.py`
- **Port Implemented**: `app.domain.ports.llm_client.LLMClient`
- **Model**: `gemini-3.8-flash` by default (set with GEMINI_MODEL; older names such as gemini-2.5-flash return 404 for new keys).
- **Resilience**: Exponential backoff retry loop with jitter handling rate limits (`HTTP 429`) and temporary outages (`HTTP 503`).
- **Structured Outputs**: Native JSON mode with automated markdown fence stripping (````json ... ````).
- **Cost & Token Tracking**: Tracks cumulative input, output, and cached tokens per session.

