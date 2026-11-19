from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str = ""
    internal_secret: str = ""
    # Empty = standalone mode: student questions use sample/student_data.json instead of Spring.
    spring_base_url: str = ""
    student_data_timeout_seconds: float = 10.0
    # "gemini" or "ollama": which one writes the answers. Embeddings always use Gemini.
    llm_provider: str = "gemini"
    # true = answers run fully on Gemini, false = fully on Claude Code, auto = use LLM_CHAIN below.
    use_gemini: str = "auto"
    # Ordered list of answer models, tried left to right. Entries: gemini:<model>, claude-code, ollama.
    # Example: gemini:gemini-3.8-flash,claude-code,gemini:gemini-3.5-flash-lite,ollama
    # Empty = the older LLM_PROVIDER / GEMINI_FALLBACK_MODELS / LLM_FALLBACK_PROVIDER settings below.
    llm_chain: str = ""
    # Attempts for a chain entry that has another one behind it: 1 = switch to the next at once.
    llm_fast_switch_attempts: int = 1
    # A used-up daily quota is skipped for this long (seconds) before the entry is tried again.
    llm_daily_quota_cooldown_seconds: float = 600.0
    # Claude Code (the `claude` command, using your own Claude login) as an answer model.
    claude_code_command: str = "claude"
    claude_code_model: str = "haiku"
    claude_code_extra_args: str = "--restricted --effort low"
    claude_code_timeout_seconds: float = 90.0
    # "ollama" = if Gemini and all its fallback models fail, answer with the local Ollama model.
    llm_fallback_provider: str = ""
    # After Gemini fails, skip it for this long so every question doesn't pay for a failed attempt.
    llm_fallback_cooldown_seconds: float = 60.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    # CPU-only machines can need a minute or more for one answer.
    ollama_timeout_seconds: float = 180.0
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Changing the embedding model changes the vector size, so re-ingest (python -m app.cli reset) after.
    gemini_embedding_model: str = "gemini-embedding-001"
    # Comma-separated, tried in order when gemini_model stays busy, over quota, or is not found.
    gemini_fallback_models: str = ""
    gemini_temperature: float = 0.2
    gemini_max_retries: int = 5
    # First wait before a retry; doubles each time (3s, 6s, 12s, 24s), capped at gemini_backoff_max_seconds.
    gemini_backoff_seconds: float = 3.0
    gemini_backoff_max_seconds: float = 30.0
    # Without a timeout a call to an overloaded Gemini can hang forever and the retry loop never advances.
    gemini_timeout_seconds: float = 30.0
    chroma_dir: str = "./data/chroma"
    sqlite_path: str = "./data/service.sqlite"
    # Job queue database. SQLite by default; use mysql+pymysql://... in production.
    database_url: str = "sqlite:///./data/jobs.sqlite"
    pdf_dir: str = "./data/pdfs"
    extracted_dir: str = "./data/extracted"
    chunks_dir: str = "./data/chunks"
    max_pdf_mb: int = 50
    # OCR for scanned pages. Leave tesseract_cmd empty if tesseract is on PATH.
    tesseract_cmd: str = ""
    ocr_language: str = "eng"
    ocr_max_pages: int = 200
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
