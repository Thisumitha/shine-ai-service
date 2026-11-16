from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str = ""
    internal_secret: str = ""
    # Empty = standalone mode: student questions use sample/student_data.json instead of Spring.
    spring_base_url: str = ""
    student_data_timeout_seconds: float = 10.0
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_retries: int = 3
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
