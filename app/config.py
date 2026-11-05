from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str = ""
    internal_secret: str = ""
    spring_base_url: str = "http://localhost:8080"
    gemini_api_key: str = ""
    chroma_dir: str = "./data/chroma"
    sqlite_path: str = "./data/service.sqlite"
    # Job queue database (MySQL). Create it once: CREATE DATABASE shine_ai CHARACTER SET utf8mb4;
    database_url: str = "mysql+pymysql://root:@localhost:3306/shine_ai"
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
