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
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
