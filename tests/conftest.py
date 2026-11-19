"""Keep every test away from the real data and the real Gemini API.

Without this, tests that build the full app wrote into `data/chroma` (replacing the vectors of real PDFs
with test documents that reuse the same item ids) and called the real embedding API, which uses up the
daily quota and fails when it is gone.
"""
from types import SimpleNamespace

import pytest

from app.config import get_settings

_FOLDER_SETTINGS = {
    "CHROMA_DIR": "chroma",
    "PDF_DIR": "pdfs",
    "EXTRACTED_DIR": "extracted",
    "CHUNKS_DIR": "chunks",
}


class _FakeEmbeddingClient:
    """Stands in for google-genai: a small, repeatable vector per text, no network."""

    class models:
        @staticmethod
        def embed_content(model, contents):
            vectors = []
            for text in contents:
                seed = sum(ord(c) for c in text) or 1
                vectors.append(SimpleNamespace(values=[((seed * (i + 3)) % 97) / 97 + 0.01 for i in range(8)]))
            return SimpleNamespace(embeddings=vectors)


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    for name, folder in _FOLDER_SETTINGS.items():
        monkeypatch.setenv(name, str(tmp_path / "isolated" / folder))
    # the developer's own choice of answer model in .env must not change what the tests see
    for name, value in (("USE_GEMINI", "auto"), ("LLM_CHAIN", ""), ("LLM_PROVIDER", "gemini"),
                        ("LLM_FALLBACK_PROVIDER", ""), ("GEMINI_FALLBACK_MODELS", ""),
                        ("SPRING_BASE_URL", ""), ("INTERNAL_SECRET", "")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "isolated" / "service.sqlite"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'isolated' / 'jobs.sqlite'}")
    get_settings.cache_clear()

    from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder

    real_get_client = GeminiEmbedder._get_client

    def get_client(self):
        # Keep a client the test injected on purpose; replace only the real Gemini one.
        return self.client if self.client is not None else _FakeEmbeddingClient()

    monkeypatch.setattr(GeminiEmbedder, "_get_client", get_client)
    yield
    get_settings.cache_clear()
