"""Model names must come from settings (.env), never from constants buried in the code."""

from unittest.mock import MagicMock

from app.config import Settings
from app.infrastructure.llm.gemini_client import GeminiClient
from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder


def test_chat_model_comes_from_settings():
    settings = Settings(gemini_api_key="k", gemini_model="my-chat-model")
    assert GeminiClient(settings).model == "my-chat-model"


def test_embedding_model_comes_from_settings(monkeypatch):
    monkeypatch.setattr("app.infrastructure.retrieval.embedder_gemini.get_settings",
                        lambda: Settings(gemini_api_key="k", gemini_embedding_model="my-embed-model"))
    assert GeminiEmbedder().model == "my-embed-model"
    assert GeminiEmbedder(model="explicit").model == "explicit"  # an explicit argument still wins


def test_embedding_request_uses_the_configured_model():
    client = MagicMock()
    client.models.embed_content.return_value = MagicMock(embeddings=[MagicMock(values=[0.1, 0.2])])
    GeminiEmbedder(client=client, model="my-embed-model").embed(["hello"])
    assert client.models.embed_content.call_args.kwargs["model"] == "my-embed-model"


def test_container_passes_the_embedding_model_to_the_embedder(tmp_path, monkeypatch):
    from app.container import build_container

    settings = Settings(
        gemini_api_key="k", gemini_embedding_model="container-embed-model",
        database_url=f"sqlite:///{tmp_path / 'j.sqlite'}", sqlite_path=str(tmp_path / "s.sqlite"),
        chroma_dir=str(tmp_path / "chroma"), chunks_dir=str(tmp_path / "chunks"),
        pdf_dir=str(tmp_path / "pdfs"), extracted_dir=str(tmp_path / "ex"),
    )
    assert build_container(settings).embedder.model == "container-embed-model"


def test_env_example_lists_both_model_settings():
    text = open(".env.example", encoding="utf-8").read()
    assert "GEMINI_MODEL=" in text and "GEMINI_EMBEDDING_MODEL=" in text
