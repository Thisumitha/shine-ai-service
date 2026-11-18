"""A brief DNS or connection failure must be retried, not kill an ingest or an evaluation run."""

import socket
from unittest.mock import MagicMock, Mock

import pytest

from app.config import Settings
from app.infrastructure.llm.gemini_client import GeminiClient
from app.infrastructure.network_errors import is_network_error
from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder


def httpx2_error(name: str = "ConnectError") -> Exception:
    """Looks like the SDK's httpx2.ConnectError: a different class from httpx's."""
    return type(name, (Exception,), {"__module__": "httpx2._exceptions"})("[Errno 11001] getaddrinfo failed")


def test_network_errors_are_recognised_whichever_library_raises_them():
    assert is_network_error(httpx2_error())
    assert is_network_error(socket.gaierror(11001, "getaddrinfo failed"))
    assert is_network_error(TimeoutError())
    assert not is_network_error(ValueError("bad input"))
    assert not is_network_error(KeyError("x"))


def test_embedder_retries_a_dns_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(GeminiEmbedder._embed_batch.retry, "sleep", lambda _: None)
    client = Mock()
    ok = Mock(embeddings=[Mock(values=[0.1, 0.2])])
    client.models.embed_content.side_effect = [httpx2_error(), httpx2_error(), ok]

    assert GeminiEmbedder(client=client).embed(["hello"]) == [[0.1, 0.2]]
    assert client.models.embed_content.call_count == 3


def test_embedder_gives_up_after_five_attempts(monkeypatch):
    monkeypatch.setattr(GeminiEmbedder._embed_batch.retry, "sleep", lambda _: None)
    client = Mock()
    client.models.embed_content.side_effect = httpx2_error()

    with pytest.raises(Exception, match="getaddrinfo"):
        GeminiEmbedder(client=client).embed(["hello"])
    assert client.models.embed_content.call_count == 5


def test_embedder_does_not_retry_a_programming_error(monkeypatch):
    monkeypatch.setattr(GeminiEmbedder._embed_batch.retry, "sleep", lambda _: None)
    client = Mock()
    client.models.embed_content.side_effect = TypeError("bug")

    with pytest.raises(TypeError):
        GeminiEmbedder(client=client).embed(["hello"])
    assert client.models.embed_content.call_count == 1


def test_chat_client_retries_a_dns_failure(monkeypatch):
    monkeypatch.setattr("app.infrastructure.llm.gemini_client.time.sleep", lambda _: None)
    sdk = MagicMock()
    response = MagicMock(text="ready", usage_metadata=None)
    sdk.models.generate_content.side_effect = [httpx2_error(), response]

    client = GeminiClient(Settings(gemini_api_key="k", gemini_max_retries=3), client=sdk)
    assert client.generate("hi") == "ready"
    assert sdk.models.generate_content.call_count == 2


def test_chat_client_does_not_retry_a_bug(monkeypatch):
    monkeypatch.setattr("app.infrastructure.llm.gemini_client.time.sleep", lambda _: None)
    sdk = MagicMock()
    sdk.models.generate_content.side_effect = KeyError("bug")

    client = GeminiClient(Settings(gemini_api_key="k", gemini_max_retries=3), client=sdk)
    with pytest.raises(KeyError):
        client.generate("hi")
    assert sdk.models.generate_content.call_count == 1
