
from unittest.mock import Mock

import pytest
from google.genai.errors import APIError

from app.infrastructure.retrieval.embedder_gemini import GeminiEmbedder


def create_mock_client(embeddings=None):
    """Create a fake Gemini client for testing."""
    client = Mock()

    response = Mock()
    response.embeddings = embeddings or []

    client.models.embed_content.return_value = response

    return client


def make_embedding(values):
    """Create a fake embedding response."""
    embedding = Mock()
    embedding.values = values
    return embedding


def test_embed_returns_vectors():
    client = create_mock_client([
        make_embedding([0.1, 0.2]),
        make_embedding([0.3, 0.4]),
    ])

    embedder = GeminiEmbedder(client=client)

    result = embedder.embed(["Hello", "World"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    client.models.embed_content.assert_called_once()


def test_embed_empty_input():
    client = create_mock_client()
    embedder = GeminiEmbedder(client=client)

    result = embedder.embed([])

    assert result == []
    client.models.embed_content.assert_not_called()


def test_embed_processes_batches():
    client = create_mock_client([
        make_embedding([0.1, 0.2]),
        make_embedding([0.3, 0.4]),
    ])

    embedder = GeminiEmbedder(client=client, batch_size=2)

    result = embedder.embed(["A", "B", "C", "D"])

    assert len(result) == 4
    assert client.models.embed_content.call_count == 2


def test_embed_rejects_empty_text():
    client = create_mock_client()
    embedder = GeminiEmbedder(client=client)

    with pytest.raises(ValueError):
        embedder.embed(["Hello", ""])

    client.models.embed_content.assert_not_called()


def test_embed_rejects_invalid_batch_size():
    client = create_mock_client()

    with pytest.raises(ValueError):
        GeminiEmbedder(client=client, batch_size=0)


def test_embed_retries_after_api_error():
    client = create_mock_client([
        make_embedding([0.1, 0.2])
    ])

    client.models.embed_content.side_effect = [
        APIError(
            code=500,
            response_json={"error": {"message": "Temporary server error"}},
        ),
        Mock(embeddings=[make_embedding([0.1, 0.2])]),
    ]

    embedder = GeminiEmbedder(client=client)

    result = embedder.embed(["Hello"])

    assert result == [[0.1, 0.2]]
    assert client.models.embed_content.call_count == 2
