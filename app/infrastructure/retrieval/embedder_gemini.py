
from google import genai
from google.genai.errors import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from app.config import get_settings
from app.domain.ports.embedder import Embedder
from app.infrastructure.network_errors import is_network_error


def _is_retryable(error: BaseException) -> bool:
    """Gemini API errors and any temporary network failure (including DNS lookups) are retried."""
    return isinstance(error, APIError) or is_network_error(error)


class GeminiEmbedder(Embedder):
    """Generate text embeddings using Google's Gemini API."""

    def __init__(
        self,
        client=None,
        model: str | None = None,
        batch_size: int = 100,
        api_key: str | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model = model or get_settings().gemini_embedding_model
        self.batch_size = batch_size

        # Keep an injected fake or real client if one was provided.
        self.client = client

        # Store the API key without creating the Gemini client yet.
        if api_key is None:
            api_key = get_settings().gemini_api_key

        self.api_key = api_key

    def _get_client(self):
        """Create the Gemini client only when an embedding is requested."""

        if self.client is not None:
            return self.client

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into embedding vectors in batches."""

        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=10,
        ),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one batch to Gemini with automatic retries."""

        client = self._get_client()

        response = client.models.embed_content(
            model=self.model,
            contents=texts,
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        if len(response.embeddings) != len(texts):
            raise RuntimeError(
                "Embedding count does not match input text count."
            )

        return [
            embedding.values
            for embedding in response.embeddings
        ]
