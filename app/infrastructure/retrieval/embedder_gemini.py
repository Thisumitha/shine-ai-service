
from google import genai
from google.genai.errors import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings
from app.domain.ports.embedder import Embedder


class GeminiEmbedder(Embedder):
    """Generate text embeddings using Google's Gemini API."""

    def __init__(
        self,
        client=None,
        model: str = "gemini-embedding-001",
        batch_size: int = 100,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model = model
        self.batch_size = batch_size

        if client is not None:
            self.client = client
        else:
            settings = get_settings()

            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")

            self.client = genai.Client(
                api_key=settings.gemini_api_key
            )

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
        retry=retry_if_exception_type(
            (APIError, TimeoutError, ConnectionError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one batch to Gemini with automatic retries."""

        response = self.client.models.embed_content(
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

