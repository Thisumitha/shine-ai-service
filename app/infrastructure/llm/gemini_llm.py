from google import genai

from app.config import Settings
from app.domain.ports.llm_client import LLMClient


class GeminiLLM(LLMClient):

    def __init__(self, settings: Settings):

        if not settings.gemini_api_key:
            raise ValueError("Gemini API key is missing.")

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model = settings.gemini_model

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text