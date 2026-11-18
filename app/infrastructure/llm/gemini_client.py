import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import errors, types
import httpx

from app.config import Settings
from app.domain.ports.llm_client import LLMClient
from app.infrastructure.network_errors import is_network_error

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Gemini LLM adapter supporting text generation, JSON mode, retry with backoff,

    and token telemetry for the free tier.
    """

    def __init__(
        self,
        settings: Settings,
        client: genai.Client | None = None,
    ):
        self.settings = settings
        self.model = settings.gemini_model
        self.temperature = settings.gemini_temperature
        self.max_retries = max(1, settings.gemini_max_retries)
        self._client = client

        # Token and cost tracking (Free Tier = $0.00 cost)
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

    @property
    def client(self) -> genai.Client:
        """Create the Gemini client only when a request is made."""
        if self._client is not None:
            return self._client
        if not self.settings.gemini_api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in .env.")
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate free-form text response with retries."""
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )
        response = self._execute_with_retry(prompt=prompt, config=config)
        return response.text or ""

    def generate_json(
        self,
        prompt: str,
        schema_dict: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        """Generate a guaranteed JSON string adhering to optional schema."""
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "system_instruction": system_instruction,
        }
        if schema_dict:
            config_kwargs["response_schema"] = schema_dict

        config = types.GenerateContentConfig(**config_kwargs)
        response = self._execute_with_retry(prompt=prompt, config=config)
        raw_text = (response.text or "").strip()

        # Sanitize if model accidentally wrapped output in markdown code fences
        cleaned_text = self._strip_markdown_code_fences(raw_text)

        # Validate that output is well-formed JSON
        try:
            json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.warning("Gemini JSON mode returned non-JSON text: %s. Error: %s", raw_text, exc)
            raise ValueError(f"Failed to decode model output as JSON: {raw_text}") from exc

        return cleaned_text

    def get_usage_metrics(self) -> dict[str, Any]:
        """Return cumulative token usage and cost metrics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_candidates_tokens": self.total_candidates_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "tier": "free",
        }

    def reset_usage_metrics(self) -> None:
        """Reset token usage counters."""
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

    def _execute_with_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        backoff_base_seconds: float = 1.0,
    ) -> Any:
        """Execute generate_content with exponential backoff on transient errors."""
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                self._record_usage(response)
                return response
            except Exception as err:
                if not (isinstance(err, errors.APIError) or is_network_error(err)):
                    raise  # a bug or bad input, not a Gemini or network failure
                last_exception = err
                is_transient = self._is_transient_error(err)
                if not is_transient or attempt >= self.max_retries:
                    logger.error(
                        "Gemini call failed permanently on attempt %d/%d: %s",
                        attempt,
                        self.max_retries,
                        err,
                    )
                    raise

                sleep_time = backoff_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini transient error on attempt %d/%d (%s). Retrying in %.1fs...",
                    attempt,
                    self.max_retries,
                    err,
                    sleep_time,
                )
                time.sleep(sleep_time)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected failure executing Gemini request.")

    def _record_usage(self, response: Any) -> None:
        """Extract token usage metadata from response."""
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + candidates_tokens)

        self.total_prompt_tokens += prompt_tokens
        self.total_candidates_tokens += candidates_tokens
        self.total_tokens += total_tokens
        # Free Tier: estimated cost is $0.00
        self.estimated_cost_usd = 0.0

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        """Determine if an exception is temporary/rate-limit and can be retried."""
        if isinstance(err, errors.APIError):
            # 429 = Rate limit/quota exceeded, 500 = Server error, 503 = Unavailable
            code = getattr(err, "code", None)
            if code in (429, 500, 502, 503, 504):
                return True
        return is_network_error(err)

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        """Remove ```json ... ``` code blocks if present."""
        match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

