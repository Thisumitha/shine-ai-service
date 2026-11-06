from unittest.mock import MagicMock
from google.genai import errors
import pytest

from app.config import Settings
from app.infrastructure.llm.gemini_client import GeminiClient


def make_mock_response(text: str, prompt_tokens: int = 15, candidates_tokens: int = 25):
    """Create a mock GenerateContentResponse with usage metadata."""
    mock_resp = MagicMock()
    mock_resp.text = text

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = candidates_tokens
    mock_usage.total_token_count = prompt_tokens + candidates_tokens
    mock_resp.usage_metadata = mock_usage

    return mock_resp


def test_gemini_client_generate_success():
    settings = Settings(gemini_api_key="fake-key", gemini_model="gemini-2.5-flash")
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = make_mock_response(
        text="Photosynthesis produces glucose and oxygen.",
        prompt_tokens=10,
        candidates_tokens=20,
    )

    client = GeminiClient(settings=settings, client=mock_genai_client)
    result = client.generate("How does photosynthesis work?")

    assert result == "Photosynthesis produces glucose and oxygen."
    metrics = client.get_usage_metrics()
    assert metrics["total_prompt_tokens"] == 10
    assert metrics["total_candidates_tokens"] == 20
    assert metrics["total_tokens"] == 30
    assert metrics["estimated_cost_usd"] == 0.0
    assert metrics["tier"] == "free"


def test_gemini_client_generate_json_strips_code_fences():
    settings = Settings(gemini_api_key="fake-key")
    mock_genai_client = MagicMock()
    # Simulate LLM wrapping JSON in markdown backticks
    mock_genai_client.models.generate_content.return_value = make_mock_response(
        text='```json\n{"intent": "COURSE_CONTENT", "confidence": 0.95}\n```'
    )

    client = GeminiClient(settings=settings, client=mock_genai_client)
    json_result = client.generate_json("Classify this query")

    assert json_result == '{"intent": "COURSE_CONTENT", "confidence": 0.95}'


def test_gemini_client_generate_json_raises_on_invalid_json():
    settings = Settings(gemini_api_key="fake-key")
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = make_mock_response(
        text="Not valid json at all!"
    )

    client = GeminiClient(settings=settings, client=mock_genai_client)
    with pytest.raises(ValueError, match="Failed to decode model output as JSON"):
        client.generate_json("Classify")


def test_gemini_client_retry_on_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)  # Don't delay tests

    settings = Settings(gemini_api_key="fake-key", gemini_max_retries=3)
    mock_genai_client = MagicMock()

    # Simulate 429 rate limit error on attempt 1, then success on attempt 2
    rate_limit_err = errors.APIError(code=429, response_json={"message": "Resource exhausted: Rate limit reached"})
    success_resp = make_mock_response(text="Recovered answer")

    mock_genai_client.models.generate_content.side_effect = [
        rate_limit_err,
        success_resp,
    ]

    client = GeminiClient(settings=settings, client=mock_genai_client)
    result = client.generate("Hello")

    assert result == "Recovered answer"
    assert mock_genai_client.models.generate_content.call_count == 2


def test_gemini_client_fails_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    settings = Settings(gemini_api_key="fake-key", gemini_max_retries=2)
    mock_genai_client = MagicMock()

    rate_limit_err = errors.APIError(code=429, response_json={"message": "Quota exceeded"})
    mock_genai_client.models.generate_content.side_effect = [
        rate_limit_err,
        rate_limit_err,
    ]

    client = GeminiClient(settings=settings, client=mock_genai_client)
    with pytest.raises(errors.APIError):
        client.generate("Hello")

    assert mock_genai_client.models.generate_content.call_count == 2


def test_gemini_client_non_transient_error_fails_immediately():
    settings = Settings(gemini_api_key="fake-key", gemini_max_retries=3)
    mock_genai_client = MagicMock()

    bad_request_err = errors.APIError(code=400, response_json={"message": "Bad Request: Invalid argument"})
    mock_genai_client.models.generate_content.side_effect = bad_request_err

    client = GeminiClient(settings=settings, client=mock_genai_client)
    with pytest.raises(errors.APIError):
        client.generate("Hello")

    assert mock_genai_client.models.generate_content.call_count == 1
