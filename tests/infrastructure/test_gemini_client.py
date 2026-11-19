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

    settings = Settings(gemini_api_key="fake-key", gemini_max_retries=2, gemini_fallback_models="")
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


def _fallback_client(side_effect):
    settings = Settings(
        gemini_api_key="fake-key",
        gemini_model="primary",
        gemini_fallback_models="second, third",
        gemini_max_retries=2,
    )
    sdk = MagicMock()
    sdk.models.generate_content.side_effect = side_effect
    return GeminiClient(settings=settings, client=sdk), sdk


def _models_called(sdk):
    return [c.kwargs["model"] for c in sdk.models.generate_content.call_args_list]


def test_falls_back_to_next_model_in_order(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    busy = errors.APIError(code=503, response_json={"message": "high demand"})
    client, sdk = _fallback_client([busy, busy, make_mock_response("from second")])

    assert client.generate("Hi") == "from second"
    assert _models_called(sdk) == ["primary", "primary", "second"]


def test_unknown_model_falls_back_without_retrying(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    missing = errors.APIError(code=404, response_json={"message": "model not found"})
    client, sdk = _fallback_client([missing, make_mock_response("from second")])

    assert client.generate("Hi") == "from second"
    assert _models_called(sdk) == ["primary", "second"]


def test_raises_after_every_model_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    busy = errors.APIError(code=503, response_json={"message": "high demand"})
    client, sdk = _fallback_client([busy] * 6)

    with pytest.raises(errors.APIError):
        client.generate("Hi")
    assert _models_called(sdk) == ["primary", "primary", "second", "second", "third", "third"]


def test_bad_request_does_not_fall_back():
    bad = errors.APIError(code=400, response_json={"message": "Invalid argument"})
    client, sdk = _fallback_client(bad)

    with pytest.raises(errors.APIError):
        client.generate("Hi")
    assert _models_called(sdk) == ["primary"]


def test_daily_quota_skips_retries_and_is_remembered(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    daily = errors.APIError(
        code=429, response_json={"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded"}
    )
    client, sdk = _fallback_client([daily, make_mock_response("one"), make_mock_response("two")])

    assert client.generate("Hi") == "one"
    assert client.generate("Hi again") == "two"
    assert _models_called(sdk) == ["primary", "second", "second"]
