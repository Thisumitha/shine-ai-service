from unittest.mock import MagicMock

import pytest
from google.genai import errors

from app.infrastructure.llm.fallback_client import FallbackLLMClient


def busy():
    return errors.APIError(code=503, response_json={"message": "high demand"})


def make(primary_effect):
    primary, fallback = MagicMock(), MagicMock()
    primary.generate.side_effect = primary_effect
    fallback.generate.return_value = "from ollama"
    return FallbackLLMClient(primary, fallback, cooldown_seconds=60), primary, fallback


def test_uses_primary_when_it_works():
    client, primary, fallback = make(["from gemini"])
    assert client.generate("Hi") == "from gemini"
    fallback.generate.assert_not_called()


def test_uses_fallback_when_primary_is_unavailable():
    client, _, fallback = make(busy())
    assert client.generate("Hi", system_instruction="s") == "from ollama"
    fallback.generate.assert_called_once_with("Hi", system_instruction="s")


def test_skips_primary_during_cooldown():
    client, primary, fallback = make(busy())
    client.generate("one")
    client.generate("two")
    assert primary.generate.call_count == 1
    assert fallback.generate.call_count == 2


def test_json_calls_fall_back_too():
    primary, fallback = MagicMock(), MagicMock()
    primary.generate_json.side_effect = busy()
    fallback.generate_json.return_value = "{}"
    client = FallbackLLMClient(primary, fallback)
    assert client.generate_json("Classify", schema_dict={"a": 1}) == "{}"


def test_bad_request_does_not_fall_back():
    client, _, fallback = make(errors.APIError(code=400, response_json={"message": "bad"}))
    with pytest.raises(errors.APIError):
        client.generate("Hi")
    fallback.generate.assert_not_called()
