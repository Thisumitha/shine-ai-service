import json

import httpx
import pytest

from app.config import Settings
from app.infrastructure.llm.ollama_client import OllamaClient


def make_client(handler):
    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaClient(Settings(ollama_model="tiny"), client=http)


def reply(text):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


def test_generate_sends_system_and_user_messages():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply("  hello  ")

    assert make_client(handler).generate("Hi", system_instruction="Be brief") == "hello"
    assert seen["model"] == "tiny" and seen["stream"] is False
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_generate_json_uses_json_mode_and_strips_fences():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply('```json\n{"intent": "COURSE_CONTENT"}\n```')

    out = make_client(handler).generate_json("Classify", schema_dict={"type": "object"})
    assert out == '{"intent": "COURSE_CONTENT"}'
    assert seen["format"] == "json"
    assert '"type": "object"' in seen["messages"][0]["content"]


def test_generate_json_rejects_non_json():
    with pytest.raises(ValueError, match="Failed to decode"):
        make_client(lambda r: reply("not json")).generate_json("Classify")


def test_missing_model_tells_user_to_pull_it():
    client = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(RuntimeError, match="ollama pull tiny"):
        client.generate("Hi")


def test_ollama_not_running_gives_clear_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        make_client(handler).generate("Hi")
