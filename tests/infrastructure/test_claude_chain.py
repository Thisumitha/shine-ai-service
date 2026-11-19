import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from app.config import Settings
from app.infrastructure.llm import claude_code_client
from app.infrastructure.llm.chain_client import ChainLLMClient
from app.infrastructure.llm.claude_code_client import ClaudeCodeClient
from app.infrastructure.llm.error_messages import LLMUnavailableError
from app.infrastructure.llm.factory import build_llm_client


def reply(result="ok", code=0, is_error=False):
    out = json.dumps({"result": result, "is_error": is_error})
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr="")


class FakeClaude:
    def __init__(self):
        self.calls = []
        self.response = reply("hello")


@pytest.fixture
def fake_claude(monkeypatch):
    fake = FakeClaude()
    monkeypatch.setattr(claude_code_client.shutil, "which", lambda name: f"/bin/{name}")

    def run(args, **kwargs):
        fake.calls.append((args, kwargs))
        return fake.response

    monkeypatch.setattr(claude_code_client.subprocess, "run", run)
    return fake


def busy(code=503, message="high demand"):
    return errors.APIError(code=code, response_json={"message": message})


def test_claude_client_runs_with_tools_off_and_returns_the_text(fake_claude, monkeypatch):
    seen = {}

    def run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        # the system prompt must be readable from its file while the command runs
        seen["system"] = open(args[args.index("--system-prompt-file") + 1], encoding="utf-8").read()
        return fake_claude.response

    monkeypatch.setattr(claude_code_client.subprocess, "run", run)
    text = ClaudeCodeClient(Settings()).generate("What is X?", system_instruction="Line one\nLine two")

    args = seen["args"]
    assert text == "hello"
    assert args[args.index("--tools") + 1] == ""  # no tools at all
    assert "--restricted" in args and "--no-session-persistence" in args
    assert "--system-prompt" not in args  # never as an argument: the Windows wrapper cuts it at a line break
    assert seen["system"] == "Line one\nLine two"  # multi-line prompt arrives whole
    assert seen["kwargs"]["input"] == "What is X?"  # the question goes on stdin, not the command line


def test_claude_client_removes_its_temporary_prompt_file(fake_claude, monkeypatch):
    used = []

    def run(args, **kwargs):
        used.append(Path(args[args.index("--system-prompt-file") + 1]))
        assert used[0].exists()  # present while the command runs
        return fake_claude.response

    monkeypatch.setattr(claude_code_client.subprocess, "run", run)
    ClaudeCodeClient(Settings()).generate("Q", system_instruction="Be brief")

    assert used and not used[0].exists()  # only this call's own file is checked: a real run may share the folder


def test_claude_client_json_mode_ignores_fences_and_text_around_the_json(fake_claude):
    client = ClaudeCodeClient(Settings())
    fake_claude.response = reply('```json\n{"intent": "COURSE_CONTENT"}\n```\n\n**Summary:** a definition question.')
    assert client.generate_json("Classify") == '{"intent": "COURSE_CONTENT"}'

    fake_claude.response = reply('Sure! {"calls": [{"tool": "get_attendance"}]} Hope that helps.')
    assert client.generate_json("Pick") == '{"calls": [{"tool": "get_attendance"}]}'


def test_claude_client_json_mode_rejects_a_reply_without_json(fake_claude):
    fake_claude.response = reply("I cannot do that.")
    with pytest.raises(ValueError, match="Failed to decode"):
        ClaudeCodeClient(Settings()).generate_json("Classify")


def test_claude_client_reports_problems_in_plain_words(monkeypatch, fake_claude):
    client = ClaudeCodeClient(Settings())

    monkeypatch.setattr(claude_code_client.shutil, "which", lambda name: None)
    with pytest.raises(LLMUnavailableError, match="not installed"):
        client.generate("Q")

    monkeypatch.setattr(claude_code_client.shutil, "which", lambda name: "/bin/claude")
    fake_claude_response = reply("Not logged in", code=1, is_error=True)
    monkeypatch.setattr(claude_code_client.subprocess, "run", lambda *a, **k: fake_claude_response)
    with pytest.raises(LLMUnavailableError, match="could not answer"):
        client.generate("Q")

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("claude", 1)

    monkeypatch.setattr(claude_code_client.subprocess, "run", slow)
    with pytest.raises(LLMUnavailableError, match="too long"):
        client.generate("Q")


def make_chain(*clients, **kw):
    return ChainLLMClient([(f"m{i}", c) for i, c in enumerate(clients)], **kw)


def llm(result=None, error=None):
    c = MagicMock()
    c.generate.side_effect = error if error else None
    if not error:
        c.generate.return_value = result
    return c


def test_chain_switches_at_once_to_the_next_model(caplog):
    first, second = llm(error=busy()), llm("from claude")
    with caplog.at_level(logging.WARNING):
        assert make_chain(first, second).generate("Q") == "from claude"
    assert first.generate.call_count == 1
    assert any("Switching to m1" in r.getMessage() and "busy" in r.getMessage() for r in caplog.records)
    assert not any("{" in r.getMessage() for r in caplog.records)  # no raw API error text


def test_chain_skips_a_model_that_just_failed():
    first, second = llm(error=busy()), llm("answer")
    chain = make_chain(first, second)
    chain.generate("one")
    chain.generate("two")
    assert first.generate.call_count == 1 and second.generate.call_count == 2


def test_chain_keeps_a_used_up_daily_quota_out_for_longer(monkeypatch):
    daily = busy(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    first, second = llm(error=daily), llm("answer")
    chain = make_chain(first, second, cooldown_seconds=1, daily_quota_cooldown_seconds=1000)
    now = [100.0]
    monkeypatch.setattr("app.infrastructure.llm.chain_client.time.monotonic", lambda: now[0])
    chain.generate("one")
    now[0] += 500  # past the normal cool-down, still inside the daily one
    chain.generate("two")
    assert first.generate.call_count == 1


def test_chain_goes_on_to_the_third_model_and_raises_if_the_last_fails():
    chain = make_chain(llm(error=busy()), llm(error=LLMUnavailableError("Claude Code took too long")), llm("third"))
    assert chain.generate("Q") == "third"

    failing = make_chain(llm(error=busy()), llm(error=LLMUnavailableError("Cannot reach Ollama")))
    with pytest.raises(LLMUnavailableError):
        failing.generate("Q")


def test_chain_does_not_switch_on_a_bad_request_or_a_bug():
    first, second = llm(error=busy(400, "bad")), llm("never used")
    with pytest.raises(errors.APIError):
        make_chain(first, second).generate("Q")
    second.generate.assert_not_called()

    with pytest.raises(KeyError):
        make_chain(llm(error=KeyError("bug")), llm("x")).generate("Q")


def test_factory_builds_the_chain_in_order_with_fast_switching(monkeypatch):
    settings = Settings(
        llm_chain="gemini:gemini-3.8-flash, claude-code, gemini:gemini-3.5-flash-lite, ollama",
        gemini_max_retries=5,
        llm_fast_switch_attempts=1,
    )
    chain = build_llm_client(settings)

    assert [name for name, _ in chain.entries] == ["gemini-3.8-flash", "claude-code", "gemini-3.5-flash-lite", "ollama"]
    first, _, lite, _ = (c for _, c in chain.entries)
    assert (first.model, first.max_retries) == ("gemini-3.8-flash", 1)  # gives up at once
    assert (lite.model, lite.max_retries) == ("gemini-3.5-flash-lite", 1)  # ollama is still behind it

    last_gemini = build_llm_client(settings.model_copy(update={"llm_chain": "claude-code,gemini:gemini-3.5-flash-lite"}))
    assert last_gemini.entries[-1][1].max_retries == 5  # the last model keeps its full retries


def test_factory_rejects_an_unknown_entry_and_keeps_the_old_settings_working():
    with pytest.raises(ValueError, match="Unknown LLM_CHAIN"):
        build_llm_client(Settings(llm_chain="gemini:a,banana"))
    assert type(build_llm_client(Settings(llm_chain="", llm_fallback_provider=""))).__name__ == "GeminiClient"
    assert type(build_llm_client(Settings(llm_chain="", llm_fallback_provider="ollama"))).__name__ == "FallbackLLMClient"


def names(client):
    return [name for name, _ in client.entries]


@pytest.mark.parametrize("value", ["true", "TRUE", " True ", "1", "yes", "on"])
def test_use_gemini_true_runs_only_on_gemini_models_in_order(value):
    settings = Settings(
        use_gemini=value,
        gemini_model="gemini-3.8-flash",
        gemini_fallback_models="gemini-3.5-flash-lite",
        llm_chain="claude-code,ollama",  # ignored: the switch wins
    )
    chain = build_llm_client(settings)
    assert names(chain) == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


@pytest.mark.parametrize("value", ["false", "FALSE", " False ", "0", "no", "off"])
def test_use_gemini_false_runs_only_on_claude_code(value):
    chain = build_llm_client(Settings(use_gemini=value, llm_chain="gemini:gemini-3.8-flash,ollama"))
    assert names(chain) == ["claude-code"]


@pytest.mark.parametrize("value", ["auto", "", "AUTO"])
def test_use_gemini_auto_keeps_using_the_chain(value):
    chain = build_llm_client(Settings(use_gemini=value, llm_chain="gemini:gemini-3.8-flash,claude-code"))
    assert names(chain) == ["gemini-3.8-flash", "claude-code"]


def test_use_gemini_rejects_a_value_it_does_not_know():
    with pytest.raises(ValueError, match="USE_GEMINI must be true, false or auto"):
        build_llm_client(Settings(use_gemini="maybe"))


def test_use_gemini_true_with_a_single_model_keeps_its_full_retries():
    chain = build_llm_client(Settings(use_gemini="true", gemini_model="gemini-3.8-flash",
                                      gemini_fallback_models="", gemini_max_retries=5))
    assert chain.entries[0][1].max_retries == 5
