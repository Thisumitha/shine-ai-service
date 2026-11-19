import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from app.domain.ports.llm_client import LLMClient
from app.infrastructure.llm.error_messages import LLMUnavailableError

logger = logging.getLogger(__name__)


class ClaudeCodeClient(LLMClient):
    """Answer with Claude Code (the `claude` command), signed in with your own Claude account.

    Each call starts `claude -p` in an empty folder with every tool turned off, so it only writes text:
    it cannot read files or run commands, even if the text it is given contains instructions.
    The question and the retrieved course text are sent to Anthropic, and the call is billed to your account.
    """

    def __init__(self, settings: Settings):
        self.command = settings.claude_code_command
        self.model = settings.claude_code_model
        self.extra_args = shlex.split(settings.claude_code_extra_args)
        self.timeout = settings.claude_code_timeout_seconds
        # An empty folder, so no project CLAUDE.md or files are picked up.
        self._cwd = Path(tempfile.gettempdir()) / "shine_claude_code"
        self._cwd.mkdir(exist_ok=True)

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._ask(prompt, system_instruction)

    def generate_json(
        self,
        prompt: str,
        schema_dict: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        system = (system_instruction or "") + "\nReply with one JSON object only: no prose and no code fences."
        if schema_dict:
            system += f"\nIt must match this schema:\n{json.dumps(schema_dict)}"
        raw = self._ask(prompt, system.strip())
        text = self._first_json(raw)
        if text is None:
            logger.warning("Claude Code did not return JSON.")
            logger.debug("Claude Code reply: %s", raw)
            raise ValueError(f"Failed to decode model output as JSON: {raw[:200]}")
        return text

    def _ask(self, prompt: str, system_instruction: str | None) -> str:
        executable = shutil.which(self.command)
        if executable is None:
            raise LLMUnavailableError(
                f"Claude Code is not installed or not on PATH (looked for '{self.command}')"
            )

        args = [executable, "-p", "--tools", "", "--model", self.model, "--no-session-persistence",
                "--disable-slash-commands", "--output-format", "json", *self.extra_args]
        prompt_file = None
        if system_instruction:
            # A file, not an argument: the Windows `claude.cmd` wrapper cuts an argument at its first line break.
            prompt_file = self._cwd / f"system-{uuid.uuid4().hex}.txt"
            prompt_file.write_text(system_instruction, encoding="utf-8")
            args += ["--system-prompt-file", str(prompt_file)]

        try:
            done = subprocess.run(
                args, input=prompt, capture_output=True, text=True, encoding="utf-8",
                timeout=self.timeout, cwd=self._cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailableError("Claude Code took too long to reply") from exc
        except OSError as exc:
            raise LLMUnavailableError(f"Claude Code could not be started ({exc.strerror or exc})") from exc
        finally:
            if prompt_file is not None:
                prompt_file.unlink(missing_ok=True)

        try:
            reply = json.loads(done.stdout)
        except json.JSONDecodeError:
            reply = {}
        if done.returncode != 0 or reply.get("is_error") or "result" not in reply:
            detail = (reply.get("result") or done.stderr or done.stdout or "no output").strip()
            logger.debug("Claude Code failure detail: %s", detail)
            raise LLMUnavailableError(f"Claude Code could not answer ({detail[:120]})")
        return (reply["result"] or "").strip()

    @staticmethod
    def _first_json(text: str) -> str | None:
        """The first JSON object or array in the reply, even if it is wrapped in a code fence or followed by prose."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[{\[]", text):
            try:
                value, end = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return text[match.start():match.start() + end]
        return None
