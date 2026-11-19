import json
import logging
import re
from typing import Any

import httpx

from app.config import Settings
from app.domain.ports.llm_client import LLMClient
from app.infrastructure.llm.error_messages import LLMUnavailableError

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Local LLM through Ollama (https://ollama.com): free, no quota, nothing leaves this machine.

    Only answer generation runs here; embeddings stay on whatever GEMINI_EMBEDDING_MODEL says.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.ollama_model
        self.temperature = settings.gemini_temperature
        self._http = client or httpx.Client(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.ollama_timeout_seconds,
        )

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._chat(prompt, system_instruction)

    def generate_json(
        self,
        prompt: str,
        schema_dict: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        system = system_instruction or ""
        if schema_dict:
            # Small models follow a schema far better when it is spelled out, not only enforced.
            system += f"\nReply with JSON only, matching this schema:\n{json.dumps(schema_dict)}"
        text = self._strip_code_fences(self._chat(prompt, system or None, json_mode=True))
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Ollama JSON mode returned non-JSON text: %s", text)
            raise ValueError(f"Failed to decode model output as JSON: {text}") from exc
        return text

    def _chat(self, prompt: str, system_instruction: str | None, json_mode: bool = False) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        try:
            response = self._http.post("/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                "Cannot reach Ollama. Start it (open the Ollama app or run `ollama serve`) "
                f"and pull the model: ollama pull {self.model}"
            ) from exc
        if response.status_code == 404:
            raise LLMUnavailableError(f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}")
        response.raise_for_status()
        return (response.json().get("message", {}).get("content") or "").strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text.strip(), flags=re.IGNORECASE)
        return match.group(1).strip() if match else text
