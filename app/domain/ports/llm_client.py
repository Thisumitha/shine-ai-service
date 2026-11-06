from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract port for large language model interactions."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generate free-form text from prompt and optional system instruction."""
        ...

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema_dict: dict | None = None,
        system_instruction: str | None = None,
    ) -> str:
        """Generate a validated JSON string adhering to the optional schema structure."""
        ...
