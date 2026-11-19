from app.config import Settings
from app.domain.ports.llm_client import LLMClient
from app.infrastructure.llm.chain_client import ChainLLMClient
from app.infrastructure.llm.gemini_client import GeminiClient


def build_llm_client(settings: Settings) -> LLMClient:
    """The model that writes answers.

    USE_GEMINI=true runs everything on Gemini, false runs everything on Claude Code. With auto (the default)
    LLM_CHAIN decides, or the older provider settings when it is empty.
    """
    mode = settings.use_gemini.strip().lower()
    if mode in ("true", "1", "yes", "on"):
        models = [settings.gemini_model, *_split(settings.gemini_fallback_models)]
        return _build_chain(settings, [f"gemini:{model}" for model in models])
    if mode in ("false", "0", "no", "off"):
        return _build_chain(settings, ["claude-code"])
    if mode not in ("", "auto"):
        raise ValueError(f"USE_GEMINI must be true, false or auto (got '{settings.use_gemini}').")

    names = _split(settings.llm_chain)
    if names:
        return _build_chain(settings, names)
    return _build_legacy(settings)


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_chain(settings: Settings, names: list[str]) -> LLMClient:
    entries: list[tuple[str, LLMClient]] = []
    for index, item in enumerate(names):
        kind, _, argument = item.partition(":")
        kind = kind.lower()
        is_last = index == len(names) - 1
        if kind == "gemini":
            model = argument or settings.gemini_model
            # Not the last choice: give up after the configured number of attempts so the switch is quick.
            attempts = settings.gemini_max_retries if is_last else settings.llm_fast_switch_attempts
            single = settings.model_copy(
                update={"gemini_model": model, "gemini_fallback_models": "", "gemini_max_retries": attempts}
            )
            entries.append((model, GeminiClient(settings=single, log_final_failure=is_last)))
        elif kind == "claude-code":
            from app.infrastructure.llm.claude_code_client import ClaudeCodeClient

            entries.append(("claude-code", ClaudeCodeClient(settings=settings)))
        elif kind == "ollama":
            from app.infrastructure.llm.ollama_client import OllamaClient

            entries.append(("ollama", OllamaClient(settings=settings)))
        else:
            raise ValueError(f"Unknown LLM_CHAIN entry '{item}'. Use gemini:<model>, claude-code or ollama.")
    return ChainLLMClient(
        entries,
        cooldown_seconds=settings.llm_fallback_cooldown_seconds,
        daily_quota_cooldown_seconds=settings.llm_daily_quota_cooldown_seconds,
    )


def _build_legacy(settings: Settings) -> LLMClient:
    if settings.llm_provider.lower() == "ollama":
        from app.infrastructure.llm.ollama_client import OllamaClient

        return OllamaClient(settings=settings)
    client: LLMClient = GeminiClient(settings=settings)
    if settings.llm_fallback_provider.lower() == "ollama":
        from app.infrastructure.llm.fallback_client import FallbackLLMClient
        from app.infrastructure.llm.ollama_client import OllamaClient

        client = FallbackLLMClient(
            primary=client,
            fallback=OllamaClient(settings=settings),
            cooldown_seconds=settings.llm_fallback_cooldown_seconds,
        )
    return client
