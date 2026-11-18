import logging

from app.domain.ports.llm_client import LLMClient
from app.services.prompts.reason_prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites a query that retrieved nothing useful, so the agent can search again with new wording."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def rewrite(self, query: str) -> str | None:
        """Return a reworded query, or None when the model gave nothing new."""
        try:
            raw = self.llm.generate(
                prompt=build_rewrite_prompt(query),
                system_instruction=REWRITE_SYSTEM_PROMPT,
            )
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return None

        if not isinstance(raw, str):
            return None
        rewritten = " ".join(raw.split()).strip(' "\'')
        if not rewritten or rewritten.lower() == query.strip().lower():
            return None
        return rewritten
