import re

from app.domain.entities.chunk import RetrievedChunk
from app.domain.entities.citation import Citation
from app.domain.entities.message import Message
from app.domain.ports.llm_client import LLMClient
from app.services.prompts.answer_prompts import (
    SYSTEM_ANSWER_INSTRUCTION,
    build_context_block,
    build_user_prompt,
)

REFUSAL_MESSAGE = "I'm sorry, but this information is not found in your course material."


class AnswerService:
    """Generates strictly grounded answers from retrieved course chunks with [S1] citations."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[Message] | None = None,
    ) -> tuple[str, list[Citation]]:
        """Generate answer text and extract matched Citation objects."""
        if not chunks:
            return REFUSAL_MESSAGE, []

        chunk_dicts = [
            {
                "file_name": c.file_name,
                "page_number": c.page_number,
                "heading_path": c.heading_path,
                "content": c.content,
            }
            for c in chunks
        ]

        context_block = build_context_block(chunk_dicts)
        history_summary = self._summarize_history(history or [])
        prompt = build_user_prompt(
            query=query,
            context_block=context_block,
            history_summary=history_summary,
        )

        raw_answer = self.llm.generate(
            prompt=prompt,
            system_instruction=SYSTEM_ANSWER_INSTRUCTION,
        ).strip()

        # Check if the model explicitly refused
        if REFUSAL_MESSAGE.lower() in raw_answer.lower():
            return REFUSAL_MESSAGE, []

        citations = self._extract_citations(raw_answer, chunks)
        return raw_answer, citations

    def _extract_citations(
        self,
        text: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """Extract [S#] markers and match them with the source chunks."""
        matches = re.findall(r"\[S(\d+)\]", text)
        if not matches:
            return []

        # Maintain insertion order while removing duplicates
        seen_indices: list[int] = []
        for m in matches:
            idx = int(m)
            if idx not in seen_indices:
                seen_indices.append(idx)

        citations: list[Citation] = []
        for idx in seen_indices:
            # 1-based index to 0-based list
            list_index = idx - 1
            if 0 <= list_index < len(chunks):
                chunk = chunks[list_index]
                snippet = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
                citations.append(
                    Citation(
                        source_id=f"S{idx}",
                        file_name=chunk.file_name,
                        page_number=chunk.page_number,
                        snippet=snippet,
                    )
                )

        return citations

    @staticmethod
    def _summarize_history(history: list[Message], max_turns: int = 4) -> str:
        """Format the last few messages for conversational context."""
        recent = history[-max_turns:] if len(history) > max_turns else history
        lines = [f"{msg.role.capitalize()}: {msg.content}" for msg in recent if msg.content]
        return "\n".join(lines)

