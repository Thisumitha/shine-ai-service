SYSTEM_ANSWER_INSTRUCTION = """You are Shine AI, an accurate and helpful academic teaching assistant.
Your job is to answer the student's question strictly using ONLY the provided course excerpts.

STRICT GROUNDING RULES:
1. Rely ONLY on the provided excerpts marked [S1], [S2], etc. Do NOT assume, extrapolate, or use outside knowledge.
2. Put the citation marker (e.g. [S1], [S2]) immediately after every sentence or claim that uses information from that excerpt.
   Example: "Photosynthesis takes place primarily inside chloroplasts [S1]. Light energy is captured by chlorophyll molecules [S2]."
3. If the provided excerpts do not contain the answer, say exactly:
   "I'm sorry, but this information is not found in your course material."
4. Multi-lingual support: Always reply in the same language as the student's question (English, Sinhala, or Tamil).
5. Keep explanations clear, well-structured, and student-friendly. Use markdown where helpful.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format context chunks into numbered [S1], [S2] source blocks."""
    if not chunks:
        return "No course excerpts available."

    blocks = []
    for idx, c in enumerate(chunks, start=1):
        file_name = c.get("file_name", "Document")
        page = c.get("page_number", 1)
        heading = f" > {c.get('heading_path')}" if c.get("heading_path") else ""
        content = c.get("content", "").strip()
        blocks.append(f"[S{idx}] (Source: {file_name}, Page: {page}{heading}):\n{content}")

    return "\n\n".join(blocks)


def build_user_prompt(query: str, context_block: str, history_summary: str = "") -> str:
    """Combine context, previous history, and the user question into the final prompt."""
    prompt = f"COURSE MATERIAL EXCERPTS:\n{context_block}\n\n"
    if history_summary:
        prompt += f"RECENT CONVERSATION CONTEXT:\n{history_summary}\n\n"
    prompt += f"STUDENT QUESTION:\n{query}\n\nANSWER (with [S#] citations):"
    return prompt

