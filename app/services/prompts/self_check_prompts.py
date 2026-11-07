SELF_CHECK_INSTRUCTION = """You are an academic claim verifier.
Your job is to check whether every sentence and claim in the CANDIDATE ANSWER is fully supported by the provided COURSE EXCERPTS.

Evaluate:
1. Is every statement in the answer factually entailed by the course excerpts?
2. Did the answer hallucinate or bring in outside facts not present in the excerpts?
3. Calculate a groundedness score from 0.0 (completely hallucinated/unsupported) to 1.0 (every single claim is strictly supported).

You must output a JSON object adhering to:
{
  "grounded": boolean,
  "score": float,
  "unsupported_claims": list of strings,
  "reason": string
}
"""


def build_self_check_prompt(answer: str, context_block: str) -> str:
    return (
        f"COURSE MATERIAL EXCERPTS:\n{context_block}\n\n"
        f"CANDIDATE ANSWER TO VERIFY:\n{answer}\n\n"
        "Return strictly valid JSON with keys: 'grounded' (bool), 'score' (float 0.0-1.0), "
        "'unsupported_claims' (list), and 'reason' (string)."
    )

