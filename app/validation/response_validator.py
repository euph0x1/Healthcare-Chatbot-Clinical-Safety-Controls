"""
Stage 4 of the pipeline: Response Validation.

Checks whether the generated answer is actually entailed by the retrieved
MedQuAD context, catching cases where the model drifted into its own
parametric knowledge despite the grounded-generation instructions. This
is your hallucination guard -- log `is_supported` across your held-out
test set to compute the hallucination rate metric for your report.
"""

import json

from app.llm_client import generate
from app.models.schemas import RetrievedChunk, ValidationResult

VALIDATION_SYSTEM_PROMPT = """You are a strict fact-checker. Given a source
passage and a generated answer, judge ONLY whether every claim in the answer
is directly supported by the source. Respond with ONLY a JSON object, no
other text, in this exact form:
{"is_supported": true or false, "confidence": 0.0-1.0, "rationale": "one short sentence"}"""


def validate(answer: str, chunks: list[RetrievedChunk]) -> ValidationResult:
    if not chunks:
        return ValidationResult(
            is_supported=False, confidence=1.0, rationale="No source material was retrieved."
        )

    source_text = "\n\n".join(f"Q: {c.question}\nA: {c.answer}" for c in chunks)
    user_content = f"Source:\n{source_text}\n\nGenerated answer:\n{answer}"

    raw = generate(system=VALIDATION_SYSTEM_PROMPT, user_content=user_content, max_tokens=200)

    # Models sometimes wrap JSON in markdown fences despite instructions --
    # strip those defensively before parsing.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return ValidationResult(
            is_supported=bool(parsed.get("is_supported", False)),
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=str(parsed.get("rationale", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fail closed: if the judge's output can't be parsed, treat the
        # answer as unsupported rather than silently trusting it.
        return ValidationResult(
            is_supported=False,
            confidence=0.0,
            rationale=f"Validator response could not be parsed: {raw[:100]}",
        )
