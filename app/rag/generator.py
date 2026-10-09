"""
Stage 3 (generation half) of the pipeline: generate an answer grounded
*only* in the retrieved MedQuAD context. The system prompt explicitly
forbids answering from the model's own parametric knowledge, which is
what Stage 4 (response validation) then checks.
"""

from app.llm_client import generate
from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = """You are a hospital's general health information assistant.
Answer ONLY using the provided reference material below. If the reference
material does not contain enough information to answer, say so plainly and
suggest the user speak with a healthcare professional -- do not use outside
knowledge and do not guess.

Never provide a diagnosis, a dosage, or personalized medical advice. Keep
answers factual, concise, and in plain language."""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i}] Q: {c.question}\nA: {c.answer}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            "I don't have reliable information on that in my current "
            "knowledge base. Please consult a healthcare professional."
        )

    context = _format_context(chunks)
    user_content = f"Reference material:\n{context}\n\nPatient question: {query}"

    return generate(system=SYSTEM_PROMPT, user_content=user_content, max_tokens=500)
