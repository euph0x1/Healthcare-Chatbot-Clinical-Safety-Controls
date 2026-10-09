"""
Orchestrates the full flow shown in the architecture diagram:

  Patient Query
    -> Intent Classifier
    -> (Structured Lookup | RAG Retrieval + Generation + Validation)
    -> Escalation Policy Check
    -> Direct Answer / Clarifying Question / Escalate to Human
"""

from app.escalation import policy as escalation_policy
from app.hospital import lookup as hospital_lookup
from app.intent.classifier import classify
from app.models.schemas import (
    ChatResponse,
    EscalationDecision,
    IntentLabel,
    RetrievedChunk,
)
from app.rag import generator as rag_generator
from app.rag import retriever as rag_retriever
from app.validation import response_validator


CLARIFY_MESSAGE = (
    "I want to make sure I understand correctly -- could you tell me a "
    "little more about what you're asking?"
)

ESCALATE_MESSAGE = (
    "I'm not confident I can safely answer this myself, so I'm connecting "
    "you with a member of our healthcare staff who can help further."
)


def run(query: str, session_id: str = "default") -> ChatResponse:
    intent = classify(query)

    sources: list[RetrievedChunk] = []

    if intent.label == IntentLabel.APPOINTMENT_SERVICE:
        # Structured lookup path -- deterministic, so treat confidence as high
        # unless the intent classifier itself was unsure.
        answer_draft = hospital_lookup.handle_query(query)
        measurement_confidence = intent.confidence
        evidence_confidence = 1.0  # no generation, nothing to hallucinate

    elif intent.label == IntentLabel.HEALTH_INFO:
        rag_result = rag_retriever.retrieve(query)
        sources = rag_result.chunks
        answer_draft = rag_generator.generate_answer(query, rag_result.chunks)
        validation = response_validator.validate(answer_draft, rag_result.chunks)
        measurement_confidence = rag_result.top_similarity
        evidence_confidence = validation.confidence if validation.is_supported else 0.0

    else:  # UNCLEAR_RISKY
        answer_draft = ""
        measurement_confidence = intent.confidence
        evidence_confidence = 0.0

    escalation = escalation_policy.evaluate(
        query=query,
        session_id=session_id,
        measurement_confidence=measurement_confidence,
        evidence_confidence=evidence_confidence,
    )

    if escalation.decision == EscalationDecision.ESCALATE:
        final_answer = ESCALATE_MESSAGE
    elif escalation.decision == EscalationDecision.CLARIFY:
        final_answer = CLARIFY_MESSAGE
    else:
        final_answer = answer_draft

    return ChatResponse(
        intent=intent,
        escalation=escalation,
        answer=final_answer,
        sources=sources,
    )
