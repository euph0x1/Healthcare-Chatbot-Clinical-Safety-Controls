"""
Stage 5 of the pipeline: Escalation Policy Check.

Applies uniformly across both branches (structured lookup and RAG) and
decides whether to answer directly, ask a clarifying question, or escalate
to a human. This mini-project version keeps the five signals explicit and
simple; your full research system can replace the scoring with something
more sophisticated without changing the DIRECT_ANSWER / CLARIFY / ESCALATE
contract the rest of the app depends on.
"""

from collections import defaultdict

from app.models.schemas import EscalationDecision, EscalationResult, EscalationSignals

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "unconscious", "bleeding heavily", "severe pain", "emergency",
    "suicide", "self harm",
]

DOSAGE_KEYWORDS = [
    "mg of", "milligrams of", "dosage of", "how much should i take",
    "overdose", "how many pills",
]

DIAGNOSIS_KEYWORDS = [
    "do i have", "diagnose me", "is this cancer", "am i having a heart attack",
]

# Very small in-memory session store for the "persistence" signal.
# Replace with Redis/DB for anything beyond a mini-project demo.
_low_confidence_turns: dict[str, int] = defaultdict(int)

CONFIDENCE_ESCALATE_THRESHOLD = 0.35
CONFIDENCE_CLARIFY_THRESHOLD = 0.6
PERSISTENCE_ESCALATE_THRESHOLD = 2


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def record_turn_outcome(session_id: str, was_low_confidence: bool) -> int:
    """Call once per turn to update the persistence counter for a session."""
    if was_low_confidence:
        _low_confidence_turns[session_id] += 1
    else:
        _low_confidence_turns[session_id] = 0
    return _low_confidence_turns[session_id]


def evaluate(
    query: str,
    session_id: str,
    measurement_confidence: float,
    evidence_confidence: float,
) -> EscalationResult:
    """
    Combine risk keywords + confidence signals + persistence into a single
    routing decision. `measurement_confidence` should come from retrieval
    similarity or lookup-match certainty; `evidence_confidence` should come
    from the Stage 4 response validator (1.0 if fully supported).
    """
    emergency_hits = _keyword_hits(query, EMERGENCY_KEYWORDS)
    dosage_hits = _keyword_hits(query, DOSAGE_KEYWORDS)
    diagnosis_hits = _keyword_hits(query, DIAGNOSIS_KEYWORDS)
    risk_hits = emergency_hits + dosage_hits + diagnosis_hits

    decision_risk = 1.0 if risk_hits else 0.0
    combined_confidence = min(measurement_confidence, evidence_confidence)

    is_low_confidence = combined_confidence < CONFIDENCE_CLARIFY_THRESHOLD
    persistence = record_turn_outcome(session_id, is_low_confidence)

    signals = EscalationSignals(
        measurement_confidence=measurement_confidence,
        evidence_confidence=evidence_confidence,
        decision_risk=decision_risk,
        persistence=persistence,
        logging_completeness=True,
        risk_keywords_hit=risk_hits,
    )

    if emergency_hits:
        return EscalationResult(
            decision=EscalationDecision.ESCALATE,
            reason=f"Emergency-risk language detected: {emergency_hits}",
            signals=signals,
        )

    if dosage_hits or diagnosis_hits:
        return EscalationResult(
            decision=EscalationDecision.ESCALATE,
            reason=f"Dosage/diagnosis request detected: {risk_hits}",
            signals=signals,
        )

    if persistence >= PERSISTENCE_ESCALATE_THRESHOLD:
        return EscalationResult(
            decision=EscalationDecision.ESCALATE,
            reason=f"Repeated low-confidence turns in this session ({persistence}).",
            signals=signals,
        )

    if combined_confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        return EscalationResult(
            decision=EscalationDecision.ESCALATE,
            reason=f"Combined confidence too low ({combined_confidence:.2f}).",
            signals=signals,
        )

    if combined_confidence < CONFIDENCE_CLARIFY_THRESHOLD:
        return EscalationResult(
            decision=EscalationDecision.CLARIFY,
            reason=f"Confidence moderate ({combined_confidence:.2f}); asking for clarification.",
            signals=signals,
        )

    return EscalationResult(
        decision=EscalationDecision.DIRECT_ANSWER,
        reason="High confidence, no risk signals.",
        signals=signals,
    )
