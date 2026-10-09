"""
Stage 1 of the pipeline: Intent Classifier.

Routes every incoming query into one of three buckets:
  - APPOINTMENT_SERVICE : booking/rescheduling/status, department/timing info
  - HEALTH_INFO         : general medical/health information questions
  - UNCLEAR_RISKY        : ambiguous, or contains risk signals that should be
                            handled cautiously regardless of topic

This starts as a transparent, rule-based classifier so it's easy to test
and evaluate for your report. The public interface (`classify`) is stable,
so you can later swap the implementation for an embedding-similarity or
LLM-based classifier without touching the rest of the pipeline.
"""

from app.models.schemas import IntentLabel, IntentResult

APPOINTMENT_SERVICE_KEYWORDS = [
    "appointment", "book", "booking", "reschedule", "cancel", "cancellation",
    "slot", "opd", "timing", "timings", "visiting hour", "visiting hours", "department",
    "doctor available", "which floor", "location of", "where is",
    "how do i pay", "billing", "insurance desk", "registration",
]

HEALTH_INFO_KEYWORDS = [
    "symptom", "cause", "treatment", "treat", "cure", "prevent",
    "how does", "risk factor", "diagnosed with",
    "disease", "vaccine", "infection", "nutrition", "diet", "contagious",
    "spread", "complication", "manage", "avoid getting",
]

# These do not decide the branch by themselves, but they always push the
# query toward UNCLEAR_RISKY so the escalation policy gets a first look.
RISK_SIGNAL_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "overdose", "suicide", "self harm", "unconscious", "bleeding heavily",
    "severe pain", "emergency", "mg of", "milligrams of", "dosage of",
    "how much should i take", "diagnose me", "do i have",
]


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def classify(query: str) -> IntentResult:
    """Classify a single user query. Pure function, no side effects."""
    risk_hits = _keyword_hits(query, RISK_SIGNAL_KEYWORDS)
    if risk_hits:
        return IntentResult(
            label=IntentLabel.UNCLEAR_RISKY,
            confidence=0.9,
            matched_keywords=risk_hits,
        )

    appt_hits = _keyword_hits(query, APPOINTMENT_SERVICE_KEYWORDS)
    info_hits = _keyword_hits(query, HEALTH_INFO_KEYWORDS)

    if appt_hits and not info_hits:
        return IntentResult(
            label=IntentLabel.APPOINTMENT_SERVICE,
            confidence=min(0.6 + 0.1 * len(appt_hits), 0.95),
            matched_keywords=appt_hits,
        )

    if info_hits and not appt_hits:
        return IntentResult(
            label=IntentLabel.HEALTH_INFO,
            confidence=min(0.6 + 0.1 * len(info_hits), 0.95),
            matched_keywords=info_hits,
        )

    if appt_hits and info_hits:
        # Ambiguous mix of both -> let the escalation policy weigh in via
        # low confidence rather than guessing.
        return IntentResult(
            label=IntentLabel.UNCLEAR_RISKY,
            confidence=0.4,
            matched_keywords=appt_hits + info_hits,
        )

    # No keyword matched either bucket, and no risk signal was found. Rather
    # than guess based on punctuation, default to HEALTH_INFO at low
    # confidence: the RAG + response-validation stages downstream will
    # correctly decline/escalate if there's no good source match, which is
    # a safer fallback than defaulting to UNCLEAR_RISKY (which currently
    # skips retrieval entirely and heads straight for escalation).
    return IntentResult(label=IntentLabel.HEALTH_INFO, confidence=0.25)