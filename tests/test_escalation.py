import uuid

from app.escalation.policy import evaluate
from app.models.schemas import EscalationDecision


def _fresh_session() -> str:
    return str(uuid.uuid4())


def test_emergency_keyword_always_escalates():
    result = evaluate(
        query="I have chest pain and can't breathe",
        session_id=_fresh_session(),
        measurement_confidence=1.0,
        evidence_confidence=1.0,
    )
    assert result.decision == EscalationDecision.ESCALATE


def test_dosage_question_escalates_even_with_high_confidence():
    result = evaluate(
        query="How many mg of paracetamol should I take?",
        session_id=_fresh_session(),
        measurement_confidence=0.95,
        evidence_confidence=0.95,
    )
    assert result.decision == EscalationDecision.ESCALATE


def test_high_confidence_no_risk_gives_direct_answer():
    result = evaluate(
        query="What is the treatment for the common cold?",
        session_id=_fresh_session(),
        measurement_confidence=0.9,
        evidence_confidence=0.9,
    )
    assert result.decision == EscalationDecision.DIRECT_ANSWER


def test_low_confidence_clarifies_not_escalates_on_first_turn():
    result = evaluate(
        query="What about that thing for the skin issue?",
        session_id=_fresh_session(),
        measurement_confidence=0.5,
        evidence_confidence=0.5,
    )
    assert result.decision == EscalationDecision.CLARIFY


def test_very_low_confidence_escalates():
    result = evaluate(
        query="Tell me about it",
        session_id=_fresh_session(),
        measurement_confidence=0.1,
        evidence_confidence=0.1,
    )
    assert result.decision == EscalationDecision.ESCALATE


def test_persistence_triggers_escalation_after_repeated_low_confidence():
    session_id = _fresh_session()
    first = evaluate("vague question one", session_id, 0.5, 0.5)
    second = evaluate("vague question two", session_id, 0.5, 0.5)

    # First low-confidence turn -> clarify. A second consecutive
    # low-confidence turn in the same session should escalate rather
    # than keep asking the user to clarify indefinitely.
    assert first.decision == EscalationDecision.CLARIFY
    assert second.decision == EscalationDecision.ESCALATE
    assert second.signals.persistence >= 2
