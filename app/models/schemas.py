"""
Shared data structures passed between pipeline stages.
Keeping these in one place makes it easy to log every stage's output
for your evaluation (hallucination rate, escalation precision/recall, etc.)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentLabel(str, Enum):
    APPOINTMENT_SERVICE = "appointment_service"
    HEALTH_INFO = "health_info"
    UNCLEAR_RISKY = "unclear_risky"


class EscalationDecision(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class IntentResult(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = []


class RetrievedChunk(BaseModel):
    question: str
    answer: str
    qtype: Optional[str] = None
    similarity: float


class RagResult(BaseModel):
    chunks: list[RetrievedChunk]
    top_similarity: float


class ValidationResult(BaseModel):
    is_supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class EscalationSignals(BaseModel):
    """The five signals the escalation policy weighs."""
    measurement_confidence: float = 1.0   # how reliable was the retrieval/lookup match
    evidence_confidence: float = 1.0      # how well-supported the generated answer is
    decision_risk: float = 0.0            # 0 = trivial, 1 = high clinical risk
    persistence: int = 0                  # repeated low-confidence turns this session
    logging_completeness: bool = True     # whether we have enough info logged to act on

    risk_keywords_hit: list[str] = []


class EscalationResult(BaseModel):
    decision: EscalationDecision
    reason: str
    signals: EscalationSignals


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    intent: IntentResult
    escalation: EscalationResult
    answer: str
    sources: list[RetrievedChunk] = []
