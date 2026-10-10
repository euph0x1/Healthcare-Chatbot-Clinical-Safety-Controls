from app.intent.classifier import classify
from app.models.schemas import IntentLabel

def test_appointment_booking_detected():
    result = classify("I want to book an appointment with cardiology")
    assert result.label == IntentLabel.APPOINTMENT_SERVICE


def test_health_info_detected():
    result = classify("What are the symptoms of diabetes?")
    assert result.label == IntentLabel.HEALTH_INFO


def test_emergency_keyword_forces_unclear_risky():
    result = classify("I have chest pain and can't breathe")
    assert result.label == IntentLabel.UNCLEAR_RISKY
    assert result.confidence > 0.5


def test_dosage_keyword_forces_unclear_risky():
    result = classify("How many mg of paracetamol should I take?")
    assert result.label == IntentLabel.UNCLEAR_RISKY


def test_visiting_hours_is_appointment_service():
    result = classify("What are the visiting hours?")
    assert result.label == IntentLabel.APPOINTMENT_SERVICE

def test_prevent_flu_is_health_info_regardless_of_punctuation():
    with_q = classify("How can I prevent the flu?")
    without_q = classify("How can I prevent the flu")
    assert with_q.label == IntentLabel.HEALTH_INFO
    assert without_q.label == IntentLabel.HEALTH_INFO