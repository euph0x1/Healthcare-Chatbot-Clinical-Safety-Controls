"""
Stage 2 of the pipeline: Structured Appointment & Service Lookup.

Handles the APPOINTMENT_SERVICE intent branch. This is a plain lookup
against a small local JSON "database" -- deliberately not LLM-generated,
so there is zero hallucination risk on this branch. Appointment booking
itself is simulated (intent + confirmation), not wired to a real backend,
per the project's stated constraints.
"""

import json
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "hospital_services.json"

BOOKING_INTENT_WORDS = ["book", "booking", "reschedule", "cancel", "cancellation"]


def _load_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_department(query: str, data: Optional[dict] = None) -> Optional[dict]:
    """Return the department dict whose name appears in the query, if any."""
    data = data or _load_data()
    query_lower = query.lower()
    for dept in data["departments"]:
        if dept["name"].lower() in query_lower:
            return dept
    return None


def handle_query(query: str) -> str:
    """
    Produce a templated response for an appointment/service query.
    Pure lookup + string templating -- no LLM call, so it's cheap and
    hallucination-free, matching the "Template-Based Response" box in the
    architecture diagram.
    """
    data = _load_data()
    query_lower = query.lower()

    dept = find_department(query, data)

    if any(word in query_lower for word in BOOKING_INTENT_WORDS):
        if dept:
            return (
                f"I can help you book with {dept['name']}. Available doctors: "
                f"{', '.join(dept['doctors'])}. Department hours: {dept['timings']}. "
                f"Please confirm a preferred date and time, and I'll simulate "
                f"the booking confirmation for you."
            )
        return (
            "I can help you book an appointment. Which department would you "
            "like to see -- for example Cardiology, General Medicine, "
            "Pediatrics, Orthopedics, or Dermatology?"
        )

    if dept:
        return (
            f"{dept['name']} is located at {dept['location']}. "
            f"Timings: {dept['timings']}. Doctors: {', '.join(dept['doctors'])}."
        )

    if "visiting hour" in query_lower:
        return f"Visiting hours are {data['hospital_info']['visiting_hours']}."

    if "registration" in query_lower or "opd" in query_lower:
        return f"OPD registration: {data['hospital_info']['opd_registration']}."

    if "billing" in query_lower or "payment" in query_lower or "insurance" in query_lower:
        return f"Billing desk: {data['hospital_info']['billing_desk']}."

    # Fall back to listing departments so the user can narrow down.
    dept_names = ", ".join(d["name"] for d in data["departments"])
    return (
        "I couldn't match that to a specific department or service. "
        f"Our departments are: {dept_names}. Could you specify which one, "
        "or ask about visiting hours, OPD registration, or billing?"
    )
