from app.hospital.lookup import find_department, handle_query


def test_find_department_matches_name():
    dept = find_department("Where is Cardiology located?")
    assert dept is not None
    assert dept["name"] == "Cardiology"


def test_find_department_no_match_returns_none():
    dept = find_department("Where is the gift shop?")
    assert dept is None


def test_handle_query_department_info():
    response = handle_query("Where is Pediatrics located?")
    assert "Pediatrics" in response
    assert "Block B" in response


def test_handle_query_booking_intent_with_department():
    response = handle_query("I want to book an appointment with Orthopedics")
    assert "Orthopedics" in response
    assert "book" in response.lower()


def test_handle_query_booking_intent_without_department():
    response = handle_query("I want to book an appointment")
    assert "which department" in response.lower()


def test_handle_query_visiting_hours():
    response = handle_query("What are the visiting hours?")
    assert "4:00 PM" in response


def test_handle_query_unmatched_falls_back_to_department_list():
    response = handle_query("Tell me about the parking situation")
    assert "Cardiology" in response  # falls back to listing departments
