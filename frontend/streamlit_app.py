"""
Minimal Streamlit chat UI that calls the FastAPI backend and shows a live
escalation log in the sidebar, so you can see the routing decision (and
why) for every turn -- useful for demoing the safety controls.

Run: streamlit run frontend/streamlit_app.py
(Make sure `uvicorn app.main:app --reload` is running in another terminal.)
"""

import uuid

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "escalation_log" not in st.session_state:
    st.session_state.escalation_log = []

st.title("🏥 Healthcare Chatbot with Clinical Safety Controls")
st.caption("Case Study 23 mini project -- ask about appointments, hospital services, or general health information.")

col_chat, col_log = st.columns([2, 1])

with col_chat:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_message = st.chat_input("Type your question...")
    if user_message:
        st.session_state.history.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

        try:
            resp = requests.post(
                BACKEND_URL,
                json={"session_id": st.session_state.session_id, "message": user_message},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]

            st.session_state.history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**Q:** {s['question']}\n\n**A:** {s['answer']}")

            st.session_state.escalation_log.append(
                {
                    "query": user_message,
                    "intent": data["intent"]["label"],
                    "decision": data["escalation"]["decision"],
                    "reason": data["escalation"]["reason"],
                }
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend at {BACKEND_URL}. Is uvicorn running? ({e})")

with col_log:
    st.subheader("Escalation Log")
    if not st.session_state.escalation_log:
        st.write("No queries yet.")
    for entry in reversed(st.session_state.escalation_log):
        decision = entry["decision"]
        color = {"direct_answer": "green", "clarify": "orange", "escalate": "red"}.get(decision, "gray")
        st.markdown(
            f"**Query:** {entry['query']}\n\n"
            f"**Intent:** {entry['intent']}\n\n"
            f"**Decision:** :{color}[{decision}]\n\n"
            f"**Reason:** {entry['reason']}"
        )
        st.divider()
