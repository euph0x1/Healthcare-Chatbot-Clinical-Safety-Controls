from fastapi import FastAPI

from app.models.schemas import ChatRequest, ChatResponse
from app.pipeline import run

app = FastAPI(
    title="Healthcare Chatbot with Clinical Safety Controls",
    description="Case Study 23 mini project -- LLM healthcare assistant "
    "with retrieval, response validation, and escalation.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return run(query=request.message, session_id=request.session_id)
