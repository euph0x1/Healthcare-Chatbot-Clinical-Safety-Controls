# Healthcare Chatbot with Clinical Safety Controls

Mini-project: an LLM-based healthcare assistant that answers appointment/service
queries (structured lookup) and general health information queries (RAG over
MedQuAD), with response validation and a graded escalation policy.

## Project Structure

```
healthcare-chatbot/
├── app/
│   ├── main.py                    # FastAPI entry point (/chat endpoint)
│   ├── config.py                  # env-based settings
│   ├── pipeline.py                # orchestrates the full flow (Stage 1-5)
│   ├── models/
│   │   └── schemas.py             # pydantic request/response models
│   ├── intent/
│   │   └── classifier.py          # Stage 1: intent classifier
│   ├── hospital/
│   │   └── lookup.py              # Stage 2: structured appointment/service lookup
│   ├── rag/
│   │   ├── ingest.py              # build the MedQuAD vector store
│   │   ├── retriever.py           # Stage 3: retrieval
│   │   └── generator.py           # Stage 3: grounded LLM generation
│   ├── validation/
│   │   └── response_validator.py  # Stage 4: hallucination / entailment check
│   └── escalation/
│       └── policy.py              # Stage 5: escalation policy check
├── data/
│   ├── medquad.csv                # <-- YOU add this (see Setup step 3)
│   ├── hospital_services.json     # synthetic hospital data (already included)
│   └── test_sets/
│       └── adversarial_queries.csv# hand-written risk test set (already included)
├── frontend/
│   └── streamlit_app.py           # simple chat UI + escalation log
├── scripts/
│   └── build_vector_store.py      # CLI: run this once after adding medquad.csv
├── tests/
│   ├── test_intent.py
│   ├── test_hospital_lookup.py
│   └── test_escalation.py
├── requirements.txt
├── .env.example
└── README.md
```

This maps directly onto the 5-stage pipeline from the architecture diagram:
`Intent Classifier -> (Structured Lookup | RAG Retrieval) -> Response
Validation -> Escalation Policy Check -> Direct Answer / Clarify / Escalate`.

## Setup (VS Code)

Requires Python 3.9+ — tested and confirmed working on Python 3.13.

1. **Open the folder in VS Code**, then create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up your environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your `GEMINI_API_KEY`. Get a free key (no
   credit card required) at https://aistudio.google.com/apikey -- this is
   the default provider, so the project runs at zero cost. If you later
   get paid Anthropic credits, set `LLM_PROVIDER=anthropic` and fill in
   `ANTHROPIC_API_KEY` instead; nothing else in the code needs to change.

3. **Add the MedQuAD dataset**:
   Download `medquad.csv` from Kaggle
   (https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)
   and place it at `data/medquad.csv`.

4. **Build the vector store** (one-time, run again if the CSV changes):
   ```bash
   python scripts/build_vector_store.py
   ```
   On the very first run, Chroma downloads a small local embedding model
   (all-MiniLM-L6-v2, ~90 MB) to embed the MedQuAD questions — this needs
   an internet connection once, after which everything runs offline. This
   step can take a few minutes for the full ~47k-row CSV.

5. **Run the backend**:
   ```bash
   uvicorn app.main:app --reload
   ```
   API docs at http://localhost:8000/docs

6. **Run the frontend** (in a second terminal):
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

7. **Run tests** (no API key needed — these cover the deterministic logic:
   intent classification, hospital lookup, escalation policy):
   ```bash
   pytest tests/ -v
   ```

## What's stubbed vs. what's real

- **Hospital lookup, intent classifier, escalation policy**: fully working
  logic on synthetic/rule-based data — run and test immediately, no API key
  needed.
- **RAG retrieval + generation + validation**: real code, but needs your
  `GEMINI_API_KEY` (free) and the MedQuAD CSV to run end-to-end.
- **Intent classifier** starts as keyword/rule-based (transparent, easy to
  evaluate for your report). Swapping in an embedding-similarity or
  LLM-based classifier later is a natural next iteration — the interface
  (`classify(query) -> IntentResult`) won't need to change.

## A note on the free Gemini tier

Google's free tier is genuinely $0, but it's rate-limited (requests per
minute and per day, not a token/dollar budget). If you're running a batch
evaluation (e.g. the whole `adversarial_queries.csv` test set at once),
add a short delay between calls or you may hit `429` rate-limit errors.
Check current limits at https://ai.google.dev/gemini-api/docs/rate-limits
since they do change.

## Next steps to build out for your report

- Expand `data/test_sets/adversarial_queries.csv` with more symptom/dosage/
  emergency examples and label them for your escalation precision/recall
  evaluation.
- Add an entailment metric log so you can compute hallucination rate over
  the held-out MedQuAD test split.
- Add session state (e.g. a simple in-memory dict keyed by session id) to
  track the "persistence" signal — repeated low-confidence turns from the
  same user should escalate faster.
