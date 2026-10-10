"""
Run this once after placing medquad.csv in data/, and again whenever the
CSV changes.

Usage:
    python scripts/build_vector_store.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingest import build_vector_store  # noqa: E402

if __name__ == "__main__":
    csv_path = Path("data/medquad.csv")
    if not csv_path.exists():
        print(
            f"ERROR: {csv_path} not found.\n"
            "Download it from https://www.kaggle.com/datasets/"
            "pythonafroz/medquad-medical-question-answer-for-ai-research "
            "and place it at data/medquad.csv, then re-run this script."
        )
        sys.exit(1)

    n = build_vector_store(str(csv_path))
    print(f"Vector store built: {n} MedQuAD QA pairs indexed.")
