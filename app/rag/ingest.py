"""
Builds the Chroma vector store from data/medquad.csv.
Run via scripts/build_vector_store.py -- do this once after adding the
CSV, and again any time the CSV changes.

Uses Chroma's default local embedding function (all-MiniLM-L6-v2 via
onnxruntime) so no external embedding API call/cost is needed. Swap this
for an API-based embedding model later if you want to compare quality.
"""

import pandas as pd
import chromadb

from app.config import settings

REQUIRED_COLUMNS = {"question", "answer"}


def load_medquad(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # MedQuAD CSV exports vary in column casing -- normalize.
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"medquad.csv is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question"])
    return df.reset_index(drop=True)


def build_vector_store(csv_path: str = "data/medquad.csv") -> int:
    df = load_medquad(csv_path)

    client = chromadb.PersistentClient(path=settings.chroma_dir)
    # Fresh build each run to keep this idempotent and simple.
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    collection = client.create_collection(settings.chroma_collection)

    batch_size = 500
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        collection.add(
            ids=[str(i) for i in batch.index],
            documents=batch["question"].tolist(),
            metadatas=[
                {
                    "answer": row["answer"],
                    "qtype": row.get("qtype", "unknown"),
                }
                for _, row in batch.iterrows()
            ],
        )
        print(f"Ingested {min(start + batch_size, len(df))}/{len(df)} rows...")

    return len(df)


if __name__ == "__main__":
    n = build_vector_store()
    print(f"Done. Indexed {n} MedQuAD QA pairs into '{settings.chroma_collection}'.")
