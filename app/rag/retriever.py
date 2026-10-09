"""
Stage 3 (retrieval half) of the pipeline: query the MedQuAD vector store
and return the top-k matches with a similarity score, so the escalation
policy has a "measurement confidence" signal to work with.
"""

import chromadb

from app.config import settings
from app.models.schemas import RagResult, RetrievedChunk

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = _client.get_collection(settings.chroma_collection)
    return _collection


def _distance_to_similarity(distance: float) -> float:
    """
    Chroma's default embedding space returns a squared-L2-ish distance;
    smaller is more similar. Convert to a rough 0-1 similarity score for
    readability in confidence thresholds and reports. This is an
    approximation -- tune/replace with your own normalization once you
    have a labeled evaluation set.
    """
    return max(0.0, 1.0 - distance / 2.0)


def retrieve(query: str, top_k: int | None = None) -> RagResult:
    top_k = top_k or settings.retrieval_top_k
    collection = _get_collection()

    results = collection.query(query_texts=[query], n_results=top_k)

    chunks: list[RetrievedChunk] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                question=doc,
                answer=meta.get("answer", ""),
                qtype=meta.get("qtype"),
                similarity=_distance_to_similarity(dist),
            )
        )

    top_similarity = chunks[0].similarity if chunks else 0.0
    return RagResult(chunks=chunks, top_similarity=top_similarity)
