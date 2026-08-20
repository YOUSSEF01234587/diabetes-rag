"""BM25 lexical search."""
import re
import json
import pickle
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] = []
_bm25_tokenized: list[list[str]] = []


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\.%]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: list[dict], persist_path: str | None = None):
    """Build BM25 index from chunks."""
    global _bm25_index, _bm25_chunks, _bm25_tokenized

    _bm25_chunks = chunks
    _bm25_tokenized = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(_bm25_tokenized)

    if persist_path:
        data = {
            "chunks": chunks,
            "tokenized": _bm25_tokenized,
        }
        with open(persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {persist_path}")

    logger.info(f"BM25 index built: {len(chunks)} chunks")
    return _bm25_index


def load_bm25_index(persist_path: str) -> bool:
    """Load BM25 index from disk."""
    global _bm25_index, _bm25_chunks, _bm25_tokenized

    if not Path(persist_path).exists():
        return False

    with open(persist_path, "rb") as f:
        data = pickle.load(f)

    _bm25_chunks = data["chunks"]
    _bm25_tokenized = data["tokenized"]
    _bm25_index = BM25Okapi(_bm25_tokenized)
    logger.info(f"BM25 index loaded: {len(_bm25_chunks)} chunks")
    return True


def search_bm25(query: str, top_k: int = 20) -> list[dict]:
    """Search using BM25."""
    global _bm25_index, _bm25_chunks

    if _bm25_index is None:
        logger.warning("BM25 index not built")
        return []

    query_tokens = _tokenize(query)
    scores = _bm25_index.get_scores(query_tokens)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append({
            "chunk_id": _bm25_chunks[idx]["chunk_id"],
            "text": _bm25_chunks[idx]["text"],
            "metadata": _bm25_chunks[idx],
            "bm25_score": float(scores[idx]),
        })

    return results
