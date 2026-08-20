"""Reranker for retrieved chunks."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_reranker_model = None


def get_reranker(model_name: str = "BAAI/bge-reranker-base"):
    """Load reranker model."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {model_name}")
            _reranker_model = CrossEncoder(model_name, max_length=512)
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            _reranker_model = False
    return _reranker_model if _reranker_model is not False else None


def rerank(query: str, results: list[dict], top_k: int = 5, model_name: str = "BAAI/bge-reranker-base") -> list[dict]:
    """Rerank results using cross-encoder."""
    if not results:
        return []

    model = get_reranker(model_name)
    if model is None:
        logger.warning("Reranker not available, returning original order")
        for i, r in enumerate(results):
            r["reranker_score"] = r.get("dense_score", 0) * 0.5 + r.get("bm25_score", 0) * 0.5
            r["rank"] = i + 1
        return results[:top_k]

    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    for i, (r, score) in enumerate(zip(results, scores)):
        r["reranker_score"] = float(score)

    results.sort(key=lambda x: x["reranker_score"], reverse=True)

    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1

    return results[:top_k]
