"""Embedding model manager."""
import logging
import os
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """Get or load an embedding model (cached)."""
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Dimension: {_model_cache[model_name].get_embedding_dimension()}")
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts."""
    model = get_embedding_model(model_name)
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str, model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Embed a single query."""
    model = get_embedding_model(model_name)
    with torch.no_grad():
        embedding = model.encode([query], normalize_embeddings=True)
    return np.array(embedding[0], dtype=np.float32)


def get_model_dimension(model_name: str = "BAAI/bge-small-en-v1.5") -> int:
    """Get the embedding dimension of a model."""
    model = get_embedding_model(model_name)
    return model.get_embedding_dimension()
