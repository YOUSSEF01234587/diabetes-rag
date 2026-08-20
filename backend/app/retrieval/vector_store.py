"""ChromaDB vector store management."""
import json
import logging
import chromadb
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def get_collection(collection_name: str = "diabetes_rag", persist_dir: str = None) -> chromadb.Collection:
    """Get or create the collection."""
    global _collection
    if _collection is not None:
        return _collection

    if persist_dir is None:
        from ..config import VECTOR_DB_DIR
        persist_dir = str(VECTOR_DB_DIR)

    client = get_client(persist_dir)
    _collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{collection_name}': {_collection.count()} documents")
    return _collection


def reset_collection(collection_name: str = "diabetes_rag", persist_dir: str = None):
    """Delete and recreate the collection."""
    global _collection, _client
    _collection = None
    _client = None

    if persist_dir is None:
        from ..config import VECTOR_DB_DIR
        persist_dir = str(VECTOR_DB_DIR)

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
    except Exception:
        pass

    _client = client
    _collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Created fresh collection '{collection_name}'")
    return _collection


def add_chunks(
    chunks: list[dict],
    embeddings: np.ndarray,
    collection_name: str = "diabetes_rag",
    batch_size: int = 500,
):
    """Add chunks with embeddings to the vector store."""
    collection = get_collection(collection_name)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "source_id": c.get("source_id", ""),
            "source_title": c.get("source_title", ""),
            "short_title": c.get("short_title", ""),
            "organization": c.get("organization", ""),
            "page_pdf": c.get("page_pdf", 0),
            "page_document": c.get("page_document", 0),
            "section": c.get("section", ""),
            "subsection": c.get("subsection") or "",
            "doi": c.get("doi") or "",
            "official_url": c.get("official_url") or "",
            "year": c.get("year") or 0,
            "authority": c.get("authority", "high"),
            "has_table": c.get("has_table", False),
            "is_reference": c.get("is_reference", False),
            "token_estimate": c.get("token_estimate", 0),
            "parent_chunk_id": c.get("parent_chunk_id") or "",
        }
        metadatas.append(meta)

    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )
        logger.info(f"Added batch {i//batch_size + 1}: {end - i} chunks")

    logger.info(f"Total chunks in collection: {collection.count()}")


def query_dense(
    query_embedding: np.ndarray,
    top_k: int = 20,
    collection_name: str = "diabetes_rag",
) -> list[dict]:
    """Dense vector search."""
    collection = get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance
            output.append({
                "chunk_id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "dense_score": max(0.0, score),
            })

    return output


def get_index_stats(collection_name: str = "diabetes_rag") -> dict:
    """Get statistics about the index."""
    collection = get_collection(collection_name)
    count = collection.count()

    sample = collection.peek(limit=min(5, count)) if count > 0 else {}

    source_ids = set()
    if sample and "metadatas" in sample:
        for meta in sample["metadatas"]:
            if meta:
                source_ids.add(meta.get("source_id", ""))

    return {
        "total_chunks": count,
        "collection_name": collection_name,
        "sample_source_ids": list(source_ids),
    }
