"""Build the full search index."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from backend.app.config import (
    EMBEDDING_MODEL, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
    VECTOR_DB_DIR, LOGS_DIR,
)
from backend.app.ingestion.loader import load_documents
from backend.app.ingestion.metadata import enrich_metadata
from backend.app.ingestion.chunker import chunk_documents, generate_chunk_report
from backend.app.retrieval.embeddings import embed_texts
from backend.app.retrieval.vector_store import reset_collection, add_chunks
from backend.app.retrieval.hybrid_search import build_bm25_index


def build_full_index():
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("BUILDING DIABETES RAG INDEX")
    logger.info("=" * 60)

    logger.info("\n[1/6] Loading documents...")
    pages = load_documents()
    logger.info(f"  Total pages loaded: {len(pages)}")

    logger.info("\n[2/6] Enriching metadata...")
    pages = enrich_metadata(pages)

    logger.info("\n[3/6] Chunking documents (source-aware)...")
    chunks = chunk_documents(pages, max_tokens=CHUNK_SIZE_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    logger.info(f"  Total chunks: {len(chunks)}")

    chunk_report = generate_chunk_report(chunks, str(LOGS_DIR / "chunk_report.json"))
    logger.info(f"  Chunk report saved")
    logger.info(f"  Avg tokens/chunk: {chunk_report['avg_token_length']:.0f}")
    logger.info(f"  Pages covered: {chunk_report['unique_pages']}")

    logger.info(f"\n[4/6] Generating embeddings with {EMBEDDING_MODEL}...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=EMBEDDING_MODEL)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    logger.info("\n[5/6] Building vector store...")
    collection = reset_collection()
    add_chunks(chunks, embeddings)
    logger.info(f"  Vector store ready: {collection.count()} documents")

    logger.info("\n[6/6] Building BM25 index...")
    bm25_path = str(VECTOR_DB_DIR / "bm25_index.pkl")
    build_bm25_index(chunks, persist_path=bm25_path)

    elapsed = time.time() - t0

    index_report = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_database": "ChromaDB",
        "total_documents": len(pages),
        "total_chunks": len(chunks),
        "chunk_config": {
            "max_tokens": CHUNK_SIZE_TOKENS,
            "overlap_tokens": CHUNK_OVERLAP_TOKENS,
        },
        "chunks_per_document": chunk_report["chunks_per_document"],
        "pages_per_document": chunk_report["pages_per_document"],
        "sections": chunk_report["sections"],
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_time_seconds": round(elapsed, 2),
    }

    report_path = str(LOGS_DIR / "index_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(index_report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)

    return index_report


if __name__ == "__main__":
    build_full_index()
