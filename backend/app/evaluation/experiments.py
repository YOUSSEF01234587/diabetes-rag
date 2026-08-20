"""Experiments for chunking and embedding comparison."""
import json
import time
import csv
import logging
from pathlib import Path

from ..config import BASE_DIR, LOGS_DIR, DATA_DIR, EMBEDDING_MODEL
from ..ingestion.loader import load_documents
from ..ingestion.metadata import enrich_metadata
from ..ingestion.chunker import chunk_documents, generate_chunk_report
from ..retrieval.embeddings import embed_texts, embed_query, get_model_dimension
from ..retrieval.vector_store import reset_collection, add_chunks, query_dense
from ..retrieval.hybrid_search import build_bm25_index, search_bm25
from ..evaluation.evaluate import load_test_questions

logger = logging.getLogger(__name__)


def run_chunk_experiments():
    """Compare different chunk configurations."""
    pages = load_documents()
    pages = enrich_metadata(pages)

    configs = [
        {"max_tokens": 200, "overlap": 0, "name": "200_0"},
        {"max_tokens": 400, "overlap": 50, "name": "400_50"},
        {"max_tokens": 600, "overlap": 100, "name": "600_100"},
        {"max_tokens": 800, "overlap": 100, "name": "800_100"},
    ]

    questions = load_test_questions()
    results = []

    for config in configs:
        logger.info(f"Testing chunk config: {config['name']}")
        chunks = chunk_documents(pages, max_tokens=config["max_tokens"], overlap_tokens=config["overlap"])

        report = generate_chunk_report(chunks, str(LOGS_DIR / f"chunk_report_{config['name']}.json"))

        embeddings = embed_texts([c["text"] for c in chunks], model_name=EMBEDDING_MODEL)

        collection_name = f"exp_{config['name']}"
        collection = reset_collection(collection_name)
        add_chunks(chunks, embeddings, collection_name=collection_name)

        scores = []
        for q in questions:
            query_emb = embed_query(q["question"], model_name=EMBEDDING_MODEL)
            dense_results = query_dense(query_emb, top_k=10, collection_name=collection_name)
            top_score = dense_results[0]["dense_score"] if dense_results else 0
            scores.append(top_score)

        avg_score = sum(scores) / len(scores) if scores else 0

        results.append({
            "config": config["name"],
            "max_tokens": config["max_tokens"],
            "overlap": config["overlap"],
            "total_chunks": report["total_chunks"],
            "avg_tokens": round(report["avg_token_length"], 1),
            "min_tokens": report["min_token_length"],
            "max_tokens_actual": report["max_token_length"],
            "avg_retrieval_score": round(avg_score, 4),
            "num_questions": len(questions),
        })

    output_path = str(BASE_DIR / "evaluation" / "chunk_experiments.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Chunk experiments saved to {output_path}")
    for r in results:
        logger.info(f"  {r['config']}: {r['total_chunks']} chunks, avg_score={r['avg_retrieval_score']}")

    return results


def run_embedding_comparison():
    """Compare different embedding models."""
    models = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ]

    pages = load_documents()
    pages = enrich_metadata(pages)
    chunks = chunk_documents(pages)
    questions = load_test_questions()

    results = []
    for model_name in models:
        logger.info(f"Testing embedding model: {model_name}")

        t0 = time.time()
        embeddings = embed_texts([c["text"] for c in chunks], model_name=model_name)
        embed_time = time.time() - t0

        dim = get_model_dimension(model_name)

        collection_name = f"emb_{model_name.split('/')[-1]}"
        reset_collection(collection_name)
        add_chunks(chunks, embeddings, collection_name=collection_name)

        scores = []
        for q in questions:
            query_emb = embed_query(q["question"], model_name=model_name)
            dense_results = query_dense(query_emb, top_k=10, collection_name=collection_name)
            top_score = dense_results[0]["dense_score"] if dense_results else 0
            scores.append(top_score)

        avg_score = sum(scores) / len(scores) if scores else 0

        results.append({
            "model": model_name,
            "dimension": dim,
            "embedding_time_s": round(embed_time, 2),
            "avg_retrieval_score": round(avg_score, 4),
            "num_chunks": len(chunks),
            "num_questions": len(questions),
        })

    output_path = str(BASE_DIR / "evaluation" / "embedding_comparison.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Embedding comparison saved to {output_path}")
    return results
