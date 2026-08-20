"""Evaluation pipeline for Diabetes RAG."""
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .metrics import compute_retrieval_metrics
from ..retrieval.retriever import hybrid_search
from ..config import TOP_K, RERANK_TOP_K, LOGS_DIR

logger = logging.getLogger(__name__)


def load_test_questions(path: Optional[str] = None) -> list[dict]:
    """Load test questions from JSON."""
    if path is None:
        path = Path(__file__).parent / "test_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    questions: list[dict] = None,
    top_k: int = TOP_K,
    output_path: str = None,
) -> dict:
    """Run full evaluation pipeline."""
    if questions is None:
        questions = load_test_questions()

    logger.info(f"Running evaluation with {len(questions)} questions")

    results = []
    for i, q in enumerate(questions):
        question_text = q["question"]
        logger.info(f"[{i+1}/{len(questions)}] {question_text[:80]}...")

        t0 = time.time()
        search_result = hybrid_search(
            question_text,
            top_k=top_k,
            rerank_top_k=RERANK_TOP_K,
            enable_reranker=False,
        )
        elapsed = time.time() - t0

        retrieved = search_result["results"]
        retrieved_source_ids = [r.get("source_id", "") for r in retrieved]
        retrieved_pages = [r.get("page_document", 0) for r in retrieved]

        expected_source = q.get("expected_source", "")
        expected_pages = q.get("expected_pages", [])

        source_correct = expected_source in retrieved_source_ids if expected_source else False
        page_correct = any(p in expected_pages for p in retrieved_pages) if expected_pages else False

        must_refuse = q.get("must_refuse", False)
        top_score = retrieved[0].get("fusion_score", 0) if retrieved else 0

        results.append({
            "question": question_text,
            "query_type": q.get("answer_type", "factual"),
            "expected_source": expected_source,
            "retrieved_sources": retrieved_source_ids[:3],
            "source_correct": source_correct,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages[:3],
            "page_correct": page_correct,
            "top_score": round(top_score, 4),
            "num_results": len(retrieved),
            "must_refuse": must_refuse,
            "retrieval_ms": round(elapsed * 1000, 1),
            "results": retrieved,
        })

    retrieval_input = []
    for r in results:
        relevant_ids = set()
        for chunk in r["results"]:
            chunk_source = chunk.get("source_id", "")
            chunk_page = chunk.get("page_document", 0)
            expected_source = r.get("expected_source", "")
            expected_pages = r.get("expected_pages", [])
            if expected_source and chunk_source == expected_source:
                if not expected_pages or chunk_page in expected_pages:
                    relevant_ids.add(chunk["chunk_id"])
        retrieval_input.append({
            "results": r["results"],
            "relevant_chunk_ids": relevant_ids,
        })

    retrieval_metrics = compute_retrieval_metrics(retrieval_input)

    source_accuracy = sum(1 for r in results if r["source_correct"]) / len(results) if results else 0
    page_accuracy = sum(1 for r in results if r["page_correct"]) / len(results) if results else 0

    eval_report = {
        "num_questions": len(questions),
        "retrieval_metrics": retrieval_metrics,
        "source_accuracy": round(source_accuracy, 4),
        "page_accuracy": round(page_accuracy, 4),
        "avg_top_score": round(
            sum(r["top_score"] for r in results) / len(results), 4
        ) if results else 0,
        "questions": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if output_path is None:
        output_path = str(LOGS_DIR / "evaluation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return eval_report
