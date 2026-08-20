"""Generation evaluation: measures answer quality, citations, and refusal accuracy."""
import json
import os
import time
import logging
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.WARNING)

from backend.app.config import TOP_K, RERANK_TOP_K, VECTOR_DB_DIR

# Load BM25 index first
from backend.app.retrieval.hybrid_search import load_bm25_index
bm25_path = str(VECTOR_DB_DIR / "bm25_index.pkl")
loaded = load_bm25_index(bm25_path)
print(f"BM25 index loaded: {loaded}")

from backend.app.retrieval.retriever import hybrid_search
from backend.app.generation.llm import generate_answer
from backend.app.evidence.evidence_validator import EvidenceValidator

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(os.path.join(LOGS_DIR, "final"), exist_ok=True)


def load_generation_test_set():
    """Load the generation evaluation test set."""
    test_path = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "evaluation", "test_set_generation.json")
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_answer(answer_text, expected_keywords):
    """Simple keyword-based answer quality check."""
    if not answer_text or not expected_keywords:
        return 0.0
    answer_lower = answer_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords) if expected_keywords else 0.0


def check_citations(answer_text):
    """Check if answer contains [Evidence N] citations."""
    import re
    citations = re.findall(r'\[Evidence\s+\d+\]', answer_text)
    return len(citations)


def check_refusal(answer_text, is_answerable):
    """Check if refusal is appropriate."""
    refusal_phrases = [
        "i don't have enough evidence",
        "i don't have evidence",
        "insufficient evidence",
        "cannot answer",
        "unable to answer",
        "outside the scope",
        "requires professional",
        "consult a qualified",
    ]
    is_refusal = any(phrase in answer_text.lower() for phrase in refusal_phrases)

    if is_answerable:
        return not is_refusal  # Should NOT refuse for answerable
    else:
        return is_refusal  # Should refuse for unsupported


def run_generation_eval(evidence_k=5):
    """Run full generation evaluation."""
    questions = load_generation_test_set()
    print(f"Loaded {len(questions)} evaluation questions")
    print(f"Evidence K={evidence_k}")

    results = []
    total_t0 = time.time()

    for i, q in enumerate(questions):
        query = q["question"]
        qid = q["id"]
        is_answerable = q.get("answerable", True)
        category = q.get("category", "other")
        expected_keywords = q.get("expected_answer_keywords", [])

        print(f"[{i+1}/{len(questions)}] {qid} ({category})...", end=" ", flush=True)

        t0 = time.time()
        try:
            search_result = hybrid_search(
                query,
                top_k=TOP_K,
                rerank_top_k=RERANK_TOP_K,
                enable_reranker=False,
            )
            search_results = search_result["results"]
        except Exception as e:
            print(f"SEARCH FAILED: {e}")
            results.append({
                "id": qid,
                "query": query,
                "category": category,
                "answerable": is_answerable,
                "search_failed": True,
                "error": str(e),
            })
            continue

        search_ms = (time.time() - t0) * 1000

        # Generate answer
        try:
            gen_result = generate_answer(
                query,
                search_results,
                evidence_k=evidence_k,
            )
        except Exception as e:
            print(f"GENERATION FAILED: {e}")
            results.append({
                "id": qid,
                "query": query,
                "category": category,
                "answerable": is_answerable,
                "generation_failed": True,
                "error": str(e),
            })
            continue

        gen_ms = gen_result.get("timings", {}).get("llm_ms", 0)

        answer_text = gen_result.get("answer", "")
        is_refused = gen_result.get("refused", False)
        grounding_score = gen_result.get("grounding_score", 0.0)

        # Metrics
        answer_quality = evaluate_answer(answer_text, expected_keywords)
        citation_count = check_citations(answer_text)
        refusal_correct = check_refusal(answer_text, is_answerable)

        result = {
            "id": qid,
            "query": query,
            "category": category,
            "answerable": is_answerable,
            "refused": is_refused,
            "answer_quality": round(answer_quality, 3),
            "citation_count": citation_count,
            "refusal_correct": refusal_correct,
            "grounding_score": grounding_score,
            "answer_length": len(answer_text),
            "search_ms": round(search_ms, 1),
            "gen_ms": round(gen_ms, 1),
            "total_ms": round(search_ms + gen_ms, 1),
            "answer_preview": answer_text[:200] + "..." if len(answer_text) > 200 else answer_text,
        }
        results.append(result)

        status = "OK" if not is_refused else "REFUSED"
        print(f"{status} | Quality={answer_quality:.2f} | Citations={citation_count} | "
              f"Refusal={'CORRECT' if refusal_correct else 'WRONG'} | "
              f"Grounding={grounding_score:.3f} | {search_ms+gen_ms:.0f}ms")

    total_elapsed = time.time() - total_t0

    # Aggregate
    n = len(results)
    answerable_results = [r for r in results if r.get("answerable", True)]
    unsupported_results = [r for r in results if not r.get("answerable", True)]

    aggregates = {
        "total_questions": n,
        "answerable_count": len(answerable_results),
        "unsupported_count": len(unsupported_results),
        "avg_answer_quality": round(sum(r.get("answer_quality", 0) for r in results) / n, 4) if n else 0,
        "answerable_answer_quality": round(sum(r.get("answer_quality", 0) for r in answerable_results) / len(answerable_results), 4) if answerable_results else 0,
        "avg_citation_count": round(sum(r.get("citation_count", 0) for r in results) / n, 2) if n else 0,
        "avg_grounding_score": round(sum(r.get("grounding_score", 0) for r in results) / n, 4) if n else 0,
        "refusal_accuracy": round(sum(1 for r in results if r.get("refusal_correct", False)) / n, 4) if n else 0,
        "answerable_not_refused": round(sum(1 for r in answerable_results if not r.get("refused", True)) / len(answerable_results), 4) if answerable_results else 0,
        "unsupported_refused": round(sum(1 for r in unsupported_results if r.get("refused", False)) / len(unsupported_results), 4) if unsupported_results else 0,
        "avg_total_ms": round(sum(r.get("total_ms", 0) for r in results) / n, 1) if n else 0,
        "wall_time_s": round(total_elapsed, 1),
    }

    # By category
    categories = {}
    for r in results:
        cat = r.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_aggs = {}
    for cat, cat_results in categories.items():
        category_aggs[cat] = {
            "count": len(cat_results),
            "avg_quality": round(sum(r.get("answer_quality", 0) for r in cat_results) / len(cat_results), 4) if cat_results else 0,
            "refusal_accuracy": round(sum(1 for r in cat_results if r.get("refusal_correct", False)) / len(cat_results), 4) if cat_results else 0,
            "avg_citations": round(sum(r.get("citation_count", 0) for r in cat_results) / len(cat_results), 2) if cat_results else 0,
            "refused_rate": round(sum(1 for r in cat_results if r.get("refused", False)) / len(cat_results), 4) if cat_results else 0,
        }

    output = {
        "experiment": "generation_evaluation",
        "evidence_k": evidence_k,
        "aggregates": aggregates,
        "by_category": category_aggs,
        "detailed_results": results,
    }

    out_path = os.path.join(LOGS_DIR, "final", "generation_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Questions: {n} ({aggregates['answerable_count']} answerable, {aggregates['unsupported_count']} unsupported)")
    print(f"Answer Quality: {aggregates['avg_answer_quality']:.4f} (answerable: {aggregates['answerable_answer_quality']:.4f})")
    print(f"Avg Citations: {aggregates['avg_citation_count']:.2f}")
    print(f"Avg Grounding: {aggregates['avg_grounding_score']:.4f}")
    print(f"Refusal Accuracy: {aggregates['refusal_accuracy']:.4f}")
    print(f"  Answerable not refused: {aggregates['answerable_not_refused']:.4f}")
    print(f"  Unsupported refused: {aggregates['unsupported_refused']:.4f}")
    print(f"Avg Latency: {aggregates['avg_total_ms']:.0f}ms")
    print(f"Wall Time: {aggregates['wall_time_s']:.0f}s")

    print("\nBy Category:")
    print(f"{'Category':>20} | {'Count':>5} | {'Quality':>8} | {'Refusal':>8} | {'Citations':>9} | {'Refused%':>8}")
    print("-" * 80)
    for cat, agg in sorted(category_aggs.items()):
        print(f"{cat:>20} | {agg['count']:>5} | {agg['avg_quality']:>8.4f} | "
              f"{agg['refusal_accuracy']:>8.4f} | {agg['avg_citations']:>9.2f} | {agg['refused_rate']:>7.2%}")

    return output


if __name__ == "__main__":
    run_generation_eval(evidence_k=3)
