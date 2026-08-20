"""Evidence selection experiment: test K=3,4,5,6 for evidence pack size.

Measures: citation correctness, grounding score, conflicts, prompt efficiency.
"""
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
from backend.app.evidence.evidence_validator import EvidenceValidator
from backend.app.evidence.evidence_pack import build_evidence_pack

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(os.path.join(LOGS_DIR, "experiments"), exist_ok=True)

K_VALUES = [3, 4, 5, 6]
SAMPLE_SIZE = 20


def load_answerable_questions():
    """Load answerable questions from test set."""
    test_path = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "evaluation", "test_set_v3.json")
    with open(test_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    answerable = [q for q in questions if q.get("answerable", True)]
    # Stratified sample
    from collections import defaultdict
    by_cat = defaultdict(list)
    for q in answerable:
        by_cat[q.get("category", "other")].append(q)
    sampled = []
    for cat, qs in by_cat.items():
        n = max(1, int(SAMPLE_SIZE * len(qs) / len(answerable)))
        sampled.extend(qs[:n])
    if len(sampled) < SAMPLE_SIZE:
        remaining = [q for q in answerable if q not in sampled]
        sampled.extend(remaining[:SAMPLE_SIZE - len(sampled)])
    return sampled[:SAMPLE_SIZE]


def run_experiment_for_k(questions, evidence_k):
    """Run retrieval + evidence validation for all questions at given K."""
    results = []
    validator = EvidenceValidator(evidence_k=evidence_k)

    for i, q in enumerate(questions):
        query = q["question"]
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
            print(f"  [{i+1}] Search failed for K={evidence_k}: {e}")
            continue

        elapsed_ms = (time.time() - t0) * 1000

        ev_result = validator.build_and_validate(query, search_results)
        ev_dict = ev_result.to_dict()

        results.append({
            "id": q["id"],
            "query": query,
            "category": q.get("category", "other"),
            "evidence_k": evidence_k,
            "grounding_score": ev_dict["grounding_score"],
            "is_grounded": ev_dict["is_grounded"],
            "citation_total": ev_dict["citation_report"]["total_citations"],
            "citation_valid": ev_dict["citation_report"]["valid_citations"],
            "citation_coverage": ev_dict["citation_report"]["citation_coverage"],
            "conflict_count": ev_dict["conflict_report"]["total_conflicts"],
            "has_population_conflict": ev_dict["conflict_report"]["has_population_conflict"],
            "needs_clarification": ev_dict["conflict_report"]["needs_clarification"],
            "evidence_chunks": ev_dict["evidence_summary"]["selected_chunks"],
            "source_agreement": ev_dict["evidence_summary"]["source_agreement"],
            "section_coherence": ev_dict["evidence_summary"]["section_coherence"],
            "has_table": ev_dict["evidence_summary"]["has_table_evidence"],
            "warnings": ev_dict["warnings"],
            "search_ms": round(elapsed_ms, 1),
        })

    return results


def compute_aggregate(results):
    """Compute aggregate metrics."""
    if not results:
        return {}
    n = len(results)
    return {
        "n": n,
        "avg_grounding_score": round(sum(r["grounding_score"] for r in results) / n, 4),
        "grounded_rate": round(sum(1 for r in results if r["is_grounded"]) / n, 4),
        "avg_citation_coverage": round(sum(r["citation_coverage"] for r in results) / n, 4),
        "avg_source_agreement": round(sum(r["source_agreement"] for r in results) / n, 4),
        "avg_section_coherence": round(sum(r["section_coherence"] for r in results) / n, 4),
        "conflict_rate": round(sum(1 for r in results if r["conflict_count"] > 0) / n, 4),
        "population_conflict_rate": round(sum(1 for r in results if r["has_population_conflict"]) / n, 4),
        "clarification_rate": round(sum(1 for r in results if r["needs_clarification"]) / n, 4),
        "avg_search_ms": round(sum(r["search_ms"] for r in results) / n, 1),
        "avg_evidence_chunks": round(sum(r["evidence_chunks"] for r in results) / n, 2),
    }


def main():
    print("=" * 60)
    print("Evidence Selection Experiment: K=3,4,5,6")
    print("=" * 60)

    questions = load_answerable_questions()
    print(f"Sampled {len(questions)} answerable questions")

    all_results = {}
    aggregates = {}

    for k in K_VALUES:
        print(f"\n--- Running K={k} ---")
        t0 = time.time()
        results = run_experiment_for_k(questions, k)
        elapsed = time.time() - t0

        agg = compute_aggregate(results)
        agg["wall_time_s"] = round(elapsed, 1)
        aggregates[f"K={k}"] = agg
        all_results[f"K={k}"] = results

        print(f"  Grounding: {agg['avg_grounding_score']:.4f} | "
              f"Grounded: {agg['grounded_rate']:.2%} | "
              f"Citation cov: {agg['avg_citation_coverage']:.4f} | "
              f"Conflicts: {agg['conflict_rate']:.2%} | "
              f"Time: {elapsed:.1f}s")

    # Save results
    output = {
        "experiment": "evidence_selection",
        "sample_size": len(questions),
        "k_values": K_VALUES,
        "aggregates": aggregates,
        "detailed_results": all_results,
    }

    out_path = os.path.join(LOGS_DIR, "experiments", "evidence_selection_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'K':>4} | {'Grounding':>10} | {'Grounded%':>10} | {'Citation':>10} | {'Conflicts':>10} | {'Time':>8}")
    print("-" * 80)
    for k in K_VALUES:
        a = aggregates[f"K={k}"]
        print(f"{k:>4} | {a['avg_grounding_score']:>10.4f} | {a['grounded_rate']:>9.2%} | "
              f"{a['avg_citation_coverage']:>10.4f} | {a['conflict_rate']:>9.2%} | {a['avg_search_ms']:>7.0f}ms")


if __name__ == "__main__":
    main()
