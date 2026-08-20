"""V5 Comprehensive Evaluation Pipeline — Retrieval + Generation + Safety + Regression.

Usage:
    python scripts/v5_eval.py --retrieval          # Retrieval eval only (fast, no LLM)
    python scripts/v5_eval.py --smoke              # 10-question generation smoke test
    python scripts/v5_eval.py --generation         # Full 119-question generation eval
    python scripts/v5_eval.py --regression         # Compare against baseline, fail if degraded
    python scripts/v5_eval.py --all                # Full pipeline
"""
import os, sys, json, time, argparse, logging, hashlib
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
logging.basicConfig(level=logging.WARNING)

LOGS_DIR = Path("logs/final")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PATH = LOGS_DIR / "final_baseline_snapshot.json"

from backend.app.config import (
    VECTOR_DB_DIR, TOP_K, RERANK_TOP_K, DENSE_WEIGHT,
    EMBEDDING_MODEL, RERANKER_ENABLED, LLM_MODEL, LLM_PROVIDER,
)
from backend.app.retrieval.hybrid_search import load_bm25_index
from backend.app.retrieval.retriever import hybrid_search
from backend.app.retrieval.page_sections import is_relevant_section

load_bm25_index(str(VECTOR_DB_DIR / "bm25_index.pkl"))


def load_test_set():
    path = Path("backend/app/evaluation/test_set_v3.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval_search(query_text):
    return hybrid_search(
        query_text,
        top_k=TOP_K,
        rerank_top_k=RERANK_TOP_K,
        enable_reranker=False,
        dense_weight=DENSE_WEIGHT,
        use_multi_query=True,
        use_query_expansion=False,
    )


def eval_retrieval(questions):
    results = []
    for i, q in enumerate(questions):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(questions)}]")
        t0 = time.time()
        sr = run_retrieval_search(q["question"])
        elapsed = (time.time() - t0) * 1000

        retrieved = sr["results"]
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_sections = [r.get("section", "") for r in retrieved]
        retrieved_pages = [r.get("page_document", 0) for r in retrieved]
        retrieved_sources = [r.get("source_id", "") for r in retrieved]

        expected_sources = q.get("expected_sources", [])
        expected_sections = q.get("expected_sections", [])
        expected_pages = q.get("expected_pages", [])
        answerable = q.get("answerable", True)

        r1 = r3 = r5 = r10 = False
        for j, chunk in enumerate(retrieved):
            cs = chunk.get("source_id", "")
            cp = chunk.get("page_document", 0)
            if cs in expected_sources:
                if not expected_pages or cp in expected_pages:
                    if j < 1: r1 = True
                    if j < 3: r3 = True
                    if j < 5: r5 = True
                    if j < 10: r10 = True

        if not r5:
            for j, chunk in enumerate(retrieved):
                cs = chunk.get("source_id", "")
                if cs in expected_sources:
                    if j < 5: r5 = True
                    if j < 10: r10 = True

        mrr = 0.0
        for j, chunk in enumerate(retrieved):
            cs = chunk.get("source_id", "")
            cp = chunk.get("page_document", 0)
            if cs in expected_sources:
                if not expected_pages or cp in expected_pages:
                    mrr = 1.0 / (j + 1)
                    break
        if mrr == 0.0:
            for j, chunk in enumerate(retrieved):
                cs = chunk.get("source_id", "")
                if cs in expected_sources:
                    mrr = 1.0 / (j + 1)
                    break

        source_correct = any(s in expected_sources for s in retrieved_sources) if expected_sources else True
        section_correct = False
        for rs, rp in zip(retrieved_sections, retrieved_pages):
            if is_relevant_section(rs, expected_sections, rp):
                section_correct = True
                break

        results.append({
            "id": q.get("id", f"Q{i+1}"),
            "question": q["question"],
            "category": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            "answerable": answerable,
            "r1": r1, "r3": r3, "r5": r5, "r10": r10,
            "mrr": mrr,
            "source_correct": source_correct,
            "section_correct": section_correct,
            "latency_ms": round(elapsed, 1),
            "intent": sr.get("intent", {}).get("primary", "") if sr.get("intent") else "",
        })

    return results


def compute_retrieval_metrics(results):
    answerable = [r for r in results if r.get("answerable", True)]
    n = len(answerable)
    if n == 0:
        return {}

    return {
        "total": len(results),
        "answerable": n,
        "r@1": sum(r["r1"] for r in answerable) / n,
        "r@3": sum(r["r3"] for r in answerable) / n,
        "r@5": sum(r["r5"] for r in answerable) / n,
        "r@10": sum(r["r10"] for r in answerable) / n,
        "mrr": sum(r["mrr"] for r in answerable) / n,
        "source_acc": sum(r["source_correct"] for r in results) / len(results),
        "section_acc": sum(r["section_correct"] for r in results) / len(results),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
    }


def run_generation_eval(questions, evidence_k=5, smoke=False):
    from backend.app.generation.llm import generate_answer
    results = []
    for i, q in enumerate(questions):
        if smoke and i >= 10:
            break
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(questions)}]")
        qid = q.get("id", f"Q{i+1}")
        sr = run_retrieval_search(q["question"])
        retrieved = sr["results"]

        t0 = time.time()
        gen_result = generate_answer(q["question"], retrieved, evidence_k=evidence_k)
        gen_ms = (time.time() - t0) * 1000

        results.append({
            "id": qid,
            "question": q["question"],
            "category": q.get("category", "unknown"),
            "answerable": q.get("answerable", True),
            "refused": gen_result.get("refused", False),
            "confidence": gen_result.get("confidence", "none"),
            "grounding_score": gen_result.get("grounding_score", 0),
            "verification_passed": gen_result.get("verification", {}).get("passed", False),
            "citation_count": len(gen_result.get("citations", [])),
            "safety_risk": gen_result.get("safety", {}).get("risk_level", "low"),
            "llm_ms": gen_result.get("timings", {}).get("llm_ms", 0),
            "llm_log": gen_result.get("timings", {}).get("llm", {}),
            "gen_ms": gen_ms,
            "has_answer": bool(gen_result.get("answer")),
            "answer_preview": gen_result.get("answer", "")[:200],
        })

    return results


def compute_generation_metrics(gen_results):
    n = len(gen_results)
    if n == 0:
        return {}

    answerable = [r for r in gen_results if r.get("answerable", True)]
    unanswerable = [r for r in gen_results if not r.get("answerable", True)]

    total_llm_time = sum(r.get("llm_ms", 0) for r in gen_results)
    failures = [r for r in gen_results if r.get("llm_log", {}).get("failure_type", "none") != "none"]

    return {
        "total": n,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "refusal_rate": sum(1 for r in gen_results if r["refused"]) / n,
        "answerable_refused": sum(1 for r in answerable if r["refused"]) / max(1, len(answerable)),
        "unanswerable_refused": sum(1 for r in unanswerable if r["refused"]) / max(1, len(unanswerable)),
        "avg_grounding": sum(r.get("grounding_score", 0) for r in gen_results) / n,
        "verification_pass_rate": sum(1 for r in gen_results if r["verification_passed"]) / n,
        "avg_citations": sum(r.get("citation_count", 0) for r in gen_results) / n,
        "safety_all_low": all(r.get("safety_risk", "low") == "low" or not r.get("answerable") for r in gen_results),
        "avg_llm_ms": total_llm_time / n,
        "llm_failure_rate": len(failures) / n,
        "timeout_rate": sum(1 for r in failures if r.get("llm_log", {}).get("failure_type") in ("timeout", "connection")) / n,
    }


def check_regression(current_metrics, baseline_metrics, thresholds=None):
    if thresholds is None:
        thresholds = {
            "r@5": 0.02,
            "mrr": 0.02,
            "source_acc": 0.02,
            "section_acc": 0.02,
        }

    issues = []
    for metric, threshold in thresholds.items():
        base_val = baseline_metrics.get(metric, 0)
        curr_val = current_metrics.get(metric, 0)
        if base_val > 0 and curr_val < base_val - threshold:
            pct = ((curr_val - base_val) / base_val) * 100
            issues.append(f"{metric}: {curr_val:.4f} < {base_val:.4f} - {threshold} ({pct:+.1f}%)")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="V5 Evaluation Pipeline")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run 10-question smoke generation test")
    parser.add_argument("--generation", action="store_true", help="Run full generation evaluation")
    parser.add_argument("--regression", action="store_true", help="Compare against baseline")
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    if not any([args.retrieval, args.smoke, args.generation, args.regression, args.all]):
        args.retrieval = True

    questions = load_test_set()
    print(f"Loaded {len(questions)} questions")

    timestamp = datetime.now().isoformat()

    if args.retrieval or args.all:
        print("\n=== RETRIEVAL EVALUATION ===")
        print(f"Config: DW={DENSE_WEIGHT}, RK={RERANK_TOP_K}, TOP_K={TOP_K}")
        retrieval_results = eval_retrieval(questions)
        metrics = compute_retrieval_metrics(retrieval_results)

        print(f"\nR@1:  {metrics.get('r@1', 0):.4f}")
        print(f"R@3:  {metrics.get('r@3', 0):.4f}")
        print(f"R@5:  {metrics.get('r@5', 0):.4f}")
        print(f"R@10: {metrics.get('r@10', 0):.4f}")
        print(f"MRR:  {metrics.get('mrr', 0):.4f}")
        print(f"Src:  {metrics.get('source_acc', 0):.4f}")
        print(f"Sec:  {metrics.get('section_acc', 0):.4f}")
        print(f"Lat:  {metrics.get('avg_latency_ms', 0):.0f}ms")

        output = {
            "timestamp": timestamp,
            "config": {
                "dense_weight": DENSE_WEIGHT,
                "rerank_top_k": RERANK_TOP_K,
                "top_k": TOP_K,
                "reranker": False,
                "multi_query": True,
            },
            "metrics": metrics,
            "results": retrieval_results,
        }
        out_path = LOGS_DIR / "v5_retrieval_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults: {out_path}")

    if args.smoke or args.generation or args.all:
        print("\n=== GENERATION EVALUATION ===")
        smoke = args.smoke and not args.generation and not args.all
        gen_results = run_generation_eval(questions, smoke=smoke)
        gen_metrics = compute_generation_metrics(gen_results)

        print(f"\nRefusal:      {gen_metrics.get('refusal_rate', 0):.4f}")
        print(f"Verify Pass:  {gen_metrics.get('verification_pass_rate', 0):.4f}")
        print(f"Avg Ground:   {gen_metrics.get('avg_grounding', 0):.4f}")
        print(f"Avg LLM ms:   {gen_metrics.get('avg_llm_ms', 0):.0f}")
        print(f"LLM Failures: {gen_metrics.get('llm_failure_rate', 0):.4f}")

        output = {
            "timestamp": timestamp,
            "metrics": gen_metrics,
            "results": gen_results,
        }
        out_path = LOGS_DIR / ("v5_smoke_gen.json" if smoke else "v5_generation_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults: {out_path}")

    if args.regression or args.all:
        print("\n=== REGRESSION CHECK ===")
        if not BASELINE_PATH.exists():
            print("No baseline found, skipping regression check")
        else:
            with open(BASELINE_PATH, "r") as f:
                baseline = json.load(f)
            base_metrics = baseline.get("retrieval_metrics", {})
            curr_metrics = compute_retrieval_metrics(eval_retrieval(questions))

            reg = check_regression(curr_metrics, base_metrics)
            if reg["passed"]:
                print("REGRESSION: PASSED — no metric dropped beyond threshold")
            else:
                print("REGRESSION: FAILED")
                for issue in reg["issues"]:
                    print(f"  {issue}")

            output = {
                "timestamp": timestamp,
                "baseline": base_metrics,
                "current": curr_metrics,
                "regression": reg,
            }
            out_path = LOGS_DIR / "v5_regression.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
