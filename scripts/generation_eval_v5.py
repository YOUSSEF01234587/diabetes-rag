"""Phase 10: Full generation evaluation — 119 questions.
Measures: retrieval, generation, citation, safety, latency.
Requires: OPENAI_API_KEY in environment or backend/app/.env.
"""
import os, sys, json, csv, time, re, logging, datetime
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
logging.basicConfig(level=logging.WARNING)

from backend.app.retrieval.hybrid_search import load_bm25_index
from backend.app.config import VECTOR_DB_DIR, TOP_K, RERANK_TOP_K
from backend.app.retrieval.retriever import hybrid_search
from backend.app.generation.llm import generate_answer, get_client
from backend.app.generation.citation_engine import validate_answer_citations
from backend.app.generation.answer_verifier import verify_answer
from backend.app.generation.safety import check_safety
from backend.app.generation.prompt import classify_query, build_refusal_response
from backend.app.evidence.evidence_validator import EvidenceValidator
from backend.app.evidence.evidence_pack import build_evidence_pack
from backend.app.retrieval.page_sections import is_relevant_section, SECTION_PAGE_RANGES

LOGS_DIR = "D:/diabetes-rag/logs"

load_bm25_index(str(VECTOR_DB_DIR / "bm25_index.pkl"))


def load_test_set():
    path = "D:/diabetes-rag/backend/app/evaluation/test_set_v3.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_answer_keywords(answer_text, expected_keywords):
    if not answer_text or not expected_keywords:
        return 0.0
    answer_lower = answer_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords) if expected_keywords else 0.0


def check_citation_format(answer_text):
    pattern = r'\[Evidence\s+\d+\]'
    return len(re.findall(pattern, answer_text))


def check_refusal_appropriateness(answer_text, is_answerable):
    refusal_phrases = [
        "i don't have enough",
        "i'm sorry, but",
        "insufficient evidence",
        "i cannot answer",
        "don't want to guess",
        "don't have enough reliable evidence",
    ]
    is_refusal = any(phrase in answer_text.lower() for phrase in refusal_phrases)
    if is_answerable:
        return not is_refusal
    else:
        return is_refusal


def check_grounding(answer_text, evidence_pack):
    if not answer_text:
        return 0.0
    if not evidence_pack:
        return 0.0
    if hasattr(evidence_pack, "chunks"):
        evidence_text = " ".join(c.text.lower() for c in evidence_pack.chunks)
    elif isinstance(evidence_pack, dict):
        chunks = evidence_pack.get("sections", [])
        if not chunks:
            return 0.5
        evidence_text = " ".join(str(s).lower() for s in chunks)
    else:
        return 0.5
    answer_lower = answer_text.lower()
    key_terms = ["a1c", "fasting", "glucose", "diabetes", "prediabetes", "ogtt", "screening",
                 "diagnostic", "classification", "threshold", "plasma"]
    evidence_terms = {t for t in key_terms if t in evidence_text}
    answer_terms = {t for t in key_terms if t in answer_lower}
    if not evidence_terms:
        return 0.5
    overlap = len(evidence_terms & answer_terms)
    return overlap / len(evidence_terms)


def check_numerical_accuracy(answer_text, evidence_pack):
    if not evidence_pack or not evidence_pack.chunks:
        return True, []
    evidence_text = " ".join(c.text for c in evidence_pack.chunks)
    threshold_pattern = r'(\d+\.?\d*)\s*(mg/dL|mmol/L|%|mmol/mol)'
    answer_vals = set(re.findall(threshold_pattern, answer_text, re.IGNORECASE))
    evidence_vals = set(re.findall(threshold_pattern, evidence_text, re.IGNORECASE))
    issues = []
    for val, unit in answer_vals:
        found = False
        for ev_val, ev_unit in evidence_vals:
            if ev_unit.lower() == unit.lower() and abs(float(ev_val) - float(val)) < 0.01:
                found = True
                break
        if not found:
            issues.append(f"{val} {unit}")
    return len(issues) == 0, issues


def run_generation_eval(evidence_k=5):
    questions = load_test_set()
    print(f"Loaded {len(questions)} questions | Evidence K={evidence_k}")

    # Verify LLM is accessible
    try:
        client = get_client()
        print(f"LLM client ready: model={client.models.list.__module__}")
    except Exception as e:
        print(f"LLM CLIENT ERROR: {e}")
        print("Set OPENAI_API_KEY in backend/app/.env or as environment variable")
        return None

    results = []
    t_start = time.time()

    for i, q in enumerate(questions):
        qid = q.get("id", f"Q{i}")
        query = q["question"]
        answerable = q.get("answerable", True)
        expected_sources = q.get("expected_sources", [])
        expected_sections = q.get("expected_sections", [])
        category = q.get("category", "other")

        print(f"[{i+1}/{len(questions)}] {qid} ({category})...", end=" ", flush=True)

        # RETRIEVAL
        t0 = time.time()
        try:
            sr = hybrid_search(query, top_k=TOP_K, rerank_top_k=RERANK_TOP_K, enable_reranker=False)
            search_results = sr["results"]
        except Exception as e:
            print(f"SEARCH ERROR: {e}")
            results.append({"id": qid, "query": query, "category": category, "answerable": answerable, "search_error": str(e)})
            continue
        search_ms = (time.time() - t0) * 1000

        # RETRIEVAL METRICS
        top_source = search_results[0].get("source_id", "") if search_results else ""
        top_section = search_results[0].get("section", "") if search_results else ""
        top_page = (search_results[0].get("page_pdf", 0) or search_results[0].get("page_document", 0)) if search_results else 0
        top_score = search_results[0].get("fusion_score", 0) if search_results else 0

        r5 = False
        r10 = False
        if answerable and search_results:
            for rank, r in enumerate(search_results[:10]):
                r_sid = r.get("source_id", "")
                r_sec = r.get("section", "")
                r_page = r.get("page_pdf", 0) or r.get("page_document", 0)
                src_match = r_sid in expected_sources if expected_sources else True
                sec_match = is_relevant_section(r_sec, expected_sections, r_page) if expected_sections else True
                if src_match and sec_match:
                    if rank < 5: r5 = True
                    if rank < 10: r10 = True
                    break

        source_correct = top_source in expected_sources if expected_sources else True
        section_correct = is_relevant_section(top_section, expected_sections, top_page) if expected_sections else True

        # GENERATION
        try:
            gen_result = generate_answer(query, search_results, evidence_k=evidence_k)
        except Exception as e:
            print(f"GEN ERROR: {e}")
            results.append({
                "id": qid, "query": query, "category": category, "answerable": answerable,
                "search_error": None, "gen_error": str(e),
                "r5": r5, "r10": r10, "source_correct": source_correct, "section_correct": section_correct,
                "search_ms": round(search_ms, 1),
            })
            continue
        gen_ms = gen_result.get("timings", {}).get("llm_ms", 0)
        total_ms = search_ms + gen_ms

        answer_text = gen_result.get("answer", "")
        is_refused = gen_result.get("refused", False)
        confidence = gen_result.get("confidence", "none")
        grounding_score = gen_result.get("grounding_score", 0.0)
        citations = gen_result.get("citations", [])

        # GENERATION METRICS
        citation_count = check_citation_format(answer_text)
        refusal_correct = check_refusal_appropriateness(answer_text, answerable)
        grounding_check = check_grounding(answer_text, (gen_result.get("evidence_validation") or {}).get("evidence_summary") or {})
        num_correct, num_issues = check_numerical_accuracy(answer_text, (gen_result.get("evidence_validation") or {}).get("evidence_summary"))

        verification = gen_result.get("verification", {})
        verification_passed = verification.get("passed", False)
        safety = gen_result.get("safety", {})

        result = {
            "id": qid,
            "query": query,
            "category": category,
            "answerable": answerable,
            "r5": r5,
            "r10": r10,
            "source_correct": source_correct,
            "section_correct": section_correct,
            "refused": is_refused,
            "refusal_correct": refusal_correct,
            "confidence": confidence,
            "grounding_score": grounding_score,
            "citation_count": citation_count,
            "verification_passed": verification_passed,
            "numerical_correct": num_correct,
            "numerical_issues": num_issues,
            "safety_risk": safety.get("risk_level", "low"),
            "refusal_reason": gen_result.get("refusal_reason"),
            "answer_length": len(answer_text),
            "search_ms": round(search_ms, 1),
            "gen_ms": round(gen_ms, 1),
            "total_ms": round(total_ms, 1),
            "answer_text": answer_text,
            "answer_preview": answer_text[:300] + "..." if len(answer_text) > 300 else answer_text,
        }
        results.append(result)

        status = "OK" if not is_refused else "REFUSED"
        print(f"{status} | conf={confidence} | cite={citation_count} | "
              f"refusal={'CORRECT' if refusal_correct else 'WRONG'} | "
              f"verify={'PASS' if verification_passed else 'FAIL'} | {total_ms:.0f}ms")

    total_time = time.time() - t_start

    # AGGREGATE
    n = len(results)
    ans_results = [r for r in results if r.get("answerable", True)]
    unans_results = [r for r in results if not r.get("answerable", True)]

    def avg(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def rate(lst, key):
        vals = [r for r in lst if key in r]
        return round(sum(1 for r in vals if r[key]) / len(vals), 4) if vals else 0.0

    agg = {
        "total": n,
        "answerable": len(ans_results),
        "unanswerable": len(unans_results),
        "retrieval": {
            "recall@5_answerable": rate(ans_results, "r5"),
            "recall@10_answerable": rate(ans_results, "r10"),
            "source_accuracy": rate(results, "source_correct"),
            "section_accuracy": rate(results, "section_correct"),
        },
        "generation": {
            "answerable_not_refused": round(sum(1 for r in ans_results if not r.get("refused", True)) / len(ans_results), 4) if ans_results else 0,
            "unanswerable_refused": rate(unans_results, "refused"),
            "refusal_accuracy": rate(results, "refusal_correct"),
            "avg_confidence_score": avg(results, "grounding_score"),
            "avg_citations": avg(results, "citation_count"),
            "verification_pass_rate": rate(results, "verification_passed"),
        },
        "safety": {
            "high_risk_queries": sum(1 for r in results if r.get("safety_risk") == "high"),
            "refusal_for_emergency": 0,
        },
        "latency": {
            "avg_search_ms": avg(results, "search_ms"),
            "avg_gen_ms": avg(results, "gen_ms"),
            "avg_total_ms": avg(results, "total_ms"),
            "wall_time_s": round(total_time, 1),
        },
    }

    # BY CATEGORY
    cats = {}
    for r in results:
        cat = r.get("category", "other")
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r)

    cat_agg = {}
    for cat, cat_results in cats.items():
        ans_in_cat = [r for r in cat_results if r.get("answerable", True)]
        cat_agg[cat] = {
            "count": len(cat_results),
            "r5_answerable": rate(ans_in_cat, "r5"),
            "refusal_accuracy": rate(cat_results, "refusal_correct"),
            "avg_citations": avg(cat_results, "citation_count"),
            "refused_rate": rate(cat_results, "refused"),
        }

    output = {
        "experiment": "generation_evaluation_v5",
        "timestamp": datetime.datetime.now().isoformat(),
        "evidence_k": evidence_k,
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "aggregates": agg,
        "by_category": cat_agg,
        "detailed_results": results,
    }

    # FAILURE ANALYSIS
    failures = []
    for r in results:
        reasons = []
        if r.get("answerable", True) and r.get("refused"):
            reasons.append("answerable_refused")
        if not r.get("answerable", True) and not r.get("refused"):
            reasons.append("unanswerable_not_refused")
        if not r.get("refusal_correct", True):
            reasons.append("wrong_refusal")
        if r.get("numerical_issues"):
            reasons.append("numerical_mismatch")
        if not r.get("verification_passed", True) and not r.get("refused", True):
            reasons.append("verification_failed")
        if not r.get("source_correct", True):
            reasons.append("wrong_source")
        if not r.get("section_correct", True):
            reasons.append("wrong_section")
        if reasons:
            failures.append({
                "id": r["id"],
                "query": r["query"],
                "category": r["category"],
                "answerable": r.get("answerable", True),
                "refused": r.get("refused", False),
                "failure_reasons": reasons,
                "answer_text": r.get("answer_text", ""),
            })

    # SEPARATE BREAKDOWNS
    answerable_results = [r for r in results if r.get("answerable", True)]
    must_refuse_results = [r for r in results if not r.get("answerable", True)]
    difficult_cats = ["difficult", "threshold_table", "supported_unanswerable"]
    difficult_results = [r for r in results if r.get("category") in difficult_cats]

    def category_stats(name, subset):
        if not subset:
            return {}
        ans_in = [r for r in subset if r.get("answerable", True)]
        unans_in = [r for r in subset if not r.get("answerable", True)]
        return {
            "count": len(subset),
            "answerable_count": len(ans_in),
            "unanswerable_count": len(unans_in),
            "refusal_accuracy": rate(subset, "refusal_correct"),
            "answerable_not_refused": round(sum(1 for r in ans_in if not r.get("refused")) / len(ans_in), 4) if ans_in else 0,
            "unanswerable_refused": rate(unans_in, "refused"),
            "avg_grounding_score": avg(subset, "grounding_score"),
            "avg_citations": avg(subset, "citation_count"),
            "verification_pass_rate": rate(subset, "verification_passed"),
            "avg_gen_ms": avg(subset, "gen_ms"),
            "source_accuracy": rate(subset, "source_correct"),
            "section_accuracy": rate(subset, "section_correct"),
        }

    answerable_stats = category_stats("answerable", answerable_results)
    must_refuse_stats = category_stats("must_refuse", must_refuse_results)
    difficult_stats = category_stats("difficult", difficult_results)

    # Save failures
    os.makedirs(f"{LOGS_DIR}/final", exist_ok=True)
    with open(f"{LOGS_DIR}/final/generation_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    output["failures"] = failures
    output["failure_count"] = len(failures)
    output["breakdowns"] = {
        "answerable": answerable_stats,
        "must_refuse": must_refuse_stats,
        "difficult": difficult_stats,
    }

    # Save
    with open(f"{LOGS_DIR}/final/generation_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(f"{LOGS_DIR}/final/generation_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "query", "category", "answerable", "r5", "r10", "source_correct",
            "section_correct", "refused", "refusal_correct", "confidence",
            "grounding_score", "citation_count", "verification_passed",
            "safety_risk", "search_ms", "gen_ms", "total_ms",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    # PRINT SUMMARY
    print("\n" + "=" * 80)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 80)
    print(f"Questions: {n} ({agg['answerable']} answerable, {agg['unanswerable']} unanswerable)")
    print(f"Model: {output['model']} | Evidence K: {evidence_k}")
    print()
    print("--- RETRIEVAL ---")
    for k, v in agg["retrieval"].items():
        print(f"  {k}: {v}")
    print()
    print("--- GENERATION ---")
    for k, v in agg["generation"].items():
        print(f"  {k}: {v}")
    print()
    print("--- LATENCY ---")
    for k, v in agg["latency"].items():
        print(f"  {k}: {v}")
    print()
    print("--- BY CATEGORY ---")
    print(f"{'Category':>20} | {'N':>4} | {'R@5':>6} | {'Refused%':>8} | {'Cites':>5} | {'RefAcc':>6}")
    print("-" * 70)
    for cat, ca in sorted(cat_agg.items()):
        print(f"{cat:>20} | {ca['count']:>4} | {ca['r5_answerable']:>6.4f} | "
              f"{ca['refused_rate']:>7.2%} | {ca['avg_citations']:>5.1f} | {ca['refusal_accuracy']:>6.4f}")
    print()
    print("--- SEPARATE BREAKDOWNS ---")
    for label, stats in [("Answerable", answerable_stats), ("Must-Refuse", must_refuse_stats), ("Difficult", difficult_stats)]:
        if stats:
            print(f"  {label}: N={stats['count']} RefAcc={stats['refusal_accuracy']:.4f} "
                  f"AnsNotRefused={stats.get('answerable_not_refused',0):.4f} "
                  f"VerPass={stats['verification_pass_rate']:.4f}")
    print()
    print(f"--- FAILURES: {len(failures)} of {n} ---")
    for f in failures[:20]:
        print(f"  {f['id']}: {', '.join(f['failure_reasons'])} ({f['category']})")

    print(f"\nResults saved to {LOGS_DIR}/final/generation_eval_results.json")
    print(f"CSV saved to {LOGS_DIR}/final/generation_results.csv")
    print(f"Failures saved to {LOGS_DIR}/final/generation_failures.json")

    return output


if __name__ == "__main__":
    run_generation_eval(evidence_k=5)
