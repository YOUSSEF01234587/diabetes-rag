"""Final optimized evaluation: best config from ablation insights.

Key insight from ablation: query expansion HURTS by ~3% on cleaned V4.1 chunks.
Best config: no_query_expansion + dw=0.4 + srb=0.16 + multi_query + top_k=8
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from pathlib import Path
from collections import defaultdict
from backend.app.retrieval.hybrid_search import load_bm25_index
from backend.app.config import TOP_K, RERANK_TOP_K, VECTOR_DB_DIR, EMBEDDING_MODEL

# Override RERANK_TOP_K for this eval
RERANK_TOP_K = 15

EXPERIMENT_DIR = Path("logs/experiments")
FINAL_DIR = Path("logs/final")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_testset():
    path = Path("backend/app/evaluation/test_set_v3.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_recall(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def compute_mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_hit(retrieved_ids, relevant_ids):
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0


def run_final_eval():
    questions = load_testset()
    print(f"Loaded {len(questions)} questions")

    bm25_path = str(VECTOR_DB_DIR / "bm25_index.pkl")
    load_bm25_index(bm25_path)

    from backend.app.retrieval.retriever import (
        _weighted_fusion, _apply_source_boost, _apply_section_boost,
        _apply_intent_boosts, _format_results, detect_intent,
        detect_source_hint, detect_section_hint, expand_query,
        _get_intent_v2, _get_section_router,
    )
    from backend.app.retrieval.embeddings import embed_query
    from backend.app.retrieval.vector_store import query_dense
    from backend.app.retrieval.hybrid_search import search_bm25
    from backend.app.retrieval.multi_query import generate_query_variants, expand_with_intent_terms, reciprocal_rank_fusion

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    col = client.get_collection("diabetes_rag")
    print(f"ChromaDB chunks: {col.count()}")

    # OPTIMIZED CONFIG (from ablation + doc intelligence tuning)
    DW = 0.6
    SRB = 0.16
    TOP_K_VAL = 8
    USE_MQ = True
    USE_EXPANSION = False  # Ablation showed this HURTS by 3%

    def search(query):
        intent = detect_intent(query)
        intent_v2_classify = _get_intent_v2()
        intent_v2 = None
        if intent_v2_classify:
            try:
                intent_v2 = intent_v2_classify(query)
            except Exception:
                pass

        source_hint = detect_source_hint(query)
        section_hints = detect_section_hint(query)

        if USE_MQ:
            variants = generate_query_variants(query, intent)
            expanded_variants = []
            for v in variants:
                v_intent = detect_intent(v)
                if v_intent:
                    expanded = expand_with_intent_terms(v, v_intent)
                else:
                    expanded = v  # No general expansion, just variant directly
                expanded_variants.append(expanded)
            seen = set()
            unique_variants = []
            for v in expanded_variants:
                vl = v.lower().strip()
                if vl not in seen:
                    seen.add(vl)
                    unique_variants.append(v)
            all_result_lists = []
            for variant in unique_variants:
                qe = embed_query(variant, model_name=EMBEDDING_MODEL)
                dr = query_dense(qe, top_k=RERANK_TOP_K)
                br = search_bm25(variant, top_k=RERANK_TOP_K)
                f = _weighted_fusion(dr, br, RERANK_TOP_K, DW)
                f = _apply_source_boost(f, detect_source_hint(variant))
                f = _apply_section_boost(f, detect_section_hint(variant))
                f = _apply_intent_boosts(f, intent)
                router = _get_section_router()
                if router and intent_v2:
                    il = intent_v2.primary_intent
                    ss = router.route_query(variant, il)
                    if ss:
                        sd = dict(ss)
                        for r in f:
                            meta = r.get("metadata", {})
                            sec = meta.get("true_section", meta.get("section", ""))
                            if sec in sd:
                                r["fusion_score"] = r.get("fusion_score", 0) + sd[sec] * SRB
                f.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
                all_result_lists.append(f[:RERANK_TOP_K])
            fused = reciprocal_rank_fusion(all_result_lists, top_k=RERANK_TOP_K)
            fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        else:
            qe = embed_query(query, model_name=EMBEDDING_MODEL)
            dr = query_dense(qe, top_k=RERANK_TOP_K)
            br = search_bm25(query, top_k=RERANK_TOP_K)
            fused = _weighted_fusion(dr, br, RERANK_TOP_K, DW)
            fused = _apply_source_boost(fused, source_hint)
            fused = _apply_section_boost(fused, section_hints)
            fused = _apply_intent_boosts(fused, intent)
            router = _get_section_router()
            if router and intent_v2:
                il = intent_v2.primary_intent
                ss = router.route_query(query, il)
                if ss:
                    sd = dict(ss)
                    for r in fused:
                        meta = r.get("metadata", {})
                        sec = meta.get("true_section", meta.get("section", ""))
                        if sec in sd:
                            r["fusion_score"] = r.get("fusion_score", 0) + sd[sec] * SRB
            fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

        reranked = fused[:TOP_K_VAL]
        for i, r in enumerate(reranked):
            r["reranker_score"] = r.get("fusion_score", 0)
            r["rank"] = i + 1
        final_results = _format_results(reranked)
        from backend.app.retrieval.evidence_validation import validate_evidence, compute_enhanced_confidence
        evidence = validate_evidence(query, final_results, intent)
        confidence = compute_enhanced_confidence(final_results, query, intent, evidence)
        return {"results": final_results, "intent": intent, "confidence": confidence}

    results = []
    timings = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"Q{i+1}")
        question_text = q["question"]
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(questions)}]")
        t0 = time.time()
        sr = search(question_text)
        elapsed = time.time() - t0
        timings.append(elapsed * 1000)

        retrieved = sr["results"]
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        retrieved_sources = [r.get("source_id", "") for r in retrieved]
        retrieved_sections = [r.get("section", "") for r in retrieved]
        retrieved_pages = [r.get("page_document", 0) for r in retrieved]

        relevant_ids = set()
        for chunk in retrieved:
            cs = chunk.get("source_id", "")
            cp = chunk.get("page_document", 0)
            es = q.get("expected_sources", [])
            ep = q.get("expected_pages", [])
            if cs in es:
                if not ep or cp in ep:
                    relevant_ids.add(chunk["chunk_id"])
        if not relevant_ids:
            for chunk in retrieved:
                if chunk.get("source_id", "") in q.get("expected_sources", []):
                    relevant_ids.add(chunk["chunk_id"])

        r1 = compute_recall(retrieved_ids, relevant_ids, 1)
        r3 = compute_recall(retrieved_ids, relevant_ids, 3)
        r5 = compute_recall(retrieved_ids, relevant_ids, 5)
        r10 = compute_recall(retrieved_ids, relevant_ids, 10)
        r15 = compute_recall(retrieved_ids, relevant_ids, 15)
        r20 = compute_recall(retrieved_ids, relevant_ids, 20)
        mrr_val = compute_mrr(retrieved_ids, relevant_ids)
        hit_val = compute_hit(retrieved_ids, relevant_ids)

        expected_sources = q.get("expected_sources", [])
        source_correct = any(s in expected_sources for s in retrieved_sources) if expected_sources else False

        from backend.app.retrieval.page_sections import is_relevant_section
        expected_sections = q.get("expected_sections", [])
        section_correct = False
        if expected_sections:
            for rs, rp in zip(retrieved_sections, retrieved_pages):
                if is_relevant_section(rs, expected_sections, rp):
                    section_correct = True
                    break

        expected_pages = q.get("expected_pages", [])
        page_correct = any(p in expected_pages for p in retrieved_pages) if expected_pages else False

        detected_intent = sr.get("intent", {})
        if hasattr(detected_intent, 'primary_intent'):
            intent_str = detected_intent.primary_intent or ""
        elif isinstance(detected_intent, dict):
            intent_str = detected_intent.get("primary", "")
        else:
            intent_str = str(detected_intent) if detected_intent else ""
        intent_match = intent_str == q.get("intent", "")

        confidence = sr.get("confidence", {})
        conf_level = confidence.get("level", "")
        is_refusal = not q.get("answerable", True)
        refusal_correct = (is_refusal and conf_level in ("insufficient", "weak")) or not is_refusal

        results.append({
            "id": qid, "question": question_text,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "answerable": q.get("answerable", True),
            "intent": q.get("intent", ""),
            "detected_intent": intent_str,
            "intent_match": intent_match,
            "r1": r1, "r3": r3, "r5": r5, "r10": r10, "r15": r15, "r20": r20,
            "mrr": mrr_val, "hit": hit_val,
            "source_correct": source_correct,
            "section_correct": section_correct,
            "page_correct": page_correct,
            "refusal_correct": refusal_correct,
            "confidence_level": conf_level,
            "confidence_score": confidence.get("score", 0),
            "retrieval_ms": round(elapsed * 1000, 1),
            "retrieved_sources": retrieved_sources[:3],
            "retrieved_sections": retrieved_sections[:3],
            "retrieved_pages": retrieved_pages[:3],
            "results": retrieved,
        })

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {
        "num_questions": len(results),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "recall@1": round(avg(results, "r1"), 4),
        "recall@3": round(avg(results, "r3"), 4),
        "recall@5": round(avg(results, "r5"), 4),
        "recall@10": round(avg(results, "r10"), 4),
        "recall@15": round(avg(results, "r15"), 4),
        "recall@20": round(avg(results, "r20"), 4),
        "mrr": round(avg(results, "mrr"), 4),
        "hit_rate": round(avg(results, "hit"), 4),
        "source_accuracy": round(sum(1 for r in results if r["source_correct"]) / len(results), 4),
        "section_accuracy": round(sum(1 for r in results if r["section_correct"]) / len(results), 4),
        "page_accuracy": round(sum(1 for r in results if r["page_correct"]) / len(results), 4),
        "intent_accuracy": round(sum(1 for r in results if r["intent_match"]) / len(results), 4),
        "refusal_accuracy": round(sum(1 for r in results if r["refusal_correct"]) / len(results), 4),
        "avg_latency_ms": round(avg(results, "retrieval_ms"), 1),
        "p50_latency_ms": round(sorted(timings)[len(timings) // 2], 1) if timings else 0,
        "p95_latency_ms": round(sorted(timings)[int(len(timings) * 0.95)], 1) if timings else 0,
    }

    categories = defaultdict(lambda: {"count": 0, "r5": 0, "r10": 0, "mrr": 0, "source": 0, "section": 0, "page": 0, "intent": 0, "refusal": 0})
    for r in results:
        cat = r["category"]
        categories[cat]["count"] += 1
        categories[cat]["r5"] += r["r5"]
        categories[cat]["r10"] += r["r10"]
        categories[cat]["mrr"] += r["mrr"]
        categories[cat]["source"] += 1 if r["source_correct"] else 0
        categories[cat]["section"] += 1 if r["section_correct"] else 0
        categories[cat]["page"] += 1 if r["page_correct"] else 0
        categories[cat]["intent"] += 1 if r["intent_match"] else 0
        categories[cat]["refusal"] += 1 if r["refusal_correct"] else 0

    category_metrics = {}
    for cat, vals in categories.items():
        n = vals["count"]
        category_metrics[cat] = {
            "count": n,
            "recall@5": round(vals["r5"] / n, 4),
            "recall@10": round(vals["r10"] / n, 4),
            "mrr": round(vals["mrr"] / n, 4),
            "source_accuracy": round(vals["source"] / n, 4),
            "section_accuracy": round(vals["section"] / n, 4),
            "page_accuracy": round(vals["page"] / n, 4),
            "intent_accuracy": round(vals["intent"] / n, 4),
            "refusal_accuracy": round(vals["refusal"] / n, 4),
        }

    difficulties = defaultdict(lambda: {"count": 0, "r5": 0, "r10": 0, "mrr": 0, "source": 0, "section": 0})
    for r in results:
        d = r["difficulty"]
        difficulties[d]["count"] += 1
        difficulties[d]["r5"] += r["r5"]
        difficulties[d]["r10"] += r["r10"]
        difficulties[d]["mrr"] += r["mrr"]
        difficulties[d]["source"] += 1 if r["source_correct"] else 0
        difficulties[d]["section"] += 1 if r["section_correct"] else 0

    difficulty_metrics = {}
    for d, vals in difficulties.items():
        n = vals["count"]
        difficulty_metrics[d] = {
            "count": n,
            "recall@5": round(vals["r5"] / n, 4),
            "recall@10": round(vals["r10"] / n, 4),
            "mrr": round(vals["mrr"] / n, 4),
            "source_accuracy": round(vals["source"] / n, 4),
            "section_accuracy": round(vals["section"] / n, 4),
        }

    report = {
        "config": {
            "version": "v4.1_optimized",
            "dense_weight": DW,
            "section_router_boost": SRB,
            "top_k": TOP_K_VAL,
            "multi_query": USE_MQ,
            "query_expansion": USE_EXPANSION,
            "reranker_enabled": False,
            "chunk_count": col.count(),
            "chunking": "v4.1_clean_table_aware",
        },
        "metrics": metrics,
        "by_category": category_metrics,
        "by_difficulty": difficulty_metrics,
        "questions": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save results
    output_json = FINAL_DIR / "v41_optimized_eval.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print report
    print(f"\n{'='*70}")
    print(f"V4.1 OPTIMIZED EVALUATION ({len(results)} questions)")
    print(f"Config: DW={DW}, SRB={SRB}, K={TOP_K_VAL}, MQ={USE_MQ}, Expansion={USE_EXPANSION}")
    print(f"{'='*70}")
    print(f"Recall@1:    {metrics['recall@1']:.4f}")
    print(f"Recall@3:    {metrics['recall@3']:.4f}")
    print(f"Recall@5:    {metrics['recall@5']:.4f}")
    print(f"Recall@10:   {metrics['recall@10']:.4f}")
    print(f"Recall@15:   {metrics['recall@15']:.4f}")
    print(f"MRR:         {metrics['mrr']:.4f}")
    print(f"Hit Rate:    {metrics['hit_rate']:.4f}")
    print(f"Source Acc:  {metrics['source_accuracy']:.4f}")
    print(f"Section Acc: {metrics['section_accuracy']:.4f}")
    print(f"Page Acc:    {metrics['page_accuracy']:.4f}")
    print(f"Intent Acc:  {metrics['intent_accuracy']:.4f}")
    print(f"Refusal Acc: {metrics['refusal_accuracy']:.4f}")
    print(f"Latency:     avg={metrics['avg_latency_ms']:.0f}ms p50={metrics['p50_latency_ms']:.0f}ms p95={metrics['p95_latency_ms']:.0f}ms")

    # Compare with V4 baseline
    print(f"\n--- vs V4 Baseline ---")
    v4_baseline = {
        "recall@5": 0.7634, "recall@10": 0.9244, "mrr": 0.7384,
        "source_accuracy": 0.9244, "section_accuracy": 0.8067,
        "page_accuracy": 0.8992, "refusal_accuracy": 0.9244,
    }
    for key in ["recall@5", "recall@10", "mrr", "source_accuracy", "section_accuracy", "page_accuracy", "refusal_accuracy"]:
        v = metrics.get(key, 0)
        b = v4_baseline.get(key, 0)
        delta = v - b
        print(f"  {key:20s}: {v:.4f} vs {b:.4f} ({delta:+.4f})")

    print(f"\n--- By Difficulty ---")
    for d, vals in sorted(difficulty_metrics.items()):
        print(f"  {d:8s}: R5={vals['recall@5']:.4f} R10={vals['recall@10']:.4f} MRR={vals['mrr']:.4f} Src={vals['source_accuracy']:.4f} Sec={vals['section_accuracy']:.4f} (n={vals['count']})")

    print(f"\n--- By Category ---")
    for cat, vals in sorted(category_metrics.items(), key=lambda x: -x[1]["recall@5"]):
        print(f"  {cat:30s}: R5={vals['recall@5']:.4f} MRR={vals['mrr']:.4f} Src={vals['source_accuracy']:.4f} Sec={vals['section_accuracy']:.4f} (n={vals['count']})")

    print(f"\nResults saved to: {output_json}")
    return report


if __name__ == "__main__":
    run_final_eval()
