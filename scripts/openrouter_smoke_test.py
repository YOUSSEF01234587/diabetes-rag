"""OpenRouter End-to-End Generation Smoke Test — Steps 1-8."""
import os, sys, json, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

report = {}

# ═══════════════════════════════════════════════════════════════
# STEP 1 — CONFIGURATION CHECK
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — CONFIGURATION CHECK")
print("=" * 60)

from backend.app.config import LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, API_KEY

key_status = "configured" if API_KEY else "missing"
print(f"Provider: {LLM_PROVIDER}")
print(f"Model:    {LLM_MODEL}")
print(f"Base URL: {LLM_BASE_URL}")
print(f"API Key:  {key_status}")

assert LLM_PROVIDER == "openrouter", f"Expected provider=openrouter, got {LLM_PROVIDER}"
assert LLM_MODEL == "openai/gpt-oss-20b:free", f"Expected model=openai/gpt-oss-20b:free, got {LLM_MODEL}"
assert LLM_BASE_URL == "https://openrouter.ai/api/v1", f"Expected base_url=https://openrouter.ai/api/v1, got {LLM_BASE_URL}"
assert API_KEY, "OPENAI_API_KEY is missing"

report["config"] = {"provider": LLM_PROVIDER, "model": LLM_MODEL, "base_url": LLM_BASE_URL, "api_key": key_status}
print("STEP 1 PASS\n")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — OPENROUTER CONNECTIVITY TEST
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2 — OPENROUTER CONNECTIVITY TEST")
print("=" * 60)

from openai import OpenAI

client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

t0 = time.time()
try:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: RAG_CONNECTION_OK"}],
        max_tokens=500,
        temperature=0.0,
    )
    latency_ms = round((time.time() - t0) * 1000, 1)
    content = resp.choices[0].message.content.strip()
    print(f"Status:   SUCCESS")
    print(f"Model:    {resp.model}")
    print(f"Response: {content}")
    print(f"Latency:  {latency_ms}ms")
    report["connectivity"] = {"success": True, "latency_ms": latency_ms, "response": content, "model_returned": resp.model}
except Exception as e:
    latency_ms = round((time.time() - t0) * 1000, 1)
    err_type = type(e).__name__
    err_msg = str(e)
    if "429" in err_msg:
        err_msg = "Rate limit / quota exceeded"
    elif "401" in err_msg:
        err_msg = "Unauthorized (check API key)"
    print(f"Status:   FAILED")
    print(f"Error:    {err_type}: {err_msg}")
    print(f"Latency:  {latency_ms}ms")
    report["connectivity"] = {"success": False, "latency_ms": latency_ms, "error_type": err_type, "error_msg": err_msg}
    print("STEP 2 FAIL — stopping")
    json.dump(report, open(LOGS / "openrouter_smoke_test.json", "w"), indent=2)
    sys.exit(1)

print("STEP 2 PASS\n")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — REAL GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3 — REAL GENERATION PIPELINE")
print("=" * 60)

from backend.app.generation.llm import reset_client, generate_answer
from backend.app.retrieval.retriever import hybrid_search
from backend.app.config import DENSE_WEIGHT, TOP_K, RERANK_TOP_K
from backend.app.retrieval.hybrid_search import load_bm25_index
from backend.app.config import VECTOR_DB_DIR

bm25_path = VECTOR_DB_DIR / "bm25_index.pkl"
if bm25_path.exists():
    load_bm25_index(str(bm25_path))
    print("BM25 index loaded")
else:
    print("WARNING: BM25 index not found")

reset_client()

test_q = "What is the diagnostic threshold for diabetes using fasting plasma glucose?"

print(f"\nQuery: \"{test_q}\"")

# Retrieval
t_r0 = time.time()
results = hybrid_search(
    test_q,
    top_k=TOP_K,
    rerank_top_k=RERANK_TOP_K,
    dense_weight=DENSE_WEIGHT,
    use_multi_query=True,
    use_query_expansion=False,
)
search_results = results["results"]
retrieval_ms = round((time.time() - t_r0) * 1000, 1)
print(f"\nRetrieval: {len(search_results)} chunks in {retrieval_ms}ms")
for i, r in enumerate(search_results[:3]):
    print(f"  [{i+1}] {r.get('chunk_id','?')[:50]}  score={r.get('fusion_score',0):.4f}  dense={r.get('dense_score',0):.4f}  src={r.get('source_id','?')}  sec={r.get('section','?')}")

# Generation
print("\nCalling generation pipeline...")
t_g0 = time.time()
gen_result = generate_answer(test_q, search_results, evidence_k=5)
generation_ms = round((time.time() - t_g0) * 1000, 1)
total_ms = round(retrieval_ms + generation_ms, 1)
print(f"Generation: {generation_ms}ms")
print(f"Total:      {total_ms}ms")

report["retrieval"] = {
    "chunks": len(search_results),
    "latency_ms": retrieval_ms,
    "top_results": [
        {"chunk_id": r["chunk_id"], "score": round(r.get("fusion_score", 0), 4), "dense_score": round(r.get("dense_score", 0), 4), "source": r.get("source_id", "")}
        for r in search_results[:5]
    ],
}
report["generation"] = {"latency_ms": generation_ms, "total_ms": total_ms}
report["pipeline"] = gen_result

print("STEP 3 DONE\n")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — VERIFY OUTPUT
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4 — VERIFY OUTPUT")
print("=" * 60)

checks = {}
answer = gen_result.get("answer", "")
checks["answer_exists"] = bool(answer)
checks["refused"] = gen_result.get("refused", False)
checks["grounded"] = gen_result.get("grounded", False)
checks["grounding_score"] = gen_result.get("grounding_score", 0)
checks["citations_count"] = len(gen_result.get("citations", []))
checks["sources_count"] = len(gen_result.get("sources", []))
checks["evidence_count"] = len(gen_result.get("evidence", []))
checks["query_type"] = gen_result.get("query_type", "unknown")
checks["confidence"] = gen_result.get("confidence", "unknown")
checks["answer_length"] = len(answer)

for k, v in checks.items():
    print(f"  {k}: {v}")

print(f"\n--- ANSWER PREVIEW ---")
print(answer[:800] if answer else "(empty)")
print("--- END ---")

report["output_checks"] = checks
print("STEP 4 DONE\n")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — VERIFY CITATION QUALITY
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5 — VERIFY CITATION QUALITY")
print("=" * 60)

citations = gen_result.get("citations", [])
evidence_list = gen_result.get("evidence", [])
verification = gen_result.get("verification", {})

print(f"  Citations generated: {len(citations)}")
print(f"  Evidence chunks in pack: {len(evidence_list)}")

if citations:
    for c in citations:
        cdict = c.to_dict() if hasattr(c, "to_dict") else c
        print(f"  Citation: [{cdict.get('evidence_index','?')}] -> {cdict.get('source_id','?')} pp.{cdict.get('page','?')}")

citation_issues = []
for c in citations:
    cdict = c.to_dict() if hasattr(c, "to_dict") else c
    eid = cdict.get("evidence_index")
    if eid is not None and eid >= len(evidence_list):
        citation_issues.append(f"Citation references nonexistent evidence index {eid}")

report["citation_verification"] = {
    "count": len(citations),
    "issues": citation_issues,
    "verification_details": verification,
}
print(f"  Citation issues: {len(citation_issues)}")
print("STEP 5 DONE\n")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — VERIFY MEDICAL SAFETY
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6 — VERIFY MEDICAL SAFETY")
print("=" * 60)

safety = gen_result.get("safety", {})
print(f"  requires_professional: {safety.get('requires_professional', False)}")
print(f"  risk_level: {safety.get('risk_level', 'unknown')}")
if safety.get("triggered_rules"):
    print(f"  triggered_rules: {safety['triggered_rules']}")

report["safety"] = safety
print("STEP 6 DONE\n")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7 — PERFORMANCE")
print("=" * 60)

perf = {
    "retrieval_latency_ms": retrieval_ms,
    "llm_latency_ms": gen_result.get("timings", {}).get("llm_ms", generation_ms),
    "total_latency_ms": total_ms,
    "retrieved_chunks": len(search_results),
    "evidence_pack_size": len(evidence_list),
    "answer_length": len(answer),
}
for k, v in perf.items():
    print(f"  {k}: {v}")

report["performance"] = perf
print("STEP 7 DONE\n")

# ═══════════════════════════════════════════════════════════════
# FINAL STATUS
# ═══════════════════════════════════════════════════════════════
all_pass = (
    report["connectivity"]["success"]
    and checks["answer_exists"]
    and checks["evidence_count"] > 0
)

status = "SMOKE_TEST_PASS" if all_pass else "SMOKE_TEST_FAIL"
report["final_status"] = status

print("=" * 60)
print(f"FINAL STATUS: {status}")
print(f"Model:        {LLM_MODEL}")
print(f"Total Latency: {total_ms}ms")
print(f"Citation:     {len(citations)} citations, {len(citation_issues)} issues")
print(f"Verification: passed={verification.get('passed', False)}")
print(f"Safety:       risk_level={safety.get('risk_level', '?')}")
print("=" * 60)

json.dump(report, open(LOGS / "openrouter_smoke_test.json", "w", encoding="utf-8"), indent=2, default=str)

md = f"""# OpenRouter Smoke Test Report

## Configuration
- Provider: {LLM_PROVIDER}
- Model: {LLM_MODEL}
- Base URL: {LLM_BASE_URL}
- API Key: {key_status}

## Connectivity
- Success: {report['connectivity']['success']}
- Latency: {report['connectivity'].get('latency_ms', '?')}ms

## Retrieval
- Chunks: {len(results)}
- Latency: {retrieval_ms}ms

## Generation
- Latency: {generation_ms}ms

## Citation Validation
- Count: {len(citations)}
- Issues: {len(citation_issues)}

## Answer Verification
- Passed: {verification.get('passed', False)}

## Safety
- Risk Level: {safety.get('risk_level', '?')}

## Final Status: {status}
"""
open(LOGS / "openrouter_smoke_test.md", "w", encoding="utf-8").write(md)
print(f"\nReports saved to logs/openrouter_smoke_test.json and logs/openrouter_smoke_test.md")
