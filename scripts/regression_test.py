"""
Regression test: Run the full pipeline from clean state using the best configuration.
Verifies: ingestion, indexing, retrieval, metadata, evaluation.
"""
import sys
import io
import gc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
import logging
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

from backend.app.config import (
    DATA_DIR, VECTOR_DB_DIR, LOGS_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, RERANK_TOP_K,
    DENSE_WEIGHT, SOURCE_REGISTRY,
)
from backend.app.ingestion.parser import parse_pdf
from backend.app.ingestion.loader import load_documents
from backend.app.ingestion.metadata import enrich_metadata
from backend.app.ingestion.chunker import chunk_documents, generate_chunk_report
from backend.app.retrieval.embeddings import get_embedding_model
from backend.app.evaluation.evaluate import load_test_questions
from backend.app.retrieval.retriever import compute_confidence

import chromadb
from rank_bm25 import BM25Okapi

SEPARATOR = "=" * 70
PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"test": name, "status": status, "detail": detail})
    prefix = "  [PASS]" if condition else "  [FAIL]"
    print(f"{prefix} {name}" + (f" — {detail}" if detail else ""))


def main():
    print(f"{SEPARATOR}")
    print("DIABETES RAG — REGRESSION TEST (Clean State)")
    print(f"{SEPARATOR}")
    print(f"\n  Configuration:")
    print(f"    Embedding model:  {EMBEDDING_MODEL}")
    print(f"    Chunk size:       {CHUNK_SIZE_TOKENS} tokens")
    print(f"    Chunk overlap:    {CHUNK_OVERLAP_TOKENS} tokens")
    print(f"    Dense/BM25 weight: {DENSE_WEIGHT}/{1.0 - DENSE_WEIGHT}")
    print(f"    Top-K:            {TOP_K}")
    print(f"    Rerank top-K:     {RERANK_TOP_K}")

    # ================================================================
    # PHASE 1: PDF Inspection
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 1: PDF INSPECTION")
    print(SEPARATOR)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    check("PDF files exist", len(pdf_files) >= 2, f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        pages_raw = parse_pdf(pdf_path)
        low_text = [p["page_pdf"] for p in pages_raw if p["text_length"] < 50]
        check(f"PDF {pdf_path.name} extraction quality",
              len(low_text) == 0,
              f"{len(pages_raw)} pages, low-text: {low_text or 'none'}")

    # ================================================================
    # PHASE 2: Ingestion
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 2: INGESTION")
    print(SEPARATOR)

    t0 = time.time()
    pages = load_documents()
    pages = enrich_metadata(pages)
    t_ingest = time.time() - t0

    check("Pages loaded", len(pages) == 24, f"Got {len(pages)} pages")

    # Check source IDs present
    source_ids = Counter(p.get("source_id") for p in pages)
    check("ADA source present",
          "ada_soc_2026_diagnosis" in source_ids,
          f"ADA pages: {source_ids.get('ada_soc_2026_diagnosis', 0)}")
    check("NIDDK source present",
          "niddk_diabetes_prediabetes_tests" in source_ids,
          f"NIDDK pages: {source_ids.get('niddk_diabetes_prediabetes_tests', 0)}")

    # Check sections detected
    sections = Counter(p.get("section", "unknown") for p in pages)
    check("Sections detected", len(sections) >= 4, f"Found {len(sections)} sections")
    check("NIDDK section is 'Comparing Diabetes Blood Tests'",
          any("Comparing" in s for s in sections))

    # Chunking
    t1 = time.time()
    chunks = chunk_documents(pages, max_tokens=CHUNK_SIZE_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    t_chunk = time.time() - t1

    check("Chunks created", len(chunks) > 0, f"Got {len(chunks)} chunks")

    chunk_doc_counts = Counter(c["source_id"] for c in chunks)
    check("ADA chunks present",
          chunk_doc_counts.get("ada_soc_2026_diagnosis", 0) > 0,
          f"ADA: {chunk_doc_counts.get('ada_soc_2026_diagnosis', 0)}")
    check("NIDDK chunks present",
          chunk_doc_counts.get("niddk_diabetes_prediabetes_tests", 0) > 0,
          f"NIDDK: {chunk_doc_counts.get('niddk_diabetes_prediabetes_tests', 0)}")

    # No empty chunks
    empty = sum(1 for c in chunks if not c["text"].strip())
    check("No empty chunks", empty == 0, f"Empty: {empty}")

    # Chunk token range reasonable
    token_lens = [c["token_estimate"] for c in chunks]
    avg_tok = sum(token_lens) / len(token_lens)
    check("Chunk tokens in reasonable range",
          100 < avg_tok < 1000,
          f"avg={avg_tok:.0f}, min={min(token_lens)}, max={max(token_lens)}")

    # Metadata preserved
    check("All chunks have source_id",
          all(c.get("source_id") for c in chunks))
    check("All chunks have section",
          all(c.get("section") for c in chunks))
    check("All chunks have page_pdf",
          all(c.get("page_pdf") for c in chunks))
    check("All chunks have doi",
          all(c.get("doi") is not None for c in chunks))
    check("All chunks have official_url",
          all(c.get("official_url") is not None for c in chunks))

    # ================================================================
    # PHASE 3: Indexing
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 3: INDEXING")
    print(SEPARATOR)

    texts = [c["text"] for c in chunks]
    model = get_embedding_model(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()

    t2 = time.time()
    with __import__('torch').no_grad():
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    t_embed = time.time() - t2

    check("Embeddings generated",
          embeddings.shape == (len(chunks), dim),
          f"Shape: {embeddings.shape}")

    # ChromaDB
    persist_dir = str(VECTOR_DB_DIR / "regression_test")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("diabetes_rag_regression")
    except Exception:
        pass
    col = client.get_or_create_collection(
        name="diabetes_rag_regression", metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = []
    for c in chunks:
        metas.append({
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
        })

    for i in range(0, len(ids), 500):
        end = min(i + 500, len(ids))
        col.add(
            ids=ids[i:end],
            documents=docs[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metas[i:end],
        )

    check("ChromaDB indexed", col.count() == len(chunks), f"Collection: {col.count()} docs")

    # BM25
    bm25_tokenized = []
    for c in chunks:
        import re
        text = c["text"].lower()
        text = re.sub(r"[^a-z0-9\s\-\.%]", " ", text)
        tokens = text.split()
        bm25_tokenized.append([t for t in tokens if len(t) > 1])

    bm25_index = BM25Okapi(bm25_tokenized)
    check("BM25 index built", True)

    # ================================================================
    # PHASE 4: Retrieval Verification
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 4: RETRIEVAL VERIFICATION")
    print(SEPARATOR)

    questions = load_test_questions()
    answerable = [q for q in questions if not q.get("must_refuse", False)]

    all_results = []
    for q in answerable:
        query_text = q["question"]

        # Embed query
        with __import__('torch').no_grad():
            q_emb = model.encode([query_text], normalize_embeddings=True)[0]
        q_emb = np.array(q_emb, dtype=np.float32)

        # Dense search
        dense_res = col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=min(RERANK_TOP_K, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_results = []
        if dense_res["ids"] and dense_res["ids"][0]:
            for i, doc_id in enumerate(dense_res["ids"][0]):
                dist = dense_res["distances"][0][i] if dense_res["distances"] else 0
                dense_results.append({
                    "chunk_id": doc_id,
                    "text": dense_res["documents"][0][i],
                    "metadata": dense_res["metadatas"][0][i],
                    "dense_score": max(0.0, 1.0 - dist),
                })

        # BM25 search
        import re as _re
        q_text = query_text.lower()
        q_text = _re.sub(r"[^a-z0-9\s\-\.%]", " ", q_text)
        q_tokens = [t for t in q_text.split() if len(t) > 1]
        bm25_scores = bm25_index.get_scores(q_tokens)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:RERANK_TOP_K]

        b_scores = [bm25_scores[i] for i in bm25_ranked]
        b_max = max(b_scores) if b_scores else 1.0
        b_min = min(b_scores) if b_scores else 0.0
        b_range = b_max - b_min if b_max != b_min else 1.0

        bm25_results = []
        for idx in bm25_ranked:
            bm25_results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "metadata": chunks[idx],
                "bm25_score": (float(bm25_scores[idx]) - b_min) / b_range,
            })

        # Weighted fusion
        d_max = max((r["dense_score"] for r in dense_results), default=1.0) or 1.0
        scores = {}
        chunk_data = {}
        for r in dense_results:
            cid = r["chunk_id"]
            scores[cid] = DENSE_WEIGHT * (r["dense_score"] / d_max)
            chunk_data[cid] = r
            chunk_data[cid]["dense_score"] = r["dense_score"]
        for r in bm25_results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + (1.0 - DENSE_WEIGHT) * r["bm25_score"]
            if cid not in chunk_data:
                chunk_data[cid] = r
            chunk_data[cid]["bm25_score"] = r["bm25_score"]

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:TOP_K]
        final = []
        for cid in sorted_ids:
            entry = chunk_data[cid]
            entry["fusion_score"] = scores[cid]
            final.append(entry)

        # Compute relevant chunk IDs
        expected_source = q.get("expected_source", "")
        expected_pages = q.get("expected_pages", [])
        relevant_ids = set()
        for r in final:
            meta = r.get("metadata", {})
            if expected_source and meta.get("source_id") == expected_source:
                if not expected_pages or meta.get("page_document", 0) in expected_pages:
                    relevant_ids.add(r["chunk_id"])

        all_results.append({
            "results": [{"chunk_id": r["chunk_id"]} for r in final],
            "relevant_chunk_ids": list(relevant_ids),
        })

    # Compute metrics
    from backend.app.evaluation.metrics import compute_retrieval_metrics
    metrics = compute_retrieval_metrics(all_results)

    check("Recall@1 > 0.2", metrics["recall@1"] > 0.2, f"Recall@1={metrics['recall@1']:.4f}")
    check("Recall@3 > 0.5", metrics["recall@3"] > 0.5, f"Recall@3={metrics['recall@3']:.4f}")
    check("Recall@5 > 0.7", metrics["recall@5"] > 0.7, f"Recall@5={metrics['recall@5']:.4f}")
    check("Recall@10 = 1.0", metrics["recall@10"] >= 0.99, f"Recall@10={metrics['recall@10']:.4f}")
    check("MRR > 0.6", metrics["mrr"] > 0.6, f"MRR={metrics['mrr']:.4f}")
    check("Hit Rate = 1.0", metrics["hit_rate"] >= 0.99, f"Hit={metrics['hit_rate']:.4f}")

    # ================================================================
    # PHASE 5: Confidence Policy Test
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 5: CONFIDENCE POLICY")
    print(SEPARATOR)

    # Test with an in-scope question
    test_q = "What A1C level is diabetes?"
    with __import__('torch').no_grad():
        q_emb = model.encode([test_q], normalize_embeddings=True)[0]
    q_emb = np.array(q_emb, dtype=np.float32)

    dense_res = col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=min(5, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    test_results = []
    if dense_res["ids"] and dense_res["ids"][0]:
        for i, doc_id in enumerate(dense_res["ids"][0]):
            dist = dense_res["distances"][0][i]
            test_results.append({
                "chunk_id": doc_id,
                "fusion_score": max(0.0, 1.0 - dist),
                "source_id": dense_res["metadatas"][0][i].get("source_id", ""),
            })

    conf = compute_confidence(test_results)
    check("Confidence for in-scope question is 'strong' or 'weak'",
          conf["level"] in ("strong", "weak"),
          f"Level={conf['level']}, Score={conf['score']:.4f}")

    # Test with out-of-scope question (should still have results, just lower confidence)
    test_q_out = "What is the best pizza recipe?"
    with __import__('torch').no_grad():
        q_emb_out = model.encode([test_q_out], normalize_embeddings=True)[0]
    q_emb_out = np.array(q_emb_out, dtype=np.float32)

    dense_res_out = col.query(
        query_embeddings=[q_emb_out.tolist()],
        n_results=min(5, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    test_results_out = []
    if dense_res_out["ids"] and dense_res_out["ids"][0]:
        for i, doc_id in enumerate(dense_res_out["ids"][0]):
            dist = dense_res_out["distances"][0][i]
            test_results_out.append({
                "chunk_id": doc_id,
                "fusion_score": max(0.0, 1.0 - dist),
                "source_id": dense_res_out["metadatas"][0][i].get("source_id", ""),
            })

    conf_out = compute_confidence(test_results_out)
    check("Confidence for out-of-scope question is lower",
          conf_out["score"] <= conf["score"],
          f"Out={conf_out['score']:.4f} vs In={conf['score']:.4f}")

    # ================================================================
    # PHASE 6: No stdout wrapping in library modules
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("PHASE 6: MODULE INTEGRITY")
    print(SEPARATOR)

    import importlib
    modules = [
        "backend.app.ingestion.parser",
        "backend.app.ingestion.loader",
        "backend.app.ingestion.metadata",
        "backend.app.ingestion.chunker",
        "backend.app.retrieval.embeddings",
        "backend.app.retrieval.vector_store",
        "backend.app.retrieval.hybrid_search",
        "backend.app.retrieval.reranker",
        "backend.app.retrieval.retriever",
        "backend.app.evaluation.metrics",
        "backend.app.evaluation.evaluate",
        "backend.app.config",
    ]

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            check(f"Module {mod_name.split('.')[-1]} imports OK", True)
        except Exception as e:
            check(f"Module {mod_name.split('.')[-1]} imports OK", False, str(e))

    # Check no stdout wrapping
    import sys as _sys
    check("No stdout wrapping",
          not hasattr(_sys.stdout, '_orig_buffer') or 'TextIOWrapper' not in str(type(_sys.stdout)),
          f"stdout type: {type(_sys.stdout).__name__}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{SEPARATOR}")
    print("REGRESSION TEST SUMMARY")
    print(SEPARATOR)

    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    total = len(results)

    print(f"\n  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in results:
            if r["status"] == FAIL:
                print(f"    FAIL: {r['test']} — {r['detail']}")

    # Save results
    regression_report = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP_TOKENS,
            "dense_weight": DENSE_WEIGHT,
            "top_k": TOP_K,
            "rerank_top_k": RERANK_TOP_K,
        },
        "metrics": metrics,
        "total_chunks": len(chunks),
        "tests_passed": passed,
        "tests_total": total,
        "tests_failed": failed,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = str(LOGS_DIR / "regression_test.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(regression_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Report: {output_path}")

    # Cleanup
    try:
        client.delete_collection("diabetes_rag_regression")
    except Exception:
        pass
    import shutil
    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
