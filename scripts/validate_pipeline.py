"""End-to-end validation of the Diabetes RAG pipeline.
Runs: PDF inspection -> ingestion -> indexing -> retrieval evaluation.
Reports ALL real results. No fabrication.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import time
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.WARNING)

from backend.app.config import (
    DATA_DIR, VECTOR_DB_DIR, LOGS_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, RERANK_TOP_K,
)
from backend.app.ingestion.parser import parse_pdf
from backend.app.ingestion.loader import load_documents
from backend.app.ingestion.metadata import enrich_metadata
from backend.app.ingestion.chunker import chunk_documents, generate_chunk_report
from backend.app.retrieval.embeddings import embed_texts, embed_query, get_model_dimension
from backend.app.retrieval.vector_store import reset_collection, add_chunks, query_dense
from backend.app.retrieval.hybrid_search import build_bm25_index, search_bm25
from backend.app.evaluation.evaluate import load_test_questions

SEPARATOR = "=" * 70


def inspect_pdfs():
    """Step 1: Inspect both PDFs."""
    print(f"\n{SEPARATOR}")
    print("STEP 1: PDF INSPECTION")
    print(SEPARATOR)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    inspection = {}

    for pdf_path in pdf_files:
        print(f"\n  File: {pdf_path.name}")
        print(f"  Size: {pdf_path.stat().st_size:,} bytes")

        pages = parse_pdf(pdf_path)
        text_lengths = [p["text_length"] for p in pages]
        pages_with_images = [p["page_pdf"] for p in pages if p["has_images"]]
        pages_with_tables = [p["page_pdf"] for p in pages if p["tables_detected"]]
        low_text = [p["page_pdf"] for p in pages if p["text_length"] < 50]

        print(f"  Pages: {len(pages)}")
        print(f"  Text length per page: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths)/len(text_lengths):.0f}")
        print(f"  Pages with images: {pages_with_images or 'none'}")
        print(f"  Pages with tables: {pages_with_tables or 'none'}")
        print(f"  Low-text pages (<50 chars): {low_text or 'none'}")
        print(f"  Extraction quality: {'GOOD' if not low_text else 'POOR - needs attention'}")

        for p in pages[:3]:
            print(f"\n  Page {p['page_pdf']} preview ({p['text_length']} chars):")
            preview = p["text"][:200].replace("\n", " ")
            print(f"    \"{preview}...\"")

        inspection[pdf_path.name] = {
            "pages": len(pages),
            "text_lengths": text_lengths,
            "pages_with_images": pages_with_images,
            "pages_with_tables": pages_with_tables,
            "low_text_pages": low_text,
        }

    return inspection


def run_ingestion():
    """Step 2: Load, enrich, and chunk."""
    print(f"\n{SEPARATOR}")
    print("STEP 2: INGESTION")
    print(SEPARATOR)

    t0 = time.time()
    pages = load_documents()
    t_load = time.time() - t0

    print(f"\n  Pages loaded: {len(pages)}")
    print(f"  Load time: {t_load:.2f}s")

    pages = enrich_metadata(pages)

    sections = Counter(p.get("section", "unknown") for p in pages)
    print(f"\n  Sections detected:")
    for sec, count in sections.most_common():
        print(f"    [{count:2d} pages] {sec}")

    t1 = time.time()
    chunks = chunk_documents(pages, max_tokens=CHUNK_SIZE_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    t_chunk = time.time() - t1

    report = generate_chunk_report(chunks, str(LOGS_DIR / "chunk_report.json"))

    print(f"\n  Chunks created: {report['total_chunks']}")
    print(f"  Chunk time: {t_chunk:.2f}s")
    print(f"  Chunks per document: {report['chunks_per_document']}")
    print(f"  Pages per document: {report['pages_per_document']}")
    print(f"  Avg tokens/chunk: {report['avg_token_length']:.0f}")
    print(f"  Min tokens/chunk: {report['min_token_length']}")
    print(f"  Max tokens/chunk: {report['max_token_length']}")
    print(f"  Empty chunks: {report['empty_chunks']}")
    print(f"  Table chunks: {report['table_chunks']}")
    print(f"  Sections: {len(report['sections'])}")

    print(f"\n  Section distribution across chunks:")
    for sec, count in sorted(report["sections"].items(), key=lambda x: -x[1]):
        print(f"    [{count:3d} chunks] {sec}")

    return chunks, report


def run_indexing(chunks):
    """Step 3: Build embeddings + vector store + BM25."""
    print(f"\n{SEPARATOR}")
    print("STEP 3: INDEXING")
    print(SEPARATOR)

    texts = [c["text"] for c in chunks]

    print(f"\n  Embedding model: {EMBEDDING_MODEL}")
    dim = get_model_dimension(EMBEDDING_MODEL)
    print(f"  Embedding dimension: {dim}")
    print(f"  Texts to embed: {len(texts)}")

    t0 = time.time()
    embeddings = embed_texts(texts, model_name=EMBEDDING_MODEL)
    t_embed = time.time() - t0
    print(f"  Embedding time: {t_embed:.2f}s")
    print(f"  Embeddings shape: {embeddings.shape}")

    t1 = time.time()
    collection = reset_collection()
    add_chunks(chunks, embeddings)
    t_vec = time.time() - t1
    print(f"\n  Vector store built: {collection.count()} documents")
    print(f"  Vector store time: {t_vec:.2f}s")

    t2 = time.time()
    bm25_path = str(VECTOR_DB_DIR / "bm25_index.pkl")
    build_bm25_index(chunks, persist_path=bm25_path)
    t_bm25 = time.time() - t2
    print(f"  BM25 index built: {len(chunks)} chunks")
    print(f"  BM25 time: {t_bm25:.2f}s")

    print(f"\n  Total indexing time: {t_embed + t_vec + t_bm25:.2f}s")

    return collection


def run_retrieval_validation(chunks):
    """Step 4: Validate retrieval with test questions."""
    print(f"\n{SEPARATOR}")
    print("STEP 4: RETRIEVAL VALIDATION")
    print(SEPARATOR)

    questions = load_test_questions()
    print(f"\n  Test questions: {len(questions)}")

    chunk_source_pages = {}
    for c in chunks:
        key = (c["source_id"], c["page_pdf"])
        chunk_source_pages.setdefault(key, []).append(c["chunk_id"])

    results_log = []

    for i, q in enumerate(questions):
        query_text = q["question"]
        expected_source = q.get("expected_source", "")
        expected_pages = q.get("expected_pages", [])
        must_refuse = q.get("must_refuse", False)

        t0 = time.time()
        query_emb = embed_query(query_text, model_name=EMBEDDING_MODEL)
        dense_results = query_dense(query_emb, top_k=RERANK_TOP_K)
        bm25_results = search_bm25(query_text, top_k=RERANK_TOP_K)
        elapsed = time.time() - t0

        all_retrieved = []
        seen_ids = set()
        for r in dense_results:
            if r["chunk_id"] not in seen_ids:
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])
        for r in bm25_results:
            if r["chunk_id"] not in seen_ids:
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])

        scores = {}
        for r in dense_results:
            scores.setdefault(r["chunk_id"], {"dense": 0, "bm25": 0})
            scores[r["chunk_id"]]["dense"] = r["dense_score"]
        for r in bm25_results:
            scores.setdefault(r["chunk_id"], {"dense": 0, "bm25": 0})
            scores[r["chunk_id"]]["bm25"] = r["bm25_score"]

        K_RRF = 60
        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            scores[cid]["fusion"] = scores[cid].get("fusion", 0) + 1.0 / (K_RRF + rank + 1)
        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            scores[cid]["fusion"] = scores[cid].get("fusion", 0) + 1.0 / (K_RRF + rank + 1)

        all_retrieved.sort(key=lambda x: scores.get(x["chunk_id"], {}).get("fusion", 0), reverse=True)

        relevant_ids = set()
        for chunk in all_retrieved:
            cs = chunk.get("metadata", {}).get("source_id", "")
            cp = chunk.get("metadata", {}).get("page_document", 0)
            if expected_source and cs == expected_source:
                if not expected_pages or cp in expected_pages:
                    relevant_ids.add(chunk["chunk_id"])

        top3 = all_retrieved[:3]
        top_source_match = False
        top_page_match = False
        if top3:
            top_meta = top3[0].get("metadata", {})
            top_source_match = (expected_source == "") or (top_meta.get("source_id", "") == expected_source)
            top_page_match = (not expected_pages) or (top_meta.get("page_document", 0) in expected_pages)

        any_source_match = False
        any_page_match = False
        for chunk in all_retrieved[:TOP_K]:
            meta = chunk.get("metadata", {})
            if expected_source and meta.get("source_id", "") == expected_source:
                any_source_match = True
            if expected_pages and meta.get("page_document", 0) in expected_pages:
                any_page_match = True

        all_results_for_entry = []
        for r in all_retrieved:
            meta = r.get("metadata", {})
            all_results_for_entry.append({
                "chunk_id": r["chunk_id"],
                "source_id": meta.get("source_id", ""),
                "page_pdf": meta.get("page_pdf", 0),
                "page_document": meta.get("page_document", 0),
                "section": meta.get("section", ""),
                "dense_score": round(scores.get(r["chunk_id"], {}).get("dense", 0), 4),
                "bm25_score": round(scores.get(r["chunk_id"], {}).get("bm25", 0), 4),
                "fusion_score": round(scores.get(r["chunk_id"], {}).get("fusion", 0), 6),
                "text_preview": r["text"][:150].replace("\n", " "),
            })

        entry = {
            "question": query_text,
            "expected_source": expected_source,
            "expected_pages": expected_pages,
            "must_refuse": must_refuse,
            "top_source_match": top_source_match,
            "top_page_match": top_page_match,
            "any_source_in_top_k": any_source_match,
            "any_page_in_top_k": any_page_match,
            "num_retrieved": len(all_retrieved),
            "num_relevant": len(relevant_ids),
            "retrieval_ms": round(elapsed * 1000, 1),
            "results": all_results_for_entry,
            "top3": all_results_for_entry[:3],
        }

        results_log.append(entry)

        status = "OK" if (top_source_match and top_page_match) else "MISS"
        if must_refuse:
            status = "OK" if not any_source_match else "SHOULD_REFUSE_BUT_FOUND"

        q_short = query_text[:65]
        print(f"\n  [{i+1:2d}] {status:6s} | {q_short}")
        if top3:
            t = top3[0]
            meta = t.get("metadata", {})
            print(f"       Top hit: {meta.get('source_id','?')} p{meta.get('page_document','?')} "
                  f"section={meta.get('section','?')[:40]} "
                  f"dense={t['dense_score']:.4f}")
            print(f"       Evidence: \"{t['text'][:100].replace(chr(10),' ')}...\"")

    return results_log


def compute_metrics(results_log, questions):
    """Step 5: Compute retrieval metrics."""
    print(f"\n{SEPARATOR}")
    print("STEP 5: RETRIEVAL METRICS")
    print(SEPARATOR)

    source_correct_top1 = sum(1 for r in results_log if r["top_source_match"])
    source_correct_topk = sum(1 for r in results_log if r["any_source_in_top_k"])
    page_correct_top1 = sum(1 for r in results_log if r["top_page_match"])
    page_correct_topk = sum(1 for r in results_log if r["any_page_in_top_k"])

    total = len(results_log)
    answerable = [r for r in results_log if not r["must_refuse"]]
    unanswerable = [r for r in results_log if r["must_refuse"]]

    print(f"\n  Total questions: {total}")
    print(f"  Answerable: {len(answerable)}")
    print(f"  Unanswerable (must-refuse): {len(unanswerable)}")

    print(f"\n  Source accuracy (answerable only):")
    print(f"    Top-1 match: {source_correct_top1}/{len(answerable)} = {source_correct_top1/len(answerable):.1%}")
    print(f"    Any in top-{TOP_K}: {source_correct_topk}/{len(answerable)} = {source_correct_topk/len(answerable):.1%}")

    print(f"\n  Page accuracy (answerable only):")
    print(f"    Top-1 match: {page_correct_top1}/{len(answerable)} = {page_correct_top1/len(answerable):.1%}")
    print(f"    Any in top-{TOP_K}: {page_correct_topk}/{len(answerable)} = {page_correct_topk/len(answerable):.1%}")

    recall_scores = {1: [], 3: [], 5: [], 10: []}
    mrr_scores = []
    hit_scores = []

    for entry in results_log:
        if entry["must_refuse"] or not entry["expected_source"]:
            continue

        expected_source = entry["expected_source"]
        expected_pages = entry["expected_pages"]

        relevant_chunk_ids = set()
        all_retrieved_ids = []
        for r in entry.get("results", []):
            all_retrieved_ids.append(r["chunk_id"])
            if r["source_id"] == expected_source:
                if not expected_pages or r["page_document"] in expected_pages:
                    relevant_chunk_ids.add(r["chunk_id"])

        for k in [1, 3, 5, 10]:
            retrieved_at_k = set(all_retrieved_ids[:k])
            if relevant_chunk_ids:
                recall_scores[k].append(len(retrieved_at_k & relevant_chunk_ids) / len(relevant_chunk_ids))
            else:
                recall_scores[k].append(0.0)

        found = False
        for idx, cid in enumerate(all_retrieved_ids):
            if cid in relevant_chunk_ids:
                mrr_scores.append(1.0 / (idx + 1))
                found = True
                break
        if not found:
            mrr_scores.append(0.0)

        hit_scores.append(1.0 if found else 0.0)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "recall@1": round(avg(recall_scores[1]), 4),
        "recall@3": round(avg(recall_scores[3]), 4),
        "recall@5": round(avg(recall_scores[5]), 4),
        "recall@10": round(avg(recall_scores[10]), 4),
        "mrr": round(avg(mrr_scores), 4),
        "hit_rate": round(avg(hit_scores), 4),
        "source_accuracy_top1": round(source_correct_top1 / len(answerable), 4) if answerable else 0,
        "source_accuracy_topk": round(source_correct_topk / len(answerable), 4) if answerable else 0,
        "page_accuracy_top1": round(page_correct_top1 / len(answerable), 4) if answerable else 0,
        "page_accuracy_topk": round(page_correct_topk / len(answerable), 4) if answerable else 0,
    }

    print(f"\n  Retrieval Metrics (answerable questions, using source+page relevance):")
    print(f"    Recall@1:  {metrics['recall@1']:.4f}")
    print(f"    Recall@3:  {metrics['recall@3']:.4f}")
    print(f"    Recall@5:  {metrics['recall@5']:.4f}")
    print(f"    Recall@10: {metrics['recall@10']:.4f}")
    print(f"    MRR:       {metrics['mrr']:.4f}")
    print(f"    Hit Rate:  {metrics['hit_rate']:.4f}")

    print(f"\n  Failed questions (answerable, top-1 source miss):")
    for r in results_log:
        if not r["must_refuse"] and r["expected_source"] and not r["top_source_match"]:
            print(f"    - \"{r['question'][:70]}\"")
            if r["top3"]:
                print(f"      Expected: {r['expected_source']} p{r['expected_pages']}")
                print(f"      Got:      {r['top3'][0]['source_id']} p{r['top3'][0]['page_document']}")

    return metrics


def main():
    print(SEPARATOR)
    print("DIABETES RAG - END-TO-END PIPELINE VALIDATION")
    print(SEPARATOR)

    inspection = inspect_pdfs()
    chunks, chunk_report = run_ingestion()
    collection = run_indexing(chunks)
    results_log = run_retrieval_validation(chunks)
    metrics = compute_metrics(results_log, load_test_questions())

    final_report = {
        "pdf_inspection": inspection,
        "chunk_report": chunk_report,
        "index": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": get_model_dimension(EMBEDDING_MODEL),
            "total_chunks_indexed": collection.count(),
        },
        "retrieval_metrics": metrics,
        "per_question_results": [
            {k: v for k, v in r.items() if k != "top3"}
            for r in results_log
        ],
        "per_question_top3": [
            {"question": r["question"][:80], "top3": r["top3"]}
            for r in results_log
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = str(LOGS_DIR / "pipeline_validation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{SEPARATOR}")
    print("PIPELINE VALIDATION COMPLETE")
    print(SEPARATOR)
    print(f"\n  Full report: {output_path}")

    return final_report


if __name__ == "__main__":
    main()
