"""Unified retriever combining dense, BM25, and reranking."""
import logging
import time
from typing import Optional

from ..config import (
    EMBEDDING_MODEL, TOP_K, RERANK_TOP_K,
    SIMILARITY_THRESHOLD, RERANKER_MODEL, RERANKER_ENABLED,
)
from .embeddings import embed_query
from .vector_store import query_dense
from .hybrid_search import search_bm25
from .reranker import rerank
from .query_expansion import expand_query, detect_source_hint, detect_section_hint
from .query_intent import detect_intent, get_section_boost, get_source_boost, get_topic_penalty, QueryIntent
from .multi_query import multi_query_retrieve, reciprocal_rank_fusion
from .evidence_validation import validate_evidence, compute_enhanced_confidence

logger = logging.getLogger(__name__)

# V4 enhancements (lazy-loaded)
_intent_v2_classify = None
_SectionRouter = None
_section_router_instance = None

def _get_intent_v2():
    global _intent_v2_classify
    if _intent_v2_classify is None:
        try:
            from .intent_v2 import classify
            _intent_v2_classify = classify
        except ImportError:
            pass
    return _intent_v2_classify

def _get_section_router():
    global _SectionRouter, _section_router_instance
    if _section_router_instance is None:
        try:
            from .section_router import SectionRouter
            _section_router_instance = SectionRouter()
        except ImportError:
            pass
    return _section_router_instance

# V4.1 optimized section router boost (from ablation: 0.16 best for R@5)
SECTION_ROUTER_BOOST = 0.16

# Medical-safety confidence thresholds
CONFIDENCE_STRONG = 0.6
CONFIDENCE_WEAK = 0.35

# Source boost factor
SOURCE_BOOST = 0.15

# Section boost factor
SECTION_BOOST = 0.10

# Intent-based boost factor
INTENT_SECTION_BOOST = 0.20

# Topic penalty factor
TOPIC_PENALTY = -0.10


def compute_confidence(results: list[dict]) -> dict:
    """Compute retrieval confidence for medical-safety.

    Returns a confidence assessment based on:
    - Top result score (how well the best match scores)
    - Score gap between top-1 and rest (is there a clear winner?)
    - Source diversity (do results agree on source?)
    """
    if not results:
        return {
            "level": "none",
            "score": 0.0,
            "reason": "No results retrieved",
        }

    top_score = results[0].get("fusion_score", results[0].get("dense_score", 0))
    top_source = results[0].get("source_id", "")

    if len(results) > 1:
        rest_scores = [r.get("fusion_score", r.get("dense_score", 0)) for r in results[1:5]]
        avg_rest = sum(rest_scores) / len(rest_scores) if rest_scores else 0
        gap = top_score - avg_rest
    else:
        gap = top_score

    top_sources = [r.get("source_id", "") for r in results[:5]]
    source_agreement = sum(1 for s in top_sources if s == top_source) / len(top_sources) if top_sources else 0

    confidence_score = (top_score * 0.4 + gap * 0.3 + source_agreement * 0.3)
    confidence_score = max(0.0, min(1.0, confidence_score))

    if confidence_score >= CONFIDENCE_STRONG:
        level = "strong"
        reason = "High-scoring, consistent results with clear top match"
    elif confidence_score >= CONFIDENCE_WEAK:
        level = "weak"
        reason = "Moderate confidence; results present but may not precisely answer the query"
    else:
        level = "insufficient"
        reason = "Low confidence; retrieval may not contain relevant evidence"

    return {
        "level": level,
        "score": round(confidence_score, 4),
        "reason": reason,
        "top_score": round(top_score, 4),
        "score_gap": round(gap, 4),
        "source_agreement": round(source_agreement, 4),
        "top_source": top_source,
    }


def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_model: str = EMBEDDING_MODEL,
    enable_reranker: bool = RERANKER_ENABLED,
    reranker_model: str = RERANKER_MODEL,
    dense_weight: float = None,
    use_query_expansion: bool = True,
    use_intent_boost: bool = True,
    use_multi_query: bool = False,
) -> dict:
    """Perform hybrid retrieval with dense + BM25 + reranking.

    V3 improvements:
    - Intent-aware section and source routing
    - Topic penalty for off-topic sections
    - Multi-query retrieval with RRF (optional)
    - Query expansion with medical synonyms
    - Source-aware routing (boost NIDDK when query hints at comparisons)
    - Section-aware ranking (boost relevant sections)
    """
    timings = {}

    if dense_weight is None:
        dense_weight = 0.5

    # Intent detection (V3 + V4 intent_v2)
    intent = detect_intent(query) if use_intent_boost else None
    intent_v2 = None
    v2_classify = _get_intent_v2()
    if v2_classify and use_intent_boost:
        try:
            intent_v2 = v2_classify(query)
        except Exception:
            pass
    
    # Multi-query retrieval path
    if use_multi_query:
        return _multi_query_hybrid_search(
            query, top_k, rerank_top_k, embedding_model,
            enable_reranker, reranker_model, dense_weight,
            use_query_expansion, intent, timings,
        )
    
    # Legacy source/section hints (V2)
    source_hint = detect_source_hint(query)
    section_hints = detect_section_hint(query)
    expanded_query = expand_query(query) if use_query_expansion else query

    if expanded_query != query:
        logger.debug(f"Expanded query: '{expanded_query[:80]}...'")

    t0 = time.time()
    query_embedding = embed_query(expanded_query, model_name=embedding_model)
    timings["embedding_ms"] = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    dense_results = query_dense(query_embedding, top_k=rerank_top_k)
    timings["dense_ms"] = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    bm25_results = search_bm25(expanded_query, top_k=rerank_top_k)
    timings["bm25_ms"] = round((time.time() - t2) * 1000, 1)

    fused = _weighted_fusion(dense_results, bm25_results, rerank_top_k, dense_weight)

    # Apply V2 boosts
    fused = _apply_source_boost(fused, source_hint)
    fused = _apply_section_boost(fused, section_hints)
    
    # Apply V3 intent boosts
    if intent:
        fused = _apply_intent_boosts(fused, intent)

    # V4: Apply section router boosts (gentle)
    fused = _apply_section_router_boosts(fused, query, intent_v2 if intent_v2 else intent)

    fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

    t3 = time.time()
    if enable_reranker and fused:
        reranked = rerank(query, fused, top_k=top_k, model_name=reranker_model)
    else:
        reranked = fused[:top_k]
        for i, r in enumerate(reranked):
            r["reranker_score"] = r.get("fusion_score", 0)
            r["rank"] = i + 1
    timings["rerank_ms"] = round((time.time() - t3) * 1000, 1)

    final_results = _format_results(reranked)

    # Enhanced confidence with evidence validation
    evidence = validate_evidence(query, final_results, intent)
    confidence = compute_enhanced_confidence(final_results, query, intent, evidence)

    return {
        "query": query,
        "intent": _intent_to_dict(intent),
        "results": final_results,
        "confidence": confidence,
        "timings": timings,
        "total_dense": len(dense_results),
        "total_bm25": len(bm25_results),
        "total_fused": len(fused),
        "total_reranked": len(final_results),
    }


def _multi_query_hybrid_search(
    query, top_k, rerank_top_k, embedding_model,
    enable_reranker, reranker_model, dense_weight,
    use_query_expansion, intent, timings,
):
    """Multi-query hybrid search with RRF fusion."""
    from .multi_query import generate_query_variants, expand_with_intent_terms
    
    # Generate variants
    variants = generate_query_variants(query, intent)
    expanded_variants = []
    for v in variants:
        v_intent = detect_intent(v) if intent else None
        expanded = expand_with_intent_terms(v, v_intent) if v_intent else expand_query(v)
        expanded_variants.append(expanded)
    
    # Deduplicate
    seen = set()
    unique_variants = []
    for v in expanded_variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    logger.debug(f"Multi-query: {len(unique_variants)} variants")
    
    # Search each variant
    all_result_lists = []
    for variant in unique_variants:
        source_hint = detect_source_hint(variant)
        section_hints = detect_section_hint(variant)
        expanded_query = expand_query(variant) if use_query_expansion else variant
        
        query_embedding = embed_query(expanded_query, model_name=embedding_model)
        dense_results = query_dense(query_embedding, top_k=rerank_top_k)
        bm25_results = search_bm25(expanded_query, top_k=rerank_top_k)
        
        fused = _weighted_fusion(dense_results, bm25_results, rerank_top_k, dense_weight)
        fused = _apply_source_boost(fused, source_hint)
        fused = _apply_section_boost(fused, section_hints)
        if intent:
            fused = _apply_intent_boosts(fused, intent)
        # V4: section router boost
        fused = _apply_section_router_boosts(fused, variant, intent)
        
        fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        all_result_lists.append(fused[:rerank_top_k])
    
    # RRF fusion
    fused = reciprocal_rank_fusion(all_result_lists, top_k=rerank_top_k)
    timings["multi_query_ms"] = 0  # placeholder
    
    t3 = time.time()
    if enable_reranker and fused:
        reranked = rerank(query, fused, top_k=top_k, model_name=reranker_model)
    else:
        reranked = fused[:top_k]
        for i, r in enumerate(reranked):
            r["reranker_score"] = r.get("fusion_score", 0)
            r["rank"] = i + 1
    timings["rerank_ms"] = round((time.time() - t3) * 1000, 1)
    
    final_results = _format_results(reranked)
    
    # Enhanced confidence with evidence validation
    evidence = validate_evidence(query, final_results, intent)
    confidence = compute_enhanced_confidence(final_results, query, intent, evidence)
    
    return {
        "query": query,
        "intent": _intent_to_dict(intent),
        "results": final_results,
        "confidence": confidence,
        "timings": timings,
        "total_dense": 0,
        "total_bm25": 0,
        "total_fused": len(fused),
        "total_reranked": len(final_results),
        "multi_query_variants": len(unique_variants),
    }


def _format_results(reranked: list[dict]) -> list[dict]:
    """Format reranked results into final output."""
    final_results = []
    for r in reranked:
        meta = r.get("metadata", {})
        final_results.append({
            "rank": r.get("rank", len(final_results) + 1),
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "fusion_score": round(r.get("fusion_score", 0), 4),
            "dense_score": round(r.get("dense_score", 0), 4),
            "bm25_score": round(r.get("bm25_score", 0), 4),
            "reranker_score": round(r.get("reranker_score", 0), 4),
            "source_id": meta.get("source_id", ""),
            "source_title": meta.get("source_title", ""),
            "short_title": meta.get("short_title", ""),
            "organization": meta.get("organization", ""),
            "page_pdf": meta.get("page_pdf", 0),
            "page_document": meta.get("page_document", 0),
            "section": meta.get("true_section", meta.get("section", "")),
            "subsection": meta.get("subsection", ""),
            "doi": meta.get("doi", ""),
            "official_url": meta.get("official_url", ""),
            "year": meta.get("year", 0),
            "authority": meta.get("authority", "high"),
            "has_table": meta.get("has_table", False),
        })
    return final_results


def _intent_to_dict(intent) -> dict | None:
    """Convert QueryIntent to dict for JSON serialization."""
    if not intent:
        return None
    return {
        "primary": intent.primary_intent,
        "confidence": round(intent.confidence, 2),
        "all_intents": intent.all_intents,
        "preferred_sections": intent.preferred_sections,
        "preferred_sources": intent.preferred_sources,
        "is_table_query": intent.is_table_query,
        "is_comparison": intent.is_comparison,
        "is_threshold_query": intent.is_threshold_query,
    }


def _apply_source_boost(results: list[dict], source_hint: str) -> list[dict]:
    """Boost scores for chunks matching source hint."""
    if not source_hint:
        return results

    for r in results:
        meta = r.get("metadata", {})
        source_id = meta.get("source_id", "")
        is_niddk_match = source_hint == "niddk" and "niddk" in source_id.lower()
        is_ada_match = source_hint == "ada" and "ada" in source_id.lower()

        if is_niddk_match or is_ada_match:
            r["fusion_score"] = r.get("fusion_score", 0) + SOURCE_BOOST

    return results


def _apply_section_boost(results: list[dict], section_hints: list[str]) -> list[dict]:
    """Boost scores for chunks matching section hints."""
    if not section_hints:
        return results

    for r in results:
        meta = r.get("metadata", {})
        section = meta.get("section", "")
        if section in section_hints:
            r["fusion_score"] = r.get("fusion_score", 0) + SECTION_BOOST

    return results


def _apply_intent_boosts(results: list[dict], intent: QueryIntent) -> list[dict]:
    """Apply intent-based boosts and penalties to results.
    
    - Boost chunks in preferred sections/sources
    - Apply topic penalty for off-topic sections
    """
    for r in results:
        meta = r.get("metadata", {})
        section = meta.get("section", "")
        source_id = meta.get("source_id", "")
        
        # Apply section boost from intent
        section_boost = get_section_boost(section, intent)
        if section_boost > 0:
            r["fusion_score"] = r.get("fusion_score", 0) + section_boost
        
        # Apply source boost from intent
        source_boost = get_source_boost(source_id, intent)
        if source_boost > 0:
            r["fusion_score"] = r.get("fusion_score", 0) + source_boost
        
        # Apply topic penalty for off-topic sections
        topic_penalty = get_topic_penalty(section, intent)
        if topic_penalty < 0:
            r["fusion_score"] = r.get("fusion_score", 0) + topic_penalty

    return results


def _apply_section_router_boosts(results: list[dict], query: str, intent) -> list[dict]:
    """Apply V4 section router boosts (gentle, on top of V3 boosts).
    
    Uses the SectionRouter to score candidate sections for the query,
    then boosts chunks whose section matches top router candidates.
    """
    router = _get_section_router()
    if not router:
        return results

    intent_label = intent.primary_intent if intent else None
    section_scores = router.route_query(query, intent_label)
    if not section_scores:
        return results

    section_dict = dict(section_scores)
    for r in results:
        meta = r.get("metadata", {})
        section = meta.get("true_section", meta.get("section", ""))
        if section in section_dict:
            r["fusion_score"] = r.get("fusion_score", 0) + section_dict[section] * SECTION_ROUTER_BOOST

    return results


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """Fuse dense and BM25 results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        rrf_score = 1.0 / (k + rank + 1)
        scores[cid] = scores.get(cid, 0) + rrf_score
        chunk_data[cid] = r
        chunk_data[cid]["dense_score"] = r.get("dense_score", 0)

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        rrf_score = 1.0 / (k + rank + 1)
        scores[cid] = scores.get(cid, 0) + rrf_score
        if cid not in chunk_data:
            chunk_data[cid] = r
        chunk_data[cid]["bm25_score"] = r.get("bm25_score", 0)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid]
        entry["fusion_score"] = scores[cid]
        if "dense_score" not in entry:
            entry["dense_score"] = 0
        if "bm25_score" not in entry:
            entry["bm25_score"] = 0
        results.append(entry)

    return results


def _weighted_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
    w_dense: float = 0.5,
) -> list[dict]:
    """Fuse dense and BM25 results using weighted scores."""
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    if dense_results:
        d_max = max(r.get("dense_score", 0) for r in dense_results) or 1.0
        for r in dense_results:
            cid = r["chunk_id"]
            norm_score = r.get("dense_score", 0) / d_max
            scores[cid] = scores.get(cid, 0) + w_dense * norm_score
            chunk_data[cid] = r
            chunk_data[cid]["dense_score"] = r.get("dense_score", 0)

    if bm25_results:
        b_scores = [r.get("bm25_score", 0) for r in bm25_results]
        b_max = max(b_scores) if b_scores else 1.0
        b_min = min(b_scores) if b_scores else 0.0
        b_range = b_max - b_min if b_max != b_min else 1.0
        for r in bm25_results:
            cid = r["chunk_id"]
            norm_score = (r.get("bm25_score", 0) - b_min) / b_range
            scores[cid] = scores.get(cid, 0) + (1.0 - w_dense) * norm_score
            if cid not in chunk_data:
                chunk_data[cid] = r
            chunk_data[cid]["bm25_score"] = r.get("bm25_score", 0)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid]
        entry["fusion_score"] = scores[cid]
        if "dense_score" not in entry:
            entry["dense_score"] = 0
        if "bm25_score" not in entry:
            entry["bm25_score"] = 0
        results.append(entry)

    return results
