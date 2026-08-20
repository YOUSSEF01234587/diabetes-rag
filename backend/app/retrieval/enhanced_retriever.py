"""V4 Enhanced Retriever - integrates section routing, intent V2, content-type awareness,
table retrieval, parent-child expansion, score normalization, and learning-to-rank features.

This replaces the V3 retrieval path when enabled.
"""
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
from .section_router import SectionRouter
from .intent_v2 import classify as classify_intent_v2, IntentResult
from .query_expansion_v2 import expand_query_v2, generate_query_variants_v2

logger = logging.getLogger(__name__)

# Lazy-init globals
_section_router: Optional[SectionRouter] = None


def _get_section_router() -> SectionRouter:
    global _section_router
    if _section_router is None:
        _section_router = SectionRouter()
    return _section_router


# ---------------------------------------------------------------------------
# Section boost constants
# ---------------------------------------------------------------------------
SECTION_ROUTER_BOOST = 0.30
SECTION_EXACT_MATCH_BOOST = 0.40
SECTION_PARENT_BOOST = 0.15
SOURCE_EXPLICIT_BOOST = 0.25
CONTENT_TYPE_TABLE_BOOST = 0.10
CONTENT_TYPE_RECOMMENDATION_BOOST = 0.08
QUERY_TERM_OVERLAP_BOOST = 0.05
TOPIC_PENALTY = -0.15
OFF_SECTION_PENALTY = -0.10


# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------

_TABLE_KEYWORDS = [
    "table", "threshold", "cutoff", "criteria", "comparison",
    "pros", "cons", "sensitivity", "specificity", "cost",
    "technical features", "mg/dl", "mmol/l", "%",
]
_RECOMMENDATION_KEYWORDS = [
    "recommend", "should", "guideline", "standard of care",
    "ADA recommends", "evidence level",
]


def _detect_content_type(query: str) -> dict:
    """Detect what content types the query prefers."""
    q = query.lower()
    return {
        "prefer_table": any(kw in q for kw in _TABLE_KEYWORDS),
        "prefer_recommendation": any(kw in q for kw in _RECOMMENDATION_KEYWORDS),
        "is_numeric": bool(__import__("re").search(r"\d+\.?\d*\s*(mg/dl|mmol/l|%|mg|mmol)", q)),
        "is_comparison": any(w in q for w in ["compare", "vs", "versus", "difference", "differ"]),
        "is_definition": any(w in q for w in ["what is", "what are", "define", "meaning", "definition"]),
    }


# ---------------------------------------------------------------------------
# Query-term overlap scoring
# ---------------------------------------------------------------------------

def _query_term_overlap_score(query: str, chunk_text: str) -> float:
    """Fraction of query terms found in chunk text (0-1)."""
    query_terms = set(query.lower().split())
    chunk_lower = chunk_text.lower()
    if not query_terms:
        return 0.0
    matched = sum(1 for t in query_terms if t in chunk_lower)
    return matched / len(query_terms)


# ---------------------------------------------------------------------------
# Source-aware candidate generation
# ---------------------------------------------------------------------------

def _source_filter_candidates(
    all_results: list[dict],
    source_hint: str,
    all_sources: list[str],
) -> list[dict]:
    """Boost results matching the source hint; don't eliminate others."""
    if not source_hint:
        return all_results
    for r in all_results:
        meta = r.get("metadata", {})
        source_id = meta.get("source_id", "")
        if source_hint == "niddk" and "niddk" in source_id.lower():
            r["fusion_score"] = r.get("fusion_score", 0) + SOURCE_EXPLICIT_BOOST
        elif source_hint == "ada" and "ada" in source_id.lower():
            r["fusion_score"] = r.get("fusion_score", 0) + SOURCE_EXPLICIT_BOOST
    return all_results


# ---------------------------------------------------------------------------
# Parent-child retrieval
# ---------------------------------------------------------------------------

def _expand_to_parents(
    results: list[dict],
    parent_map: dict[str, dict],
    top_k: int,
) -> list[dict]:
    """For child chunks, optionally expand to include parent context info."""
    seen_parents = set()
    expanded = []
    for r in results:
        parent_id = r.get("metadata", {}).get("parent_chunk_id", "")
        if parent_id and parent_id in parent_map and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent = parent_map[parent_id].copy()
            parent["is_parent_context"] = True
            parent["child_rank"] = r.get("rank", 0)
            expanded.append(parent)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Evidence diversity
# ---------------------------------------------------------------------------

def _apply_diversity(results: list[dict], top_k: int) -> list[dict]:
    """Apply controlled diversity: don't let one section dominate all top-K."""
    if not results:
        return results
    section_counts = {}
    diversified = []
    seen_sections = set()
    max_per_section = max(2, top_k // 2)

    # First pass: allow normal ranking
    for r in results:
        section = r.get("metadata", {}).get("section", "")
        count = section_counts.get(section, 0)
        if count < max_per_section:
            diversified.append(r)
            section_counts[section] = count + 1
        if len(diversified) >= top_k:
            break

    # Fill remaining from excluded results if needed
    if len(diversified) < top_k:
        for r in results:
            if r not in diversified:
                diversified.append(r)
            if len(diversified) >= top_k:
                break

    return diversified[:top_k]


# ---------------------------------------------------------------------------
# Learning-to-rank features
# ---------------------------------------------------------------------------

def _compute_ltr_features(
    query: str,
    result: dict,
    intent: Optional[IntentResult],
    content_types: dict,
    section_scores: list[tuple[str, float]],
) -> dict:
    """Compute transparent ranking features for a single result."""
    meta = result.get("metadata", {})
    section = meta.get("section", "")
    source_id = meta.get("source_id", "")
    text = result.get("text", "")

    # Semantic score (use normalized if available)
    semantic_score = result.get("dense_score_norm", result.get("dense_score", 0.0))

    # BM25 score (use normalized if available)
    bm25_score = result.get("bm25_score_norm", 0.0)

    # Intent match
    intent_match = 0.0
    if intent:
        if section in getattr(intent, "preferred_sections", []):
            intent_match = 1.0
        elif any(s.lower() in section.lower() for s in getattr(intent, "preferred_sections", [])):
            intent_match = 0.5

    # Source match
    source_match = 0.0
    if intent:
        for ps in getattr(intent, "preferred_sources", []):
            if ps in source_id:
                source_match = 1.0
                break

    # Section router match
    section_router_score = 0.0
    for sec_name, sec_score in section_scores:
        if sec_name == section:
            section_router_score = sec_score
            break
        if sec_name.lower() in section.lower():
            section_router_score = sec_score * 0.7
            break

    # Content type match
    content_type_match = 0.0
    has_table = meta.get("has_table", False)
    if content_types.get("prefer_table") and has_table:
        content_type_match = 1.0
    elif content_types.get("prefer_recommendation"):
        # Check if text contains recommendation language
        if any(kw in text.lower() for kw in ["recommend", "should"]):
            content_type_match = 0.8

    # Query term overlap
    query_overlap = _query_term_overlap_score(query, text)

    return {
        "semantic_score": round(semantic_score, 4),
        "bm25_score_raw": round(bm25_score, 4),
        "intent_match": round(intent_match, 4),
        "source_match": round(source_match, 4),
        "section_router_score": round(section_router_score, 4),
        "content_type_match": round(content_type_match, 4),
        "query_term_overlap": round(query_overlap, 4),
    }


def _combined_ltr_score(features: dict) -> float:
    """Transparent weighted combination of features into a single score."""
    return (
        features["semantic_score"] * 0.30 +
        features["bm25_score_raw"] * 0.15 +
        features["intent_match"] * 0.20 +
        features["source_match"] * 0.10 +
        features["section_router_score"] * 0.15 +
        features["content_type_match"] * 0.05 +
        features["query_term_overlap"] * 0.05
    )


# ---------------------------------------------------------------------------
# Main V4 enhanced search
# ---------------------------------------------------------------------------

def v4_enhanced_search(
    query: str,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    embedding_model: str = EMBEDDING_MODEL,
    enable_reranker: bool = False,
    use_section_router: bool = True,
    use_intent_v2: bool = True,
    use_content_type: bool = True,
    use_diversity: bool = True,
    use_ltr: bool = True,
    dense_weight: float = 0.5,
) -> dict:
    """V4 enhanced retrieval pipeline.

    Stages:
    1. Intent classification (V2 layered)
    2. Query expansion (V2 context-aware)
    3. Query variant generation
    4. Section routing
    5. Content-type detection
    6. Per-variant dense + BM25 search
    7. Weighted fusion
    8. Section/source/intent boosts
    9. LTR scoring
    10. Diversity control
    11. Top-K selection
    12. (Optional) reranking
    13. Format results
    """
    timings = {}
    t_start = time.time()

    # 1. Intent classification
    t0 = time.time()
    intent = classify_intent_v2(query) if use_intent_v2 else None
    timings["intent_ms"] = round((time.time() - t0) * 1000, 1)

    # 2. Query expansion
    t0 = time.time()
    expanded_query = expand_query_v2(query, intent)
    timings["expansion_ms"] = round((time.time() - t0) * 1000, 1)

    # 3. Query variants
    t0 = time.time()
    variants_with_reasons = generate_query_variants_v2(query, intent)
    # Always include the expanded original as the primary variant
    all_variants = [(expanded_query, "primary expanded query")]
    all_variants.extend(variants_with_reasons)
    # Deduplicate
    seen = set()
    unique_variants = []
    for v, reason in all_variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append((v, reason))
    timings["variants_ms"] = round((time.time() - t0) * 1000, 1)

    # 4. Section routing
    t0 = time.time()
    router = _get_section_router() if use_section_router else None
    intent_label = intent.primary_intent if intent else None
    section_scores = router.route_query(query, intent_label) if router else []
    timings["section_route_ms"] = round((time.time() - t0) * 1000, 1)

    # 5. Content-type detection
    content_types = _detect_content_type(query) if use_content_type else {}

    # 6. Source hint
    source_hint = ""
    if intent and hasattr(intent, "preferred_sources"):
        for src in getattr(intent, "preferred_sources", []):
            if "niddk" in src:
                source_hint = "niddk"
                break
            if "ada" in src:
                source_hint = "ada"
                break
    if not source_hint:
        query_lower = query.lower()
        if "niddk" in query_lower:
            source_hint = "niddk"
        elif "ada" in query_lower or "standards of care" in query_lower:
            source_hint = "ada"

    # 6. Per-variant search
    t0 = time.time()
    all_result_lists = []
    for variant, reason in unique_variants:
        try:
            q_embedding = embed_query(variant, model_name=embedding_model)
            dense_results = query_dense(q_embedding, top_k=rerank_top_k)
            bm25_results = search_bm25(variant, top_k=rerank_top_k)
        except Exception as e:
            logger.warning(f"Variant search failed ({reason}): {e}")
            continue

        # Weighted fusion
        fused = _weighted_fusion_v4(dense_results, bm25_results, rerank_top_k, dense_weight)

        # Source boost
        fused = _source_filter_candidates(fused, source_hint, [])

        # Section boosts from router
        if use_section_router and section_scores:
            section_dict = dict(section_scores)
            for r in fused:
                sec = r.get("metadata", {}).get("section", "")
                if sec in section_dict:
                    r["fusion_score"] = r.get("fusion_score", 0) + section_dict[sec] * SECTION_ROUTER_BOOST
                # Check parent sections
                if router:
                    parent = router.get_parent(sec)
                    if parent and parent in section_dict:
                        r["fusion_score"] = r.get("fusion_score", 0) + section_dict[parent] * SECTION_PARENT_BOOST

        # Intent boosts
        if use_intent_v2 and intent:
            for r in fused:
                meta = r.get("metadata", {})
                sec = meta.get("section", "")
                src = meta.get("source_id", "")

                # Section boost
                from .intent_v2 import get_section_boost, get_source_boost, get_topic_penalty
                sec_boost = get_section_boost(sec, intent)
                if sec_boost > 0:
                    r["fusion_score"] = r.get("fusion_score", 0) + sec_boost

                # Source boost
                src_boost = get_source_boost(src, intent)
                if src_boost > 0:
                    r["fusion_score"] = r.get("fusion_score", 0) + src_boost

                # Topic penalty
                penalty = get_topic_penalty(sec, intent)
                if penalty < 0:
                    r["fusion_score"] = r.get("fusion_score", 0) + penalty

        # Content-type boosts
        if use_content_type:
            for r in fused:
                meta = r.get("metadata", {})
                if content_types.get("prefer_table") and meta.get("has_table", False):
                    r["fusion_score"] = r.get("fusion_score", 0) + CONTENT_TYPE_TABLE_BOOST

        fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        all_result_lists.append(fused[:rerank_top_k])

    timings["search_ms"] = round((time.time() - t0) * 1000, 1)

    # 7. RRF fusion across variants
    t0 = time.time()
    if len(all_result_lists) > 1:
        fused = _rrf_fusion(all_result_lists, rerank_top_k, k=60)
    elif all_result_lists:
        fused = all_result_lists[0]
    else:
        fused = []

    # 8. Normalize scores for LTR
    if fused:
        _normalize_scores(fused)

    # 9. LTR scoring
    if use_ltr:
        for r in fused:
            features = _compute_ltr_features(query, r, intent, content_types, section_scores)
            r["ltr_features"] = features
            r["ltr_score"] = _combined_ltr_score(features)
        # Re-rank by LTR score
        fused.sort(key=lambda x: x.get("ltr_score", 0), reverse=True)

    # 10. Diversity
    if use_diversity:
        fused = _apply_diversity(fused, rerank_top_k)

    timings["fusion_ms"] = round((time.time() - t0) * 1000, 1)

    # 11. Reranking (optional)
    t0 = time.time()
    if enable_reranker and fused:
        reranked = rerank(query, fused, top_k=top_k, model_name=RERANKER_MODEL)
    else:
        reranked = fused[:top_k]
        for i, r in enumerate(reranked):
            r["reranker_score"] = r.get("ltr_score", r.get("fusion_score", 0))
            r["rank"] = i + 1
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    # 12. Format results
    final = _format_v4_results(reranked)

    timings["total_ms"] = round((time.time() - t_start) * 1000, 1)

    # Evidence validation (simplified)
    from .evidence_validation import validate_evidence, compute_enhanced_confidence
    evidence = validate_evidence(query, final, intent)
    confidence = compute_enhanced_confidence(final, query, intent, evidence)

    return {
        "query": query,
        "intent": _intent_to_dict(intent),
        "section_scores": [(s, round(sc, 3)) for s, sc in section_scores[:5]],
        "content_types": content_types,
        "source_hint": source_hint,
        "variants_used": len(unique_variants),
        "results": final,
        "confidence": confidence,
        "timings": timings,
        "total_candidates": sum(len(rl) for rl in all_result_lists),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_fusion_v4(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
    w_dense: float = 0.5,
) -> list[dict]:
    """Fuse dense and BM25 with normalized scores."""
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    if dense_results:
        d_max = max(r.get("dense_score", 0) for r in dense_results) or 1.0
        for r in dense_results:
            cid = r["chunk_id"]
            norm = r.get("dense_score", 0) / d_max
            scores[cid] = scores.get(cid, 0) + w_dense * norm
            chunk_data[cid] = r
            chunk_data[cid]["dense_score"] = r.get("dense_score", 0)

    if bm25_results:
        b_scores = [r.get("bm25_score", 0) for r in bm25_results]
        b_max = max(b_scores) if b_scores else 1.0
        b_min = min(b_scores) if b_scores else 0.0
        b_range = b_max - b_min if b_max != b_min else 1.0
        for r in bm25_results:
            cid = r["chunk_id"]
            norm = (r.get("bm25_score", 0) - b_min) / b_range
            scores[cid] = scores.get(cid, 0) + (1.0 - w_dense) * norm
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


def _rrf_fusion(
    result_lists: list[list[dict]],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists."""
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, r in enumerate(result_list):
            cid = r["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf_score
            if cid not in chunk_data:
                chunk_data[cid] = r
            else:
                # Keep the one with higher fusion_score
                if r.get("fusion_score", 0) > chunk_data[cid].get("fusion_score", 0):
                    chunk_data[cid] = r

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid]
        entry["rrf_score"] = scores[cid]
        entry["fusion_score"] = scores[cid]  # Use RRF as fusion score
        results.append(entry)
    return results


def _normalize_scores(results: list[dict]):
    """Normalize dense and bm25 scores to 0-1 range within the result list."""
    if not results:
        return
    dense_scores = [r.get("dense_score", 0) for r in results]
    bm25_scores = [r.get("bm25_score", 0) for r in results]

    d_max = max(dense_scores) if dense_scores else 1.0
    d_min = min(dense_scores) if dense_scores else 0.0
    d_range = d_max - d_min if d_max != d_min else 1.0

    b_max = max(bm25_scores) if bm25_scores else 1.0
    b_min = min(bm25_scores) if bm25_scores else 0.0
    b_range = b_max - b_min if b_max != b_min else 1.0

    for r in results:
        r["dense_score_norm"] = (r.get("dense_score", 0) - d_min) / d_range
        r["bm25_score_norm"] = (r.get("bm25_score", 0) - b_min) / b_range


def _format_v4_results(reranked: list[dict]) -> list[dict]:
    """Format reranked results into final output."""
    final = []
    for r in reranked:
        meta = r.get("metadata", {})
        ltr_features = r.get("ltr_features", {})
        final.append({
            "rank": r.get("rank", len(final) + 1),
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "fusion_score": round(r.get("fusion_score", 0), 4),
            "dense_score": round(r.get("dense_score", 0), 4),
            "bm25_score": round(r.get("bm25_score", 0), 4),
            "reranker_score": round(r.get("reranker_score", 0), 4),
            "ltr_score": round(r.get("ltr_score", 0), 4),
            "rrf_score": round(r.get("rrf_score", 0), 4),
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
            "is_parent_context": r.get("is_parent_context", False),
            "ltr_features": ltr_features,
        })
    return final


def _intent_to_dict(intent) -> dict | None:
    """Convert IntentResult to dict for JSON serialization."""
    if not intent:
        return None
    return {
        "primary": getattr(intent, "primary_intent", ""),
        "confidence": round(getattr(intent, "confidence", 0), 2),
        "all_intents": getattr(intent, "all_intents", []),
        "preferred_sections": getattr(intent, "preferred_sections", []),
        "preferred_sources": getattr(intent, "preferred_sources", []),
        "excluded_sections": getattr(intent, "excluded_sections", []),
        "is_table_query": getattr(intent, "is_table_query", False),
        "is_comparison": getattr(intent, "is_comparison", False),
        "is_threshold_query": getattr(intent, "is_threshold_query", False),
    }
