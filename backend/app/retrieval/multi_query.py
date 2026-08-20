"""Phase 4-7: Multi-query retrieval with intent-aware expansion and decomposition."""
import re
import logging
from typing import Optional

from .query_intent import detect_intent, QueryIntent

logger = logging.getLogger(__name__)


# Intent-specific expansion templates
INTENT_EXPANSIONS = {
    "diagnostic_criteria": [
        "diagnostic criteria thresholds A1C FPG OGTT classification",
        "diagnostic thresholds mg/dL mmol/L percentage",
    ],
    "diagnosis_confirmation": [
        "confirm diagnosis repeat testing two different tests",
        "confirmation requires repeat A1C or OGTT",
    ],
    "prediabetes_criteria": [
        "prediabetes impaired fasting glucose IFG impaired glucose tolerance IGT",
        "prediabetes A1C 5.7-6.4% FPG 100-125 OGTT 140-199",
    ],
    "test_comparison": [
        "pros cons advantages disadvantages comparison sensitivity specificity",
        "test characteristics reproducibility convenience cost",
    ],
    "test_interference": [
        "interference hemoglobin variants G6PD sickle cell anemia",
        "conditions affecting test accuracy reliability",
    ],
    "type1": [
        "type 1 diabetes autoantibodies autoantibody autoimmune",
        "type 1 diabetes staging risk screening prediction",
    ],
    "type2": [
        "type 2 diabetes risk factors screening assessment",
        "type 2 diabetes prevention identification",
    ],
    "gestational": [
        "gestational diabetes GDM pregnancy prenatal screening",
        "gestational diabetes diagnosis thresholds 75g OGTT",
    ],
    "monogenic": [
        "monogenic diabetes MODY maturity-onset diabetes young",
        "monogenic diabetes genetic testing syndromes",
    ],
    "screening": [
        "screening frequency risk assessment when to test",
        "screening recommendations asymptomatic adults children",
    ],
    "table_lookup": [
        "table comparison criteria chart diagnostic",
    ],
    "fpg": [
        "fasting plasma glucose FPG fasting blood sugar glucose",
        "fasting test 12 hours 8 hours overnight",
    ],
    "ogtt": [
        "oral glucose tolerance test OGTT 2-hour glucose",
        "OGTT 75 grams glucose load WHO IADPSG",
    ],
    "a1c": [
        "A1C hemoglobin HbA1c glycated hemoglobin HbA",
        "A1C 6.5% NGSP DCCT IFCC standardization",
    ],
    "random_glucose": [
        "random plasma glucose RPG random blood sugar",
        "random glucose 200 mg/dL symptoms hyperglycemia",
    ],
}


def generate_query_variants(query: str, intent: QueryIntent, max_variants: int = 3) -> list[str]:
    """Generate multiple query variants for better retrieval coverage.
    
    Returns list of query strings to search with, including:
    - Original query
    - Intent-aware expanded query
    - Section-targeted variant (if section is known)
    """
    variants = [query]
    
    # Add intent-aware expansion
    if intent.primary_intent in INTENT_EXPANSIONS:
        expansions = INTENT_EXPANSIONS[intent.primary_intent]
        if expansions:
            expanded = f"{query} {expansions[0]}"
            variants.append(expanded)
    
    # Add section-targeted variant for specific sections
    if intent.preferred_sections:
        for section in intent.preferred_sections[:2]:
            section_terms = {
                "Gestational Diabetes Mellitus": "75-g OGTT 24-28 weeks IADPSG one-step two-step",
                "Monogenic Diabetes Syndromes": "MODY HNF1A GCK KCNJ11 MODY1 MODY2 MODY3 genetic testing monogenic",
                "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults": "screening frequency risk assessment every 3 years asymptomatic adults",
                "Type 1 Diabetes": "autoantibodies autoantibody IAA GADA IA-2A ZnT8 staging risk",
                "Diagnosis of Prediabetes": "prediabetes IFG IGT impaired fasting glucose 100-125 140-199 5.7-6.4",
                "Comparing Diabetes Blood Tests": "pros cons advantages disadvantages sensitivity specificity coefficient variation",
                "Confirming the Diagnosis": "confirming diagnosis confirmation repeat test two different results",
            }
            terms = section_terms.get(section, "")
            section_query = f"{query} {terms}" if terms else f"{query} section:{section}"
            variants.append(section_query)
    
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    return unique_variants[:max_variants]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """Fuse multiple result lists using Reciprocal Rank Fusion.
    
    Each result list comes from a different query variant.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    
    for result_list in result_lists:
        for rank, r in enumerate(result_list):
            cid = r["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf_score
            
            if cid not in chunk_data:
                chunk_data[cid] = r
            # Keep highest individual score
            if r.get("fusion_score", 0) > chunk_data[cid].get("fusion_score", 0):
                chunk_data[cid] = r
    
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
    
    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid]
        entry["fusion_score"] = scores[cid]
        entry["rrf_score"] = scores[cid]
        results.append(entry)
    
    return results


def decompose_compound_query(query: str) -> list[str]:
    """Decompose compound queries into sub-queries.
    
    Handles queries like "What are the criteria for diabetes AND prediabetes?"
    by splitting into separate sub-queries.
    """
    # Check for compound conjunctions
    compound_patterns = [
        (r"^(.+?)\s+and\s+(.+?)\s*(?:\?)?$", None),
        (r"^(.+?)\s+versus\s+(.+?)\s*(?:\?)?$", None),
        (r"^(.+?)\s+vs\.?\s+(.+?)\s*(?:\?)?$", None),
    ]
    
    for pattern, _ in compound_patterns:
        m = re.match(pattern, query, re.IGNORECASE)
        if m:
            sub1 = m.group(1).strip()
            sub2 = m.group(2).strip()
            # Only decompose if both parts look like valid queries
            if len(sub1.split()) >= 3 and len(sub2.split()) >= 3:
                return [sub1, sub2]
    
    return [query]


def expand_with_intent_terms(query: str, intent: QueryIntent) -> str:
    """Expand query with intent-specific medical terms.
    
    This is a more targeted expansion than the general query_expansion.py.
    """
    query_lower = query.lower()
    additions = []
    
    # Add concepts from intent that aren't already in the query
    for concept in intent.required_concepts:
        concept_lower = concept.lower()
        if concept_lower not in query_lower and len(concept) > 2:
            additions.append(concept)
    
    # Add section-specific terms
    if intent.preferred_sections:
        for section in intent.preferred_sections:
            if "Screening" in section and "screen" not in query_lower:
                additions.append("screening")
            elif "Confirming" in section and "confirm" not in query_lower:
                additions.append("confirmation")
            elif "Prediabetes" in section and "prediabetes" not in query_lower:
                additions.append("prediabetes")
            elif "Gestational" in section and "gestational" not in query_lower and "pregnancy" not in query_lower:
                additions.append("gestational diabetes")
            elif "Monogenic" in section and "monogenic" not in query_lower and "mody" not in query_lower:
                additions.append("monogenic MODY")
            elif "Type 1" in section and "type 1" not in query_lower:
                additions.append("type 1 diabetes")
            elif "Comparing" in section and "compar" not in query_lower:
                additions.append("comparison pros cons")
    
    if additions:
        expanded = f"{query} {' '.join(additions[:5])}"
        return expanded
    
    return query


def multi_query_retrieve(
    query: str,
    search_fn,
    top_k: int = 10,
    use_decomposition: bool = True,
    use_expansion: bool = True,
) -> dict:
    """Perform multi-query retrieval with intent-aware expansion.
    
    Args:
        query: Original query
        search_fn: Function that takes (query, top_k) and returns search results
        top_k: Number of final results
        use_decomposition: Whether to decompose compound queries
        use_expansion: Whether to use intent-aware expansion
    
    Returns:
        dict with fused results and metadata
    """
    intent = detect_intent(query)
    
    # Generate query variants
    variants = generate_query_variants(query, intent)
    
    # Also decompose if compound
    if use_decomposition:
        decomposed = decompose_compound_query(query)
        if len(decomposed) > 1:
            for sub_q in decomposed:
                sub_intent = detect_intent(sub_q)
                sub_variants = generate_query_variants(sub_q, sub_intent)
                variants.extend(sub_variants)
    
    # Expand each variant with intent terms
    if use_expansion:
        expanded_variants = []
        for v in variants:
            expanded = expand_with_intent_terms(v, detect_intent(v))
            expanded_variants.append(expanded)
        variants.extend(expanded_variants)
    
    # Deduplicate variants
    seen = set()
    unique_variants = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    logger.debug(f"Multi-query: {len(unique_variants)} variants for '{query[:50]}'")
    
    # Retrieve with each variant
    all_results = []
    for variant in unique_variants:
        results = search_fn(variant, top_k=top_k)
        all_results.append(results)
    
    # Fuse using RRF
    fused = reciprocal_rank_fusion(all_results, top_k=top_k)
    
    return {
        "fused_results": fused,
        "num_variants": len(unique_variants),
        "variants": unique_variants,
        "intent": intent,
    }
