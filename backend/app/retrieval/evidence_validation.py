"""Phase 12-13: Evidence validation and confidence estimation."""
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvidenceReport:
    """Evidence validation report for a set of retrieval results."""
    has_evidence: bool
    evidence_score: float
    source_agreement: float
    section_coherence: float
    concept_coverage: float
    top_evidence: list
    warnings: list


def validate_evidence(
    query: str,
    results: list[dict],
    intent=None,
    required_concepts: list[str] = None,
) -> EvidenceReport:
    """Validate that retrieved evidence supports the query.
    
    Checks:
    1. Source agreement - do top results agree on source?
    2. Section coherence - do top results come from related sections?
    3. Concept coverage - do results contain key medical concepts?
    4. Score distribution - is there a clear evidence leader?
    """
    if not results:
        return EvidenceReport(
            has_evidence=False,
            evidence_score=0.0,
            source_agreement=0.0,
            section_coherence=0.0,
            concept_coverage=0.0,
            top_evidence=[],
            warnings=["No results retrieved"],
        )
    
    warnings = []
    
    # Source agreement
    top_n = min(5, len(results))
    top_sources = [r.get("source_id", "") for r in results[:top_n]]
    if top_sources:
        most_common_source = max(set(top_sources), key=top_sources.count)
        source_agreement = top_sources.count(most_common_source) / len(top_sources)
    else:
        source_agreement = 0.0
    
    # Section coherence
    top_sections = [r.get("section", "") for r in results[:top_n]]
    if top_sections:
        most_common_section = max(set(top_sections), key=top_sections.count)
        section_coherence = top_sections.count(most_common_section) / len(top_sections)
    else:
        section_coherence = 0.0
    
    # Concept coverage
    if required_concepts:
        all_text = " ".join(r.get("text", "") for r in results[:top_n]).lower()
        covered = sum(1 for c in required_concepts if c.lower() in all_text)
        concept_coverage = covered / len(required_concepts)
    elif intent and intent.required_concepts:
        all_text = " ".join(r.get("text", "") for r in results[:top_n]).lower()
        covered = sum(1 for c in intent.required_concepts if c.lower() in all_text)
        concept_coverage = covered / len(intent.required_concepts)
    else:
        # Use query terms as proxy
        query_terms = set(query.lower().split())
        all_text = " ".join(r.get("text", "") for r in results[:top_n]).lower()
        text_terms = set(all_text.split())
        common = query_terms & text_terms
        concept_coverage = len(common) / len(query_terms) if query_terms else 0.0
    
    # Score distribution - check for clear leader
    scores = [r.get("fusion_score", 0) for r in results[:top_n]]
    if len(scores) >= 2:
        top_score = scores[0]
        avg_rest = sum(scores[1:]) / len(scores[1:]) if len(scores) > 1 else 0
        score_gap = top_score - avg_rest
        score_leader = score_gap > 0.1 and top_score > 0.3
    else:
        score_leader = scores[0] > 0.3 if scores else False
    
    # Overall evidence score
    evidence_score = (
        source_agreement * 0.25 +
        section_coherence * 0.25 +
        concept_coverage * 0.35 +
        (1.0 if score_leader else 0.0) * 0.15
    )
    
    # Generate warnings
    if source_agreement < 0.5:
        warnings.append("Low source agreement - results span multiple sources")
    if section_coherence < 0.4:
        warnings.append("Low section coherence - results span multiple sections")
    if concept_coverage < 0.3:
        warnings.append("Low concept coverage - key terms not found in results")
    if not score_leader:
        warnings.append("No clear score leader - multiple similar results")
    
    has_evidence = evidence_score >= 0.4 and not any(
        "Low" in w and "concept" in w for w in warnings
    )
    
    return EvidenceReport(
        has_evidence=has_evidence,
        evidence_score=round(evidence_score, 4),
        source_agreement=round(source_agreement, 4),
        section_coherence=round(section_coherence, 4),
        concept_coverage=round(concept_coverage, 4),
        top_evidence=[r.get("chunk_id", "") for r in results[:3]],
        warnings=warnings,
    )


def compute_enhanced_confidence(
    results: list[dict],
    query: str,
    intent=None,
    evidence_report: EvidenceReport = None,
) -> dict:
    """Compute enhanced retrieval confidence with evidence validation.
    
    Combines:
    - Score-based confidence (existing)
    - Evidence validation (new)
    - Intent-match confidence (new)
    """
    if not results:
        return {
            "level": "none",
            "score": 0.0,
            "reason": "No results retrieved",
            "evidence": None,
        }
    
    # Score-based confidence
    top_score = results[0].get("fusion_score", 0)
    if len(results) > 1:
        rest_scores = [r.get("fusion_score", 0) for r in results[1:5]]
        avg_rest = sum(rest_scores) / len(rest_scores) if rest_scores else 0
        gap = top_score - avg_rest
    else:
        gap = top_score
    
    top_source = results[0].get("source_id", "")
    top_sources = [r.get("source_id", "") for r in results[:5]]
    source_agreement = sum(1 for s in top_sources if s == top_source) / len(top_sources) if top_sources else 0
    
    score_confidence = (top_score * 0.4 + gap * 0.3 + source_agreement * 0.3)
    score_confidence = max(0.0, min(1.0, score_confidence))
    
    # Evidence-based confidence
    evidence_confidence = evidence_report.evidence_score if evidence_report else 0.5
    
    # Intent-match confidence
    intent_confidence = intent.confidence if intent else 0.3
    
    # Combined confidence
    combined = (
        score_confidence * 0.40 +
        evidence_confidence * 0.40 +
        intent_confidence * 0.20
    )
    combined = max(0.0, min(1.0, combined))
    
    # Determine level
    if combined >= 0.65:
        level = "strong"
        reason = "High-confidence retrieval with consistent evidence"
    elif combined >= 0.40:
        level = "moderate"
        reason = "Moderate confidence - evidence present but may need verification"
    elif combined >= 0.25:
        level = "weak"
        reason = "Low confidence - limited evidence found"
    else:
        level = "insufficient"
        reason = "Insufficient evidence - system should abstain"
    
    return {
        "level": level,
        "score": round(combined, 4),
        "reason": reason,
        "score_confidence": round(score_confidence, 4),
        "evidence_confidence": round(evidence_confidence, 4),
        "intent_confidence": round(intent_confidence, 4),
        "top_score": round(top_score, 4),
        "score_gap": round(gap, 4),
        "source_agreement": round(source_agreement, 4),
        "warnings": evidence_report.warnings if evidence_report else [],
    }
