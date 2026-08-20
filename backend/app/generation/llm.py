"""LLM generation pipeline with provider chain.

Provider order (auto): Gemini → Groq → OpenRouter → safe refusal.
All providers receive the SAME prompt (evidence, instructions, safety constraints).
"""
import logging
import time
from typing import Optional

from ..config import (
    LLM_MODEL, LLM_BASE_URL, API_KEY, LLM_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
)
from .providers import build_provider_chain

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    search_results: list[dict],
    model: str = LLM_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    evidence_k: int = 5,
) -> dict:
    """Generate a grounded answer from retrieved evidence.

    Pipeline:
    1. Pre-flight safety checks (emergency, no evidence, low relevance)
    2. Build evidence pack from retrieval results
    3. Build grounded generation prompt
    4. Try providers in chain: Gemini → Groq → OpenRouter
    5. Post-generation: citation validation, answer verification
    6. Return structured response
    """
    timings = {}
    from .prompt import classify_query
    query_type = classify_query(query)

    if query_type == "emergency":
        return _emergency_response(query, query_type)

    if query_type in ("medication_question", "personal_medical_advice"):
        return _refusal_response(
            query, "medical_advice",
            f"This question asks for {'medication dosing' if query_type == 'medication_question' else 'personal medical advice'}, "
            "which requires evaluation by a qualified healthcare professional.",
            query_type,
        )

    if not search_results:
        return _refusal_response(query, "no_evidence", "No relevant evidence was retrieved from the indexed documents.", query_type)

    top_score = (
        search_results[0].get("dense_score")
        or search_results[0].get("reranker_score")
        or search_results[0].get("fusion_score", 0)
    )
    if top_score < 0.15:
        return _refusal_response(
            query, "low_relevance",
            f"The retrieved evidence has very low relevance (top score: {top_score:.3f}). "
            "The indexed documents may not contain information about this topic.",
            query_type,
        )

    from ..evidence.evidence_validator import EvidenceValidator
    evidence_validator = EvidenceValidator(evidence_k=evidence_k)
    evidence_result = evidence_validator.build_and_validate(query, search_results)

    from .prompt import build_grounded_prompt
    system_prompt, user_message = build_grounded_prompt(
        query,
        evidence_result.evidence_pack,
        evidence_result.conflict_report,
    )

    # ── Provider chain ──
    forced = LLM_PROVIDER if LLM_PROVIDER not in ("openai", "openrouter", "") else ""
    chain = build_provider_chain(
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        groq_api_key=GROQ_API_KEY,
        groq_model=GROQ_MODEL,
        groq_base_url=GROQ_BASE_URL,
        openrouter_api_key=API_KEY,
        openrouter_model=LLM_MODEL,
        openrouter_base_url=LLM_BASE_URL,
        forced_provider=forced,
    )

    provider_names = [p.name for p in chain]
    timings["provider_chain"] = provider_names
    logger.info(f"[LLM] Provider chain: {' → '.join(provider_names) if provider_names else 'none configured'}")

    answer_text = None
    provider_meta = {}
    provider_results = []

    for provider in chain:
        logger.info(f"[PROVIDER] Trying {provider.name}...")
        text, meta = provider.generate(system_prompt, user_message, max_tokens, temperature)
        timings["provider"] = meta.get("provider", provider.name)
        timings["llm_ms"] = meta.get("latency_ms", 0)
        timings["llm"] = meta

        latency_str = f"{meta.get('latency_ms', 0) / 1000:.1f}s"
        if text:
            provider_results.append(f"{provider.name}: {latency_str} — success")
            logger.info(f"[PROVIDER] {provider.name} result: success ({latency_str})")
            answer_text = text
            provider_meta = meta
            break
        else:
            error_type = meta.get("failure_type", "unknown")
            provider_results.append(f"{provider.name}: {latency_str} — {error_type}")
            provider_meta = meta
            logger.warning(f"[PROVIDER] {provider.name} result: {error_type} ({latency_str})")

    timings["provider_results"] = provider_results

    if answer_text is None:
        error_detail = provider_meta.get("error", "All providers failed")
        failure_type = provider_meta.get("failure_type", "unknown")
        logger.error(f"[LLM] All providers failed. Last: {provider_meta.get('provider', '?')} — {error_detail}")
        return _provider_failure_response(
            query, query_type, timings, provider_names, failure_type,
            evidence_result=evidence_result,
        )

    post_result = evidence_validator.validate_post_generation(evidence_result, answer_text)

    from .citation_engine import build_citations_from_evidence, validate_answer_citations
    citations = build_citations_from_evidence(evidence_result.evidence_pack)
    citation_validation = validate_answer_citations(answer_text, evidence_result.evidence_pack)
    citation_issues = [
        issue for issue in citation_validation.issues
        if "Hallucinated" in issue or "non-existent" in issue.lower()
    ]

    from .answer_verifier import verify_answer
    verification = verify_answer(
        answer_text=answer_text,
        evidence_pack=evidence_result.evidence_pack,
        citations=citations,
        citation_issues=citation_issues,
    )

    from .safety import check_safety
    safety_result = check_safety(query, answer_text, query_type)

    grounded = post_result.is_grounded and verification.passed
    grounding_score = post_result.grounding_score

    if not verification.passed:
        logger.warning(f"Answer verification failed for '{query}': {verification.issues}")
        return _refusal_response(
            query, "verification_failed",
            f"Answer verification found issues: {'; '.join(verification.issues)}",
            query_type, timings,
            evidence_validation=post_result.to_dict(),
            verification=verification.to_dict(),
            evidence_result=evidence_result,
            citations=citations,
        )

    if grounding_score < 0.3:
        return _refusal_response(
            query, "low_grounding",
            f"Grounding score too low ({grounding_score:.2f}). Evidence is insufficient or citations are invalid.",
            query_type, timings,
            evidence_validation=post_result.to_dict(),
            verification=verification.to_dict(),
            evidence_result=evidence_result,
            citations=citations,
        )

    confidence = _assess_confidence(top_score, grounded, len(search_results), grounding_score)

    evidence_summary = [
        {
            "chunk_id": c.chunk_id,
            "source_id": c.source_id,
            "organization": c.organization,
            "section": c.section,
            "page": c.page,
            "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
        }
        for c in evidence_result.evidence_pack.chunks
    ]

    sources = evidence_result.evidence_pack.source_citations()

    return {
        "answer": answer_text,
        "confidence": confidence,
        "grounded": grounded,
        "grounding_score": grounding_score,
        "citations": citations,
        "sources": sources,
        "evidence": evidence_summary,
        "refused": False,
        "refusal_reason": None,
        "query_type": query_type,
        "evidence_validation": post_result.to_dict(),
        "verification": verification.to_dict(),
        "safety": safety_result,
        "timings": timings,
    }


def _emergency_response(query: str, query_type: str) -> dict:
    return {
        "answer": (
            "This appears to be an urgent medical situation. I cannot diagnose or provide emergency "
            "medical advice. **Please seek immediate medical attention or call emergency services.**\n\n"
            "If you need general information from the indexed medical sources, please rephrase your "
            "question without the emergency context."
        ),
        "confidence": "low",
        "grounded": False,
        "grounding_score": 0.0,
        "citations": [],
        "sources": [],
        "evidence": [],
        "refused": True,
        "refusal_reason": "emergency",
        "query_type": query_type,
        "verification": {"passed": False, "issues": ["Emergency query"]},
        "safety": {"requires_professional": True, "risk_level": "high"},
        "timings": {},
    }


def _refusal_response(
    query: str, reason: str, detail: str, query_type: str,
    timings: dict = None, evidence_validation=None, verification=None,
    evidence_result=None, citations=None,
) -> dict:
    """Build refusal response. Preserves evidence when available
    (e.g. verification_failed, low_grounding) so frontend can show
    retrieved sources without presenting them as answering the question."""
    from .prompt import build_refusal_response

    evidence_summary = []
    sources = []
    if evidence_result and evidence_result.evidence_pack:
        evidence_summary = [
            {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "organization": c.organization,
                "section": c.section,
                "page": c.page,
                "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
            }
            for c in evidence_result.evidence_pack.chunks
        ]
        sources = evidence_result.evidence_pack.source_citations()

    return {
        "answer": build_refusal_response(query, reason, detail),
        "confidence": "insufficient",
        "grounded": False,
        "grounding_score": 0.0,
        "citations": citations or [],
        "sources": sources,
        "evidence": evidence_summary,
        "refused": True,
        "refusal_reason": reason,
        "query_type": query_type,
        "evidence_validation": evidence_validation,
        "verification": verification or {"passed": False, "issues": [detail]},
        "safety": {"requires_professional": False, "risk_level": "low"},
        "timings": timings or {},
    }


def _provider_failure_response(
    query: str, query_type: str, timings: dict,
    provider_chain: list, failure_type: str, evidence_result=None,
) -> dict:
    """Response when all LLM providers fail. NOT a safety refusal.
    Preserves evidence metadata so frontend can show what was retrieved."""
    from .prompt import build_refusal_response
    providers_tried = ", ".join(provider_chain) if provider_chain else "none configured"
    detail = (
        f"Generation service unavailable. Providers tried: {providers_tried}. "
        f"Last failure: {failure_type}. The evidence retrieval system is working, "
        f"but the answer-generation service is temporarily unavailable."
    )

    evidence_summary = []
    sources = []
    if evidence_result and evidence_result.evidence_pack:
        evidence_summary = [
            {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "organization": c.organization,
                "section": c.section,
                "page": c.page,
                "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text,
            }
            for c in evidence_result.evidence_pack.chunks
        ]
        sources = evidence_result.evidence_pack.source_citations()

    return {
        "answer": build_refusal_response(query, "provider_failure", detail),
        "confidence": "insufficient",
        "grounded": False,
        "grounding_score": 0.0,
        "citations": [],
        "sources": sources,
        "evidence": evidence_summary,
        "refused": True,
        "refusal_reason": "provider_failure",
        "query_type": query_type,
        "evidence_validation": None,
        "verification": {"passed": False, "issues": [detail]},
        "safety": {"requires_professional": False, "risk_level": "low"},
        "timings": timings,
    }


def _assess_confidence(top_score: float, grounded: bool, num_results: int, grounding_score: float = 0.0) -> str:
    """Assess confidence using grounding_score as primary metric.

    grounding_score is the composite metric from EvidenceValidator that already
    considers evidence quality, citation quality, conflict penalties, and coverage.
    top_score (raw retrieval score) is used only as a secondary signal.
    """
    if not grounded:
        return "insufficient"
    if grounding_score > 0.7 and num_results >= 3 and top_score > 0.3:
        return "high"
    if grounding_score > 0.5 and num_results >= 2:
        return "moderate"
    if grounding_score > 0.3 and num_results >= 1:
        return "low"
    return "low"
