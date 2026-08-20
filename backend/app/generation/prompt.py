"""Medical system prompt and prompt construction for grounded generation."""
SYSTEM_PROMPT = """You are a Diabetes Evidence Assistant — a clinical information retrieval system for diabetes diagnosis and classification.

You answer questions using ONLY the evidence provided below. You do NOT use external medical knowledge.

## STRICT RULES

### Evidence-Only
1. Answer ONLY from the evidence below. Every factual claim must cite its source using [Evidence N].
2. NEVER fabricate thresholds, values, test names, page numbers, section names, URLs, or DOIs.
3. Preserve all numerical values exactly as they appear in the evidence. Do not round, convert units, or modify thresholds.
4. If the evidence does not contain information to answer the question, say so. Do not guess.

### Population Distinction
5. Distinguish populations: general adults, pregnant women, children, elderly, high-risk groups.
6. Do NOT apply thresholds from one population to another (e.g., gestational diabetes thresholds ≠ general adult thresholds).

### Conflict Handling
7. If evidence from different sources conflicts (different thresholds, terminology, or values), present BOTH perspectives with source attribution.
8. If ADA and NIDDK use different terminology for the same concept, note both terms.

### Citation
9. Every clinical claim must cite: [Evidence N] where N is the evidence number.
10. Never invent citations. Only cite evidence actually provided.

### Refusal
11. Refuse when:
    a. No relevant evidence was retrieved
    b. Evidence is insufficient or contradictory for the specific question
    c. Question asks for personalized diagnosis, medication dosing, or treatment decisions
    d. Question is outside the scope of diabetes diagnosis and classification

### Safety
12. NEVER diagnose individual patients. Explain relevant criteria but state that diagnosis requires professional evaluation.
13. NEVER prescribe or recommend specific medications or doses.
14. For emergency symptoms, advise seeking urgent medical care immediately.

## ANSWER FORMAT

**Answer**
[Concise evidence-based answer with inline [Evidence N] citations]

**Evidence**
[Evidence 1] [Source] — [Section], p.[Page]
[Evidence 2] ...

**Sources**
[Numbered list of all unique sources used]

**Confidence**
[High / Moderate / Low / Insufficient] — brief justification

**Limitations**
[Any caveats, conflicting evidence, or missing information]"""


def build_grounded_prompt(
    query: str,
    evidence_pack,
    conflict_report=None,
) -> tuple[str, str]:
    """Build system prompt and user message for grounded generation.

    Uses the EvidencePack to format evidence with proper citation metadata.
    """
    evidence_text = evidence_pack.to_prompt_evidence()

    conflict_note = ""
    if conflict_report and conflict_report.total_conflicts > 0:
        conflict_note = "\n\n⚠️ CONFLICTS DETECTED in evidence:\n"
        for c in conflict_report.conflicts:
            conflict_note += f"  - [{c.conflict_type}] {c.description} (severity: {c.severity})\n"
        conflict_note += "Please present both perspectives with source attribution.\n"

    system = SYSTEM_PROMPT + conflict_note + "\n\nRETRIEVAL EVIDENCE:\n" + evidence_text
    user_message = (
        f"User question: {query}\n\n"
        "Answer based ONLY on the provided evidence. "
        "Cite every factual claim with [Evidence N]. "
        "Include all four sections: Answer, Evidence, Sources, Confidence."
    )

    return system, user_message


def classify_query(query: str) -> str:
    """Classify query type for routing."""
    q = query.lower()

    emergency_words = ["emergency", "urgent", "chest pain", "fainting", "severe", "dka", "coma", "anaphylaxis"]
    if any(w in q for w in emergency_words):
        return "emergency"

    personal_indicators = ["my a1c", "my fasting", "my result", "my test", "do i have", "am i diabetic", "am i prediabetic"]
    if any(ind in q for ind in personal_indicators):
        return "personal_medical_advice"

    medication_words = ["medication", "drug", "prescribe", "metformin", "insulin", "treatment", "dose", "should i take"]
    if any(w in q for w in medication_words):
        return "medication_question"

    threshold_words = ["threshold", "cutoff", "level", ">= ", "at least", "indicates", "criteria", "diagnostic value"]
    if any(w in q for w in threshold_words):
        return "threshold_question"

    comparison_words = ["difference", "compare", "versus", "vs", "better", "which test", "which is"]
    if any(w in q for w in comparison_words):
        return "comparison"

    return "factual_medical_question"


def build_refusal_response(query: str, reason: str = "insufficient_evidence", detail: str = "") -> str:
    """Build a professional refusal response.

    User-facing categories:
      A) no_evidence / low_relevance / low_grounding  → "Not enough information"
      B) verification_failed                          → "Answer could not be verified"
      C) provider_failure / llm_error                 → "Answer temporarily unavailable"
      D) emergency                                    → emergency message (unchanged)
      E) medical_advice                               → safety refusal (unchanged)
    """
    reason_messages = {
        "no_evidence": (
            "I couldn't find enough relevant information in the available clinical sources "
            "to answer this question safely.\n\n"
            "I don't want to guess or provide unsupported medical information. "
            "Please try rephrasing your question or ask about a topic covered by the "
            "available sources, such as diabetes diagnosis, testing, or classification."
        ),
        "low_relevance": (
            "I found some information, but it isn't strong or relevant enough to support "
            "a reliable answer.\n\n"
            "Please try asking about specific diagnostic tests like FPG, A1C, or OGTT, "
            "or about diabetes classification criteria."
        ),
        "verification_failed": (
            "I found relevant information, but I couldn't verify the generated answer "
            "against the clinical sources.\n\n"
            "I won't provide an unsupported medical answer. "
            "Please try rephrasing your question or ask about a more specific topic."
        ),
        "low_grounding": (
            "I found some information, but it isn't strong or relevant enough to support "
            "a reliable answer.\n\n"
            "Please try rephrasing your question or ask about a topic covered by the "
            "available sources, such as diabetes diagnosis, testing, or classification."
        ),
        "llm_error": (
            "I found relevant clinical evidence, but the answer-generation service "
            "encountered a technical issue.\n\n"
            "Please try again shortly."
        ),
        "provider_failure": (
            "I found relevant clinical evidence, but the answer-generation service "
            "is temporarily unavailable.\n\n"
            "No clinical answer was fabricated. Please try again shortly."
        ),
        "emergency": (
            "This appears to be an urgent medical situation. I cannot diagnose or provide emergency "
            "medical advice. **Please seek immediate medical attention or call emergency services.**\n\n"
            "If you need general information from the indexed medical sources, please rephrase your "
            "question without the emergency context."
        ),
        "medical_advice": (
            "This question asks for personalized medical advice, "
            "which requires evaluation by a qualified healthcare professional.\n\n"
            "I can provide general information about diabetes diagnostic criteria, "
            "testing methods, and classification from clinical guidelines instead."
        ),
        "insufficient_evidence": (
            "I couldn't find enough reliable information in the available clinical sources "
            "to answer this question safely.\n\n"
            "I don't want to guess or provide unsupported medical information. "
            "Please try rephrasing your question."
        ),
    }

    msg = reason_messages.get(reason, reason_messages["insufficient_evidence"])

    if detail and reason not in ("no_evidence", "insufficient_evidence", "emergency", "medical_advice"):
        msg += f"\n\n*Technical detail: {detail}*"

    msg += "\n\nIf you have a medical concern, please consult a qualified healthcare professional."
    return msg
