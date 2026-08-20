"""Clinical safety rules for the diabetes RAG system."""
import re
import logging

logger = logging.getLogger(__name__)

HIGH_RISK_PATTERNS = {
    "diagnosis_request": [
        r"\bdo\s+i\s+have\b",
        r"\bam\s+i\s+(?:a\s+)?(?:diabetic|prediabetic)\b",
        r"\bcan\s+you\s+diagnose\b",
        r"\bwhat\s+do\s+my\s+results?\s+mean\b",
    ],
    "medication_dosing": [
        r"\bwhat\s+dose\b",
        r"\bhow\s+much\s+(?:metformin|insulin|medication)\b",
        r"\bshould\s+i\s+take\b.*\b(?:metformin|insulin|medication)\b",
        r"\bprescribe\b",
    ],
    "treatment_change": [
        r"\bshould\s+i\s+(?:stop|start|change)\b.*\b(?:insulin|metformin|medication)\b",
        r"\bcan\s+i\s+(?:stop|discontinue)\b",
    ],
    "emergency_symptoms": [
        r"\bsevere\s+hypoglycemia\b",
        r"\bdiabetic\s+ketoacidosis\b",
        r"\bdka\b",
        r"\bhh\s*s\b",
        r"\bcoma\b",
        r"\bunconscious\b",
        r"\bchest\s+pain\b",
        r"\btrouble\s+breathing\b",
    ],
}


def check_safety(query: str, answer_text: str, query_type: str) -> dict:
    """Run safety checks on the query and generated answer.

    Returns safety assessment with risk level and recommendations.
    """
    risk_flags = []
    risk_level = "low"
    requires_professional = False

    q_lower = query.lower()

    for category, patterns in HIGH_RISK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower):
                risk_flags.append(category)
                break

    if query_type == "emergency":
        risk_level = "high"
        requires_professional = True
    elif "diagnosis_request" in risk_flags or "medication_dosing" in risk_flags:
        risk_level = "high"
        requires_professional = True
    elif "treatment_change" in risk_flags:
        risk_level = "medium"
        requires_professional = True
    elif "emergency_symptoms" in risk_flags:
        risk_level = "high"
        requires_professional = True

    if "I'm sorry" in answer_text and risk_level == "low":
        risk_level = "low"

    return {
        "risk_flags": risk_flags,
        "risk_level": risk_level,
        "requires_professional": requires_professional,
        "has_safety_note": "professional" in answer_text.lower() or "healthcare" in answer_text.lower(),
    }
