"""Deterministic query expansion for medical terminology."""
import re
import logging

logger = logging.getLogger(__name__)

# Medical terminology mappings (query → expanded terms)
MEDICAL_SYNONYMS = {
    # FPG variants
    "fasting blood sugar": "fasting plasma glucose FPG fasting blood glucose fasting glucose",
    "fasting glucose": "fasting plasma glucose FPG fasting blood sugar",
    "fasting blood glucose": "fasting plasma glucose FPG fasting glucose",
    "fasting test": "fasting plasma glucose FPG fasting blood glucose",
    "fasting blood test": "fasting plasma glucose FPG",
    "blood sugar fasting": "fasting plasma glucose FPG fasting glucose",

    # OGTT variants
    "glucose tolerance test": "oral glucose tolerance test OGTT 2-hour glucose",
    "oral glucose tolerance": "OGTT 2-hour glucose glucose tolerance test",
    "2 hour glucose test": "OGTT 2-hour glucose oral glucose tolerance test",
    "2-hour glucose": "OGTT 2-h PG oral glucose tolerance test",
    "glucose tolerance": "OGTT oral glucose tolerance test 2-hour glucose",

    # A1C variants
    "hemoglobin a1c": "A1C glycated hemoglobin HbA1c",
    "glycated hemoglobin": "A1C HbA1c hemoglobin a1c",
    "hba1c": "A1C glycated hemoglobin hemoglobin a1c",
    "a1c test": "A1C glycated hemoglobin HbA1c",
    "a1c level": "A1C glycated hemoglobin",
    "a1c percentage": "A1C glycated hemoglobin",
    "glycosylated hemoglobin": "A1C HbA1c glycated hemoglobin",

    # Random glucose
    "random blood sugar": "random plasma glucose RPG random blood glucose",
    "random glucose": "random plasma glucose RPG",
    "random blood glucose": "random plasma glucose RPG",

    # Diagnostic thresholds
    "prediabetes criteria": "prediabetes diagnosis criteria impaired fasting glucose IFG impaired glucose tolerance IGT",
    "prediabetes thresholds": "prediabetes diagnostic thresholds IFG IGT A1C 5.7-6.4",
    "diabetes thresholds": "diabetes diagnostic thresholds FPG OGTT A1C 6.5",
    "diagnostic criteria": "diagnostic criteria diagnosis classification A1C FPG OGTT",

    # Type 1 diabetes
    "autoantibodies": "autoantibodies type 1 diabetes IAA GADA IA-2A ZnT8A ICA",
    "type 1 diabetes antibodies": "autoantibodies type 1 diabetes IAA GADA IA-2A ZnT8A",
    "islet cell antibodies": "autoantibodies type 1 diabetes ICA GADA",

    # Gestational diabetes
    "gestational diabetes": "gestational diabetes mellitus GDM pregnancy diabetes",
    "pregnancy diabetes": "gestational diabetes mellitus GDM",
    "gdm": "gestational diabetes mellitus GDM pregnancy diabetes",

    # Monogenic diabetes
    "mody": "maturity-onset diabetes of the young MODY monogenic diabetes",
    "monogenic diabetes": "monogenic diabetes MODY neonatal diabetes",

    # CGM
    "continuous glucose monitoring": "CGM continuous glucose monitoring",
    "cgm": "continuous glucose monitoring CGM",

    # Confirmation
    "confirm diagnosis": "confirming diagnosis confirmation repeat test",
    "diagnosis confirmation": "confirming diagnosis confirmation repeat test",
}

# Source detection keywords
SOURCE_KEYWORDS = {
    "niddk": [
        "niddk", "national institute of diabetes", "nih",
        "compared to", "comparison", "pros and cons",
        "sensitivity of", "coefficient of variation",
        "technical features",
    ],
    "ada": [
        "ada", "american diabetes association", "standards of care",
        "diagnostic criteria", "screening", "classification",
        "type 1", "type 2", "gestational", "monogenic",
        "prediabetes", "confirming",
    ],
}

# Section keywords for boosting
SECTION_KEYWORDS = {
    "Screening and Diagnosis of Diabetes": [
        "diagnostic criteria", "screening", "diagnosis",
        "threshold", "cutoff", "level", "mg/dl", "mmol/l",
        "a1c", "fpg", "ogtt", "fasting", "glucose",
    ],
    "Type 1 Diabetes": [
        "type 1", "autoantibodies", "autoimmune",
        "autoimmunity", "islet", "beta cell",
    ],
    "Gestational Diabetes Mellitus": [
        "gestational", "pregnancy", "gdm", "pregnant",
        "prenatal", "maternal",
    ],
    "Monogenic Diabetes Syndromes": [
        "monogenic", "mody", "neonatal", "genetic",
        "kcnj11", "abcc8", "hnf1a", "gck",
    ],
    "Diagnosis of Prediabetes": [
        "prediabetes", "ifg", "igt", "impaired fasting",
        "impaired glucose", "5.7", "6.4",
    ],
    "Comparing Diabetes Blood Tests": [
        "comparison", "pros", "cons", "sensitivity",
        "specificity", "coefficient of variation",
        "cost", "convenience",
    ],
}


def expand_query(query: str) -> str:
    """Expand query with medical synonyms.

    Returns expanded query with additional terms for better retrieval.
    No LLM used - purely deterministic keyword mapping.
    """
    query_lower = query.lower().strip()
    expansions = []

    for key, expansion in MEDICAL_SYNONYMS.items():
        if key in query_lower:
            expansions.append(expansion)

    if expansions:
        expanded = query + " " + " ".join(expansions)
        logger.debug(f"Query expanded: '{query}' -> '{expanded[:100]}...'")
        return expanded

    return query


def detect_source_hint(query: str) -> str:
    """Detect if query hints at a specific source.

    Returns 'niddk', 'ada', or '' (both).
    """
    query_lower = query.lower()

    niddk_score = sum(1 for kw in SOURCE_KEYWORDS["niddk"] if kw in query_lower)
    ada_score = sum(1 for kw in SOURCE_KEYWORDS["ada"] if kw in query_lower)

    if niddk_score > ada_score:
        return "niddk"
    elif ada_score > niddk_score:
        return "ada"
    return ""


def detect_section_hint(query: str) -> list[str]:
    """Detect which sections are most relevant to the query.

    Returns list of section names to boost.
    """
    query_lower = query.lower()
    scores = {}

    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[section] = score

    if not scores:
        return []

    max_score = max(scores.values())
    boosted = [s for s, sc in scores.items() if sc >= max_score * 0.5]

    logger.debug(f"Section hints for '{query[:50]}': {boosted}")
    return boosted
