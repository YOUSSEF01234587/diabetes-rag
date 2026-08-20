"""Phase 3: Intent-aware query analysis for medical RAG."""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Structured representation of query intent."""
    primary_intent: str
    confidence: float
    all_intents: list = field(default_factory=list)
    preferred_sections: list = field(default_factory=list)
    preferred_sources: list = field(default_factory=list)
    excluded_sections: list = field(default_factory=list)
    required_concepts: list = field(default_factory=list)
    is_table_query: bool = False
    is_comparison: bool = False
    is_threshold_query: bool = False
    is_confirmation_query: bool = False
    topic_keywords: list = field(default_factory=list)


# Intent detection patterns with (pattern, intent, sections, sources, concepts)
INTENT_PATTERNS = [
    # Diagnostic criteria
    (r"\b(diagnostic criteria|diagnos(is|e|ing) (diabetes|prediabetes))\b",
     "diagnostic_criteria",
     ["Screening and Diagnosis of Diabetes", "Diagnosis of Prediabetes"],
     ["ada_soc_2026_diagnosis"],
     ["A1C", "FPG", "OGTT", "threshold"]),
    (r"\b(confirm(ation|ing|ed)?|repeat test|two abnormal)\b",
     "diagnosis_confirmation",
     ["Confirming the Diagnosis"],
     ["ada_soc_2026_diagnosis"],
     ["two abnormal results", "different tests", "repeat"]),
    (r"\b(prediabetes|impaired fasting glucose|impaired glucose tolerance|IFG|IGT)\b",
     "prediabetes_criteria",
     ["Diagnosis of Prediabetes"],
     ["ada_soc_2026_diagnosis"],
     ["100-125", "140-199", "5.7-6.4"]),

    # Specific tests
    (r"\b(A1C|HbA1c|hemoglobin a1c|glycated hemoglobin|glycosylated)\b",
     "a1c",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis"],
     ["6.5%", "5.7-6.4%", "NGSP", "DCCT"]),
    (r"\b(A1C.*(threshold|level|criteria|diagnos))\b",
     "diagnostic_criteria",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis"],
     ["6.5%", "A1C", "threshold"]),
    (r"\b(fasting plasma glucose|FPG|fasting blood (sugar|glucose)|fasting test)\b",
     "fpg",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
     ["126", "100-125", "fasting", "8 hours"]),
    (r"\b(OGTT|oral glucose tolerance|2.hour (PG|glucose)|glucose tolerance test)\b",
     "ogtt",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
     ["200", "140-199", "75 grams", "WHO"]),
    (r"\b(random plasma glucose|RPG|random blood (sugar|glucose)|random glucose)\b",
     "random_glucose",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis"],
     ["200", "symptoms", "hyperglycemia"]),

    # Test comparison/interference
    (r"\b(pros and cons|advantages|disadvantages|comparison|compar(e|ing|ed)|compared to|versus|vs\.?|sensitivity of|differences between|differ from)\b",
     "test_comparison",
     ["Comparing Diabetes Blood Tests"],
     ["niddk_diabetes_prediabetes_tests"],
     ["sensitivity", "specificity", "cost", "convenience"]),
    (r"\b(interfere|interference|affect(s|ed)? (result|test)|hemoglobin variant|G6PD|sickle)\b",
     "test_interference",
     ["Screening and Diagnosis of Diabetes"],
     ["ada_soc_2026_diagnosis"],
     ["variants", "interference", "G6PD", "sickle cell"]),

    # Diabetes types
    (r"\b(type 1 diabetes|autoantibod|autoimmun|stage[sd]? of type 1|IAA|GADA|IA-2A|ZnT8)\b",
     "type1",
     ["Type 1 Diabetes"],
     ["ada_soc_2026_diagnosis"],
     ["autoantibodies", "staging", "autoimmune"]),
    (r"\b(type 2 diabetes|risk factor|screening for type 2)\b",
     "type2",
     ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"],
     ["ada_soc_2026_diagnosis"],
     ["risk factors", "screening"]),
    (r"\b(gestational diabetes|GDM|pregnancy diabetes|pregnant)\b",
     "gestational",
     ["Gestational Diabetes Mellitus"],
     ["ada_soc_2026_diagnosis"],
     ["75-g OGTT", "24-28 weeks", "IADPSG"]),
    (r"\b(monogenic diabetes|MODY|neonatal diabetes|HNF1A|GCK|KCNJ11)\b",
     "monogenic",
     ["Monogenic Diabetes Syndromes"],
     ["ada_soc_2026_diagnosis"],
     ["MODY", "genetic testing"]),

    # Screening
    (r"\b(screen(ing|ed)?|how often|frequency|when to (test|screen|check))\b",
     "screening",
     ["Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"],
     ["ada_soc_2026_diagnosis"],
     ["every 3 years", "risk factors"]),

    # Threshold queries
    (r"\b(what (is|are) the? (threshold|level|cutoff|range|value) (for|of|that|indicate|indicates)?)\b",
     "diagnostic_criteria",
     ["Screening and Diagnosis of Diabetes", "Diagnosis of Prediabetes"],
     ["ada_soc_2026_diagnosis"],
     ["threshold", "level", "cutoff"]),

    # Monogenic diagnosis (fix: detect "monogenic" in any context)
    (r"\b(monogenic|how.*monogenic|monogenic.*diagnos)\b",
     "monogenic",
     ["Monogenic Diabetes Syndromes"],
     ["ada_soc_2026_diagnosis"],
     ["monogenic", "MODY", "genetic testing"]),

    # Table queries
    (r"\b(Table \d+\.\d+|criteria table|diagnostic table|comparison table)\b",
     "table_lookup",
     ["Screening and Diagnosis of Diabetes", "Diagnosis of Prediabetes"],
     ["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
     ["table"]),

    # Special populations
    (r"\b(cystic fibrosis|HIV|special population|pregnan(cy|t) specific)\b",
     "special_population",
     ["Diagnosis of Prediabetes", "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults"],
     ["ada_soc_2026_diagnosis"],
     []),

    # Unsupported/out-of-scope
    (r"\b(medication|drug|metformin|insulin treatment|treatment plan|dosing)\b",
     "unsupported_medication",
     [],
     [],
     []),
    (r"\b(exercise|diet|nutrition|weight loss|lifestyle|intervention|what (should|can) I eat)\b",
     "unsupported_lifestyle",
     [],
     [],
     []),
    (r"\b(cancer|heart disease|hypertension treatment|cholesterol)\b",
     "unsupported_other_condition",
     [],
     [],
     []),
]


# Section-to-topic mapping for topic filtering
SECTION_TOPICS = {
    "Screening and Diagnosis of Diabetes": ["diagnostic_criteria", "fpg", "ogtt", "a1c", "random_glucose", "diagnostic"],
    "Confirming the Diagnosis": ["confirmation", "diagnostic_criteria"],
    "Diagnostic Tests for Diabetes": ["diagnostic_criteria", "technical"],
    "Type 1 Diabetes": ["type1", "autoantibodies", "autoimmune"],
    "Diagnosis of Prediabetes": ["prediabetes", "ifg", "igt"],
    "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults": ["screening", "type2", "risk"],
    "Gestational Diabetes Mellitus": ["gestational", "pregnancy", "gdm"],
    "Monogenic Diabetes Syndromes": ["monogenic", "mody", "genetic"],
    "Comparing Diabetes Blood Tests": ["test_comparison", "sensitivity", "pros", "cons"],
}


def detect_intent(query: str) -> QueryIntent:
    """Detect the primary intent and metadata for a query.
    
    Returns structured QueryIntent with:
    - primary_intent: the main intent category
    - confidence: how confident we are (0-1)
    - all_intents: all detected intents with scores
    - preferred_sections: sections to boost
    - preferred_sources: sources to boost
    - excluded_sections: sections to penalize
    - required_concepts: key concepts to look for
    - is_table_query: whether the query asks about tables
    - is_comparison: whether it's a comparison query
    - is_threshold_query: whether it asks for specific thresholds
    - is_confirmation_query: whether it asks about confirmation
    - topic_keywords: keywords for topic matching
    """
    query_lower = query.lower().strip()
    
    intent_scores = {}
    all_sections = []
    all_sources = []
    all_concepts = []
    is_table = False
    is_comparison = False
    is_threshold = False
    is_confirmation = False
    topic_keywords = []
    
    for pattern, intent, sections, sources, concepts in INTENT_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            # Comparison/meta intents get extra weight to beat individual test intents
            weight = 2 if intent in ("test_comparison", "test_interference", "diagnostic_criteria", "diagnosis_confirmation", "prediabetes_criteria") else 1
            intent_scores[intent] = intent_scores.get(intent, 0) + weight
            all_sections.extend(sections)
            all_sources.extend(sources)
            all_concepts.extend(concepts)
            
            if intent == "table_lookup":
                is_table = True
            if intent == "test_comparison":
                is_comparison = True
            if any(t in query_lower for t in ["threshold", "level", "cutoff", ">=", ">=", "≥"]):
                is_threshold = True
            if intent == "diagnosis_confirmation":
                is_confirmation = True
    
    # Check for threshold queries even if not explicitly tagged
    if re.search(r"\b(\d+\.?\d*\s*(mg/dL|mmol/L|%))\b", query_lower):
        is_threshold = True
    if re.search(r"\b(what (is|are) the? (threshold|level|cutoff|criteria|range))\b", query_lower):
        is_threshold = True
    
    if not intent_scores:
        primary_intent = "general"
        confidence = 0.3
    else:
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(1.0, 0.5 + 0.2 * intent_scores[primary_intent])
    
    # Collect topic keywords
    for intent_name, score in intent_scores.items():
        for _, _, _, _, concepts in INTENT_PATTERNS:
            pass
    topic_keywords = list(set(all_concepts[:10]))
    
    # Determine preferred sections based on primary intent
    preferred_sections = []
    preferred_sources = []
    
    if intent_scores:
        for pattern, intent, sections, sources, concepts in INTENT_PATTERNS:
            if intent == primary_intent:
                preferred_sections.extend(sections)
                preferred_sources.extend(sources)
                break
    
    # Deduplicate
    preferred_sections = list(dict.fromkeys(preferred_sections))
    preferred_sources = list(dict.fromkeys(preferred_sources))
    all_sections = list(dict.fromkeys(all_sections))
    all_sources = list(dict.fromkeys(all_sources))
    
    return QueryIntent(
        primary_intent=primary_intent,
        confidence=confidence,
        all_intents=[(intent, score) for intent, score in sorted(intent_scores.items(), key=lambda x: -x[1])],
        preferred_sections=preferred_sections or all_sections,
        preferred_sources=preferred_sources or all_sources,
        excluded_sections=[],
        required_concepts=list(dict.fromkeys(all_concepts)),
        is_table_query=is_table,
        is_comparison=is_comparison,
        is_threshold_query=is_threshold,
        is_confirmation_query=is_confirmation,
        topic_keywords=topic_keywords,
    )


def get_section_boost(section: str, intent: QueryIntent) -> float:
    """Calculate section boost based on intent match."""
    if not intent.preferred_sections:
        return 0.0
    
    if section in intent.preferred_sections:
        return 0.25
    
    for preferred in intent.preferred_sections:
        if preferred.lower() in section.lower() or section.lower() in preferred.lower():
            return 0.20
    
    return 0.0


def get_source_boost(source_id: str, intent: QueryIntent) -> float:
    """Calculate source boost based on intent match."""
    if not intent.preferred_sources:
        return 0.0
    
    if source_id in intent.preferred_sources:
        return 0.20
    
    return 0.0


def get_topic_penalty(section: str, intent: QueryIntent) -> float:
    """Calculate penalty for off-topic sections.
    
    For example, if the query is about general diabetes diagnosis,
    gestational diabetes sections should be penalized.
    """
    intent_name = intent.primary_intent
    
    # Penalize type 1 sections for type 2/gestational/monogenic queries
    if intent_name in ("type2", "gestational", "prediabetes_criteria", "monogenic"):
        if "Type 1 Diabetes" in section:
            return -0.12
    
    # Penalize gestational sections for non-gestational queries
    if intent_name not in ("gestational",):
        if "Gestational" in section:
            return -0.12
    
    # Penalize monogenic sections for non-monogenic queries
    if intent_name not in ("monogenic",):
        if "Monogenic" in section:
            return -0.12
    
    # Penalize general diagnostic sections for specific type queries
    if intent_name in ("type1", "type2", "gestational", "monogenic"):
        if section in ("Screening and Diagnosis of Diabetes", "Confirming the Diagnosis", "Diagnosis of Prediabetes"):
            return -0.08
    
    # Penalize specific-type sections for general queries
    if intent_name in ("diagnostic_criteria", "a1c", "fpg", "ogtt", "random_glucose"):
        if "Type 1 Diabetes" in section or "Gestational" in section or "Monogenic" in section:
            return -0.08
    
    # Penalize comparison sections for non-comparison queries
    if intent_name not in ("test_comparison", "test_interference"):
        if "Comparing" in section:
            return -0.08
    
    return 0.0
