"""Layered intent classifier for medical RAG (v2).

Architecture:
  Layer 1 – High-precision regex rules (exact medical-term matches, highest weight)
  Layer 2 – Fuzzy medical terminology matching (keyword overlap with tolerance)
  Layer 3 – Semantic intent matching (profile-based scoring)
  Layer 4 – Fallback general classification

The classifier is backward-compatible with the QueryIntent dataclass from
query_intent.py so existing consumers can switch incrementally.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IntentProfile:
    """Metadata that defines a single intent category."""
    intent_name: str
    description: str
    positive_keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    target_sections: list[str] = field(default_factory=list)
    target_sources: list[str] = field(default_factory=list)
    priority_weight: float = 1.0
    # Layer-1 exact patterns (compiled at module load for speed)
    _layer1_patterns: list[re.Pattern] = field(default_factory=list, repr=False)
    # Layer-2 fuzzy stems / sub-strings
    _layer2_stems: list[str] = field(default_factory=list, repr=False)


@dataclass
class IntentResult:
    """Classification output – drop-in replacement for QueryIntent."""
    primary_intent: str
    confidence: float
    all_intents: list[tuple[str, float]] = field(default_factory=list)
    preferred_sections: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    excluded_sections: list[str] = field(default_factory=list)
    required_concepts: list[str] = field(default_factory=list)
    is_table_query: bool = False
    is_comparison: bool = False
    is_threshold_query: bool = False
    is_confirmation_query: bool = False
    topic_keywords: list[str] = field(default_factory=list)

    # --- backward-compatible alias properties (optional) ---
    @property
    def primary_intent_name(self) -> str:
        return self.primary_intent


# ---------------------------------------------------------------------------
# Intent registry – define every supported intent
# ---------------------------------------------------------------------------

def _build_intent_profiles() -> dict[str, IntentProfile]:
    """Construct the full intent registry with compiled patterns."""
    profiles: dict[str, IntentProfile] = {}

    def _add(
        name: str,
        desc: str,
        pos: list[str],
        neg: list[str],
        sections: list[str],
        sources: list[str],
        weight: float = 1.0,
        layer1: list[str] | None = None,
        layer2: list[str] | None = None,
    ) -> None:
        p = IntentProfile(
            intent_name=name,
            description=desc,
            positive_keywords=pos,
            negative_keywords=neg,
            target_sections=sections,
            target_sources=sources,
            priority_weight=weight,
        )
        if layer1:
            p._layer1_patterns = [re.compile(r, re.IGNORECASE) for r in layer1]
        if layer2:
            p._layer2_stems = layer2
        profiles[name] = p

    # ---- diagnostic_threshold ----
    _add(
        name="diagnostic_threshold",
        desc="Asking about numeric thresholds that define diabetes diagnosis",
        pos=[
            "diagnostic criteria", "diagnostic threshold", "diagnostic level",
            "diagnostic cutoff", "diagnos", "classify", "classification",
            "≥ 6.5", ">= 6.5", "126 mg/dl", "200 mg/dl",
            "≥ 126", "≥ 200", "threshold", "cutoff", "level", "value",
            "what is the", "what are the", "what level",
            "indicate", "indicates", "define", "defines",
        ],
        neg=[
            "prediabetes", "impaired fasting", "impaired glucose",
            "5.7", "6.4", "IFG", "IGT",
            "type 1", "type 2", "gestational", "monogenic",
        ],
        sections=[
            "Screening and Diagnosis of Diabetes",
            "Diagnosis of Prediabetes",
        ],
        sources=["ada_soc_2026_diagnosis"],
        weight=2.0,
        layer1=[
            r"\b(diagnostic\s+(criteria|threshold|level|cutoff|value|range|test))\b",
            r"\b(diagnos(is|e|ing)\s+(diabetes|prediabetes))\b",
            r"\bwhat\s+(is|are)\s+the?\s+(threshold|level|cutoff|criteria|range|value)\b",
            r"\b(≥|>=)\s*(6\.5|126|200)\b",
            r"\b(threshold|cutoff)\s+(for|of|that)\s+diabet",
        ],
        layer2=[
            "diagnos", "threshold", "cutoff", "criteria", "classify",
            "level", "value", "range",
        ],
    )

    # ---- prediabetes_criteria ----
    _add(
        name="prediabetes_criteria",
        desc="Prediabetes diagnostic thresholds and definitions",
        pos=[
            "prediabetes", "pre-diabetes", "impaired fasting glucose",
            "impaired glucose tolerance", "IFG", "IGT",
            "5.7", "5.7%", "6.4", "6.4%", "5.7-6.4",
            "100-125", "140-199",
            "borderline", "at risk",
        ],
        neg=[
            "type 1", "type 2", "gestational", "monogenic",
            "treatment", "medication", "insulin",
        ],
        sections=["Diagnosis of Prediabetes"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.8,
        layer1=[
            r"\b(prediabet(es|ic)|pre-diabet(es|ic))\b",
            r"\b(impaired\s+fasting\s+glucose|IFG)\b",
            r"\b(impaired\s+glucose\s+tolerance|IGT)\b",
            r"\b5[\.\s]*7\s*[-–to]+\s*6[\.\s]*4\b",
            r"\b100\s*[-–to]+\s*125\s*(mg/dL|mg\s*/?\s*dL)?\b",
            r"\b140\s*[-–to]+\s*199\s*(mg/dL|mg\s*/?\s*dL)?\b",
        ],
        layer2=[
            "prediabet", "impaired", "fasting", "IFG", "IGT",
            "borderline", "5.7", "6.4", "100-125", "140-199",
        ],
    )

    # ---- confirmation ----
    _add(
        name="confirmation",
        desc="Repeat testing / two different tests to confirm diagnosis",
        pos=[
            "confirm", "confirmation", "confirmed", "retest",
            "repeat test", "repeat testing", "two abnormal",
            "two tests", "different test", "second test",
            "reconfirm", "verify",
        ],
        neg=[
            "first time", "initial", "screening",
        ],
        sections=["Confirming the Diagnosis"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.8,
        layer1=[
            r"\b(confirm(ation|ing|ed)?)\b",
            r"\brepeat\s+(test|testing|measure|measurement)\b",
            r"\btwo\s+(different\s+)?(abnormal\s+)?results?\b",
            r"\b(second|re-?test|retest)\b",
        ],
        layer2=[
            "confirm", "repeat", "retest", "verify", "second",
        ],
    )

    # ---- screening ----
    _add(
        name="screening",
        desc="When to test / who should be screened / screening frequency",
        pos=[
            "screening", "screen", "when to test", "when to screen",
            "who should", "how often", "frequency", "routine",
            "asymptomatic", "risk factor", "risk assessment",
            "every 3 years", "every year",
        ],
        neg=[
            "diagnostic", "diagnosis", "confirm",
            "treatment", "medication",
        ],
        sections=[
            "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
        ],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.5,
        layer1=[
            r"\b(screen(ing|ed)?)\b",
            r"\bwhen\s+to\s+(test|screen|check)\b",
            r"\bwho\s+(should|needs?|must)\b",
            r"\bhow\s+often\s+(should|do|to)\b",
            r"\b(every\s+\d+\s+years?)\b",
            r"\basymptomatic\s+adult",
        ],
        layer2=[
            "screen", "when to test", "who should", "how often",
            "frequency", "routine", "asymptomatic",
        ],
    )

    # ---- a1c_test ----
    _add(
        name="a1c_test",
        desc="A1C-specific questions (what it measures, accuracy, NGSP)",
        pos=[
            "a1c", "hba1c", "hemoglobin a1c", "glycated hemoglobin",
            "glycosylated hemoglobin", "ngsp", "dcct",
            "a1c test", "a1c level", "a1c percentage",
        ],
        neg=[
            "fpg", "fasting", "ogtt", "random",
            "type 1", "type 2", "gestational",
        ],
        sections=["Screening and Diagnosis of Diabetes"],
        sources=["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
        weight=1.5,
        layer1=[
            r"\b(A1C|HbA1c|hemoglobin\s+a1c|glycated\s+hemoglobin|glycosylated)\b",
        ],
        layer2=[
            "a1c", "hba1c", "hemoglobin", "glycated", "glycosylated",
        ],
    )

    # ---- fpg_test ----
    _add(
        name="fpg_test",
        desc="Fasting plasma glucose specific questions",
        pos=[
            "fpg", "fasting plasma glucose", "fasting blood sugar",
            "fasting blood glucose", "fasting glucose",
            "fasting test", "fasting level",
            "8 hours", "overnight fast",
        ],
        neg=[
            "a1c", "ogtt", "random", "non-fasting",
        ],
        sections=["Screening and Diagnosis of Diabetes"],
        sources=["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
        weight=1.5,
        layer1=[
            r"\b(fasting\s+(plasma\s+glucose|blood\s+(sugar|glucose)|glucose)|FPG)\b",
            r"\bfasting\s+test\b",
        ],
        layer2=[
            "fpg", "fasting", "fast", "8 hours", "overnight",
        ],
    )

    # ---- ogtt_test ----
    _add(
        name="ogtt_test",
        desc="Oral glucose tolerance test specific questions",
        pos=[
            "ogtt", "oral glucose tolerance", "glucose tolerance test",
            "2-hour glucose", "2 hour glucose", "2-h pg",
            "75 grams", "75g", "who", "iadpsg",
        ],
        neg=[
            "a1c", "fpg", "fasting", "random",
        ],
        sections=["Screening and Diagnosis of Diabetes"],
        sources=["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
        weight=1.5,
        layer1=[
            r"\b(OGTT|oral\s+glucose\s+tolerance|2[\s-]?hour\s+(PG|glucose)|glucose\s+tolerance\s+test)\b",
        ],
        layer2=[
            "ogtt", "oral glucose tolerance", "glucose tolerance",
            "2-hour", "75 grams", "75g",
        ],
    )

    # ---- random_glucose ----
    _add(
        name="random_glucose",
        desc="Random plasma glucose specific questions",
        pos=[
            "random plasma glucose", "random blood sugar",
            "random blood glucose", "random glucose", "rpg",
            "symptoms", "hyperglycemia",
        ],
        neg=[
            "a1c", "fpg", "fasting", "ogtt",
        ],
        sections=["Screening and Diagnosis of Diabetes"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.3,
        layer1=[
            r"\b(random\s+(plasma\s+glucose|blood\s+(sugar|glucose)|glucose)|RPG)\b",
        ],
        layer2=[
            "random", "rpg", "symptom",
        ],
    )

    # ---- test_comparison ----
    _add(
        name="test_comparison",
        desc="Comparing tests: pros/cons, sensitivity, specificity",
        pos=[
            "pros and cons", "advantages", "disadvantages",
            "comparison", "compare", "comparing", "compared",
            "versus", "vs", "differences between", "differ from",
            "sensitivity", "specificity", "which test",
            "which is better", "best test",
        ],
        neg=[
            "diagnostic criteria", "threshold", "cutoff",
        ],
        sections=["Comparing Diabetes Blood Tests"],
        sources=["niddk_diabetes_prediabetes_tests"],
        weight=2.0,
        layer1=[
            r"\b(pros\s+and\s+cons|advantages?|disadvantages?)\b",
            r"\b(compar(e|ing|ed|ison)|versus|vs\.?)\b",
            r"\b(sensitivity\s+of|specificity\s+of)\b",
            r"\b(differences?\s+between|differ\s+from)\b",
            r"\bwhich\s+(is\s+)?(better|test|one)\b",
        ],
        layer2=[
            "compar", "pros", "cons", "versus", "sensitivity",
            "specificity", "which test", "which is better",
        ],
    )

    # ---- test_interference ----
    _add(
        name="test_interference",
        desc="Conditions that interfere with test accuracy",
        pos=[
            "interfere", "interference", "affect result",
            "affect test", "hemoglobin variant", "g6pd",
            "sickle", "accuracy", "unreliable",
            "false", "falsely", "conditions affecting",
        ],
        neg=[
            "diagnostic criteria", "threshold", "screening",
        ],
        sections=["Screening and Diagnosis of Diabetes"],
        sources=["ada_soc_2026_diagnosis", "niddk_diabetes_prediabetes_tests"],
        weight=2.0,
        layer1=[
            r"\b(interfer(e|ence|ing))\b",
            r"\b(affect(s|ed)?\s+(result|test|accuracy|level))\b",
            r"\b(hemoglobin\s+variant|G6PD|sickle\s+cell)\b",
            r"\b(false(ly)?\s+(high|low|elevated|elevat))\b",
        ],
        layer2=[
            "interfer", "inaccurat", "unreliable", "false",
            "g6pd", "sickle", "hemoglobin variant",
        ],
    )

    # ---- type1_diabetes ----
    _add(
        name="type1_diabetes",
        desc="Type 1 diabetes: autoantibodies, autoimmune, staging",
        pos=[
            "type 1", "type 1 diabetes", "t1d",
            "autoantibodies", "autoantibody", "autoimmune",
            "iaa", "gada", "ia-2a", "znt8", "ica",
            "islet cell", "staging", "stage 1", "stage 2", "stage 3",
            "beta cell", "beta-cell",
        ],
        neg=[
            "type 2", "gestational", "monogenic", "mody",
            "risk factor", "screening",
        ],
        sections=["Type 1 Diabetes"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.8,
        layer1=[
            r"\b(type\s*1\s*(diabetes|t1d)?)\b",
            r"\b(autoantibod(y|ies))\b",
            r"\b(autoimmune\s+diabetes)\b",
            r"\b(IAA|GADA|IA-2A|ZnT8|ICA)\b",
            r"\b(stage[sd]?\s+(of\s+)?type\s*1)\b",
            r"\b(islet\s+cell\s+antibod)",
        ],
        layer2=[
            "type 1", "autoantibod", "autoimmun", "iaa", "gada",
            "ia-2a", "znt8", "islet", "staging", "beta cell",
        ],
    )

    # ---- type2_diabetes ----
    _add(
        name="type2_diabetes",
        desc="Type 2 diabetes: risk factors, screening, classification",
        pos=[
            "type 2", "type 2 diabetes", "t2d",
            "risk factor", "risk factors",
            "screening for type 2", "insulin resistance",
        ],
        neg=[
            "type 1", "gestational", "monogenic", "mody",
            "autoantibodies", "autoimmune",
        ],
        sections=[
            "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
        ],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.5,
        layer1=[
            r"\b(type\s*2\s*(diabetes|t2d)?)\b",
            r"\b(risk\s+factors?\s+(for|of|associated))\b",
            r"\b(screening\s+for\s+type\s*2)\b",
            r"\binsulin\s+resistance\b",
        ],
        layer2=[
            "type 2", "risk factor", "insulin resistance",
        ],
    )

    # ---- gestational_diabetes ----
    _add(
        name="gestational_diabetes",
        desc="Gestational diabetes: GDM, pregnancy, prenatal",
        pos=[
            "gestational diabetes", "gdm", "gestational",
            "pregnancy diabetes", "pregnant", "pregnancy",
            "prenatal", "maternal", "24-28 weeks",
            "iadpsg", "75-g ogtt",
        ],
        neg=[
            "type 1", "type 2", "monogenic", "mody",
            "non-pregnant", "men",
        ],
        sections=["Gestational Diabetes Mellitus"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.8,
        layer1=[
            r"\b(gestational\s+diabetes|GDM)\b",
            r"\b(pregnancy\s+diabetes|pregnant)\b",
            r"\b(prenatal\s+glucose|maternal\s+glucose)\b",
        ],
        layer2=[
            "gestational", "pregnan", "gdm", "prenatal",
            "maternal",
        ],
    )

    # ---- monogenic_diabetes ----
    _add(
        name="monogenic_diabetes",
        desc="Monogenic diabetes: MODY, neonatal, genetic testing",
        pos=[
            "monogenic", "mody", "neonatal diabetes",
            "hnf1a", "gck", "kcnj11", "abcc8",
            "genetic testing", "genetic form",
        ],
        neg=[
            "type 1", "type 2", "gestational",
            "common", "prevalent",
        ],
        sections=["Monogenic Diabetes Syndromes"],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.8,
        layer1=[
            r"\b(monogenic\s+diabetes|monogenic)\b",
            r"\b(MODY|maturity.onset\s+diabetes\s+of\s+the\s+young)\b",
            r"\b(neonatal\s+diabetes)\b",
            r"\b(HNF1A|GCK|KCNJ11|ABCC8)\b",
        ],
        layer2=[
            "monogenic", "mody", "neonatal", "genetic",
            "hnf1a", "gck", "kcnj11",
        ],
    )

    # ---- special_population ----
    _add(
        name="special_population",
        desc="Special populations: cystic fibrosis, HIV, specific groups",
        pos=[
            "cystic fibrosis", "cf-related", "hiv",
            "special population", "special group",
            "transplant", "organ transplant",
            "corticosteroid", "atypical antipsychotic",
        ],
        neg=[
            "general population", "adult", "pediatric",
        ],
        sections=[
            "Screening and Diagnosis of Diabetes",
            "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
        ],
        sources=["ada_soc_2026_diagnosis"],
        weight=1.6,
        layer1=[
            r"\b(cystic\s+fibrosis|CF.related)\b",
            r"\b(HIV.*diabetes|diabetes.*HIV)\b",
            r"\b(special\s+populat)\b",
            r"\b(transplant\s+(patient|recipient))\b",
        ],
        layer2=[
            "cystic fibrosis", "hiv", "special pop", "transplant",
            "corticosteroid",
        ],
    )

    # ---- cgm ----
    _add(
        name="cgm",
        desc="Continuous glucose monitoring questions",
        pos=[
            "cgm", "continuous glucose monitoring",
            "continuous glucose", "glucose monitor",
            "glucose sensor", "dexcom", "freestyle libre",
            "flash glucose",
        ],
        neg=[
            "diagnostic", "threshold", "a1c", "fpg", "ogtt",
        ],
        sections=[],
        sources=[],
        weight=1.2,
        layer1=[
            r"\b(CGM|continuous\s+glucose\s+monitor)\b",
            r"\b(glucose\s+(monitor|sensor))\b",
            r"\b(dexcom|freestyle\s+libre|flash\s+glucose)\b",
        ],
        layer2=[
            "cgm", "continuous glucose", "glucose monitor",
            "glucose sensor",
        ],
    )

    # ---- source_specific ----
    _add(
        name="source_specific",
        desc="Explicitly mentions a specific source (NIDDK, ADA)",
        pos=[
            "niddk", "national institute of diabetes",
            "nih", "ada", "american diabetes association",
            "standards of care", "source", "reference",
        ],
        neg=[],
        sections=[],
        sources=[],
        weight=1.2,
        layer1=[
            r"\b(NIDDK|national\s+institute\s+of\s+diabetes)\b",
            r"\b(ADA|american\s+diabetes\s+association)\b",
            r"\b(standards?\s+of\s+care)\b",
            r"\b(NIH)\b",
        ],
        layer2=[
            "niddk", "ada", "nih", "standards of care",
        ],
    )

    # ---- definition ----
    _add(
        name="definition",
        desc="Asking what a term or concept means",
        pos=[
            "what is", "what are", "what does", "define",
            "definition", "meaning", "what mean",
            "explain", "how does", "how do",
        ],
        neg=[
            "diagnostic", "threshold", "screening",
            "comparison", "treatment",
        ],
        sections=[],
        sources=[],
        weight=0.8,
        layer1=[
            r"\bwhat\s+(is|are|does)\s+\w+\s+(mean|means)?\b",
            r"\b(define|definition|meaning)\b",
            r"\bexplain\s+(the\s+)?\w+\b",
            r"\bwhat\s+is\s+(a|an|the)\s+\w+\b",
        ],
        layer2=[
            "what is", "what does", "define", "definition",
            "meaning", "explain",
        ],
    )

    # ---- unsupported ----
    _add(
        name="unsupported",
        desc="Out-of-scope: medication, treatment, diet, lifestyle",
        pos=[
            "medication", "drug", "metformin", "insulin",
            "treatment", "dosing", "dose",
            "exercise", "diet", "nutrition", "weight loss",
            "lifestyle", "intervention",
            "what should i eat", "what can i eat",
            "cancer", "heart disease", "hypertension treatment",
            "cholesterol",
        ],
        neg=[],
        sections=[],
        sources=[],
        weight=0.5,
        layer1=[
            r"\b(medication|drug|metformin|insulin\s+(treatment|therapy|dose|dosing))\b",
            r"\b(treatment\s+plan|dosing)\b",
            r"\b(exercise|diet|nutrition|weight\s+loss|lifestyle\s+intervention)\b",
            r"\b(what\s+(should|can)\s+I\s+eat)\b",
            r"\b(cancer|heart\s+disease|hypertension\s+treatment|cholesterol)\b",
        ],
        layer2=[
            "medication", "metformin", "insulin", "treatment",
            "exercise", "diet", "nutrition", "weight loss",
            "lifestyle", "cancer", "cholesterol",
        ],
    )

    return profiles


# Module-level registry (built once)
INTENT_PROFILES: dict[str, IntentProfile] = _build_intent_profiles()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _layer1_score(query: str, profile: IntentProfile) -> float:
    """Layer 1: exact regex matches. Returns 0.0 – 1.0."""
    hits = sum(1 for pat in profile._layer1_patterns if pat.search(query))
    if not hits:
        return 0.0
    # Normalise: 1 hit → 0.7, 2+ → up to 1.0
    return min(1.0, 0.5 + 0.25 * hits)


def _layer2_score(query_lower: str, profile: IntentProfile) -> float:
    """Layer 2: fuzzy stem matching. Returns 0.0 – 1.0."""
    if not profile._layer2_stems:
        return 0.0
    hits = sum(1 for stem in profile._layer2_stems if stem.lower() in query_lower)
    return min(1.0, hits / max(len(profile._layer2_stems), 1) * 3)


def _layer3_score(query_lower: str, profile: IntentProfile) -> float:
    """Layer 3: keyword overlap (positive minus negative). Returns 0.0 – 1.0."""
    if not profile.positive_keywords:
        return 0.0
    pos_hits = sum(1 for kw in profile.positive_keywords if kw.lower() in query_lower)
    neg_hits = sum(1 for kw in profile.negative_keywords if kw.lower() in query_lower)
    raw = (pos_hits - neg_hits * 0.5) / max(len(profile.positive_keywords), 1)
    return max(0.0, min(1.0, raw * 2.5))


def _combined_score(
    query: str,
    query_lower: str,
    profile: IntentProfile,
) -> tuple[float, str]:
    """Run all four layers and return (weighted_score, layer_label)."""
    l1 = _layer1_score(query, profile)
    l2 = _layer2_score(query_lower, profile)
    l3 = _layer3_score(query_lower, profile)

    # Weighted combination – layer 1 dominates when it fires
    if l1 > 0:
        combined = l1 * 0.60 + l2 * 0.20 + l3 * 0.20
        layer = "L1"
    elif l2 > 0:
        combined = l2 * 0.55 + l3 * 0.45
        layer = "L2"
    else:
        combined = l3
        layer = "L3"

    # Apply priority weight from intent profile
    combined *= profile.priority_weight
    return combined, layer


# ---------------------------------------------------------------------------
# Query flags helpers
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(
    r"\b(Table\s+\d+\.\d+|criteria\s+table|diagnostic\s+table|comparison\s+table)\b",
    re.IGNORECASE,
)
_THRESHOLD_RE = re.compile(
    r"(threshold|level|cutoff|range|value|≥|>=|\b\d+\.?\d*\s*(mg/dL|mmol/L|%))",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(compar|versus|vs\.?|pros\s+and\s+cons|differ|better)\b",
    re.IGNORECASE,
)
_CONFIRMATION_RE = re.compile(
    r"\b(confirm|repeat|retest|two\s+(different\s+)?(abnormal\s+)?results?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(query: str) -> IntentResult:
    """Classify a query into intents using the layered architecture.

    Returns an IntentResult compatible with the legacy QueryIntent dataclass.
    """
    query_lower = query.lower().strip()
    scores: dict[str, tuple[float, str]] = {}

    for name, profile in INTENT_PROFILES.items():
        score, layer = _combined_score(query, query_lower, profile)
        if score > 0.01:
            scores[name] = (score, layer)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1][0])

    if not ranked:
        return IntentResult(
            primary_intent="general",
            confidence=0.3,
            all_intents=[],
            preferred_sections=[],
            preferred_sources=[],
        )

    primary_name = ranked[0][0]
    primary_score = ranked[0][1][0]
    # Map raw score to confidence 0.4 – 0.95
    confidence = min(0.95, 0.40 + primary_score * 0.55)

    # Collect sections & sources from the primary intent
    primary_profile = INTENT_PROFILES[primary_name]
    preferred_sections = list(primary_profile.target_sections)
    preferred_sources = list(primary_profile.target_sources)

    # Also collect from high-confidence secondary intents (score >= 60% of primary)
    all_intents: list[tuple[str, float]] = []
    for name, (score, layer) in ranked:
        all_intents.append((name, round(score, 3)))
        if score >= primary_score * 0.60 and name != primary_name:
            prof = INTENT_PROFILES[name]
            for s in prof.target_sections:
                if s not in preferred_sections:
                    preferred_sections.append(s)
            for src in prof.target_sources:
                if src not in preferred_sources:
                    preferred_sources.append(src)

    # Query flags
    is_table = bool(_TABLE_RE.search(query))
    is_comparison = bool(_COMPARISON_RE.search(query))
    is_threshold = bool(_THRESHOLD_RE.search(query))
    is_confirmation = bool(_CONFIRMATION_RE.search(query))

    # Topic keywords = positive keywords from primary + top secondary
    topic_kw: list[str] = []
    topic_kw.extend(primary_profile.positive_keywords[:6])
    if len(ranked) > 1:
        sec_name = ranked[1][0]
        topic_kw.extend(INTENT_PROFILES[sec_name].positive_keywords[:3])
    topic_kw = list(dict.fromkeys(topic_kw))  # dedupe, preserve order

    # Excluded sections = sections from intents that are NOT selected but
    # would conflict (e.g. gestational when query is about type 2)
    excluded: list[str] = []
    all_section_set = set()
    for prof in INTENT_PROFILES.values():
        all_section_set.update(prof.target_sections)
    # Sections that belong to intents scoring < 20% of primary
    for name, (score, _) in ranked:
        if score < primary_score * 0.20:
            excluded.extend(INTENT_PROFILES[name].target_sections)
    excluded = list(dict.fromkeys(ex for ex in excluded if ex not in preferred_sections))

    logger.debug(
        "classify('%s') → primary=%s conf=%.2f all=%s layer=%s",
        query[:60], primary_name, confidence,
        [(n, round(s, 2)) for n, s in all_intents[:5]],
        ranked[0][1][1],
    )

    return IntentResult(
        primary_intent=primary_name,
        confidence=confidence,
        all_intents=all_intents,
        preferred_sections=preferred_sections,
        preferred_sources=preferred_sources,
        excluded_sections=excluded,
        required_concepts=topic_kw,
        is_table_query=is_table,
        is_comparison=is_comparison,
        is_threshold_query=is_threshold,
        is_confirmation_query=is_confirmation,
        topic_keywords=topic_kw,
    )


# ---------------------------------------------------------------------------
# Section / source / topic boost functions (backward-compatible API)
# ---------------------------------------------------------------------------

def get_section_boost(section: str, intent: IntentResult) -> float:
    """Calculate section boost based on intent match."""
    if not intent.preferred_sections:
        return 0.0

    # Exact match
    if section in intent.preferred_sections:
        return 0.30

    # Substring / fuzzy match
    for preferred in intent.preferred_sections:
        if preferred.lower() in section.lower() or section.lower() in preferred.lower():
            return 0.22

    return 0.0


def get_source_boost(source_id: str, intent: IntentResult) -> float:
    """Calculate source boost based on intent match."""
    if not intent.preferred_sources:
        return 0.0

    if source_id in intent.preferred_sources:
        return 0.25

    return 0.0


def get_topic_penalty(section: str, intent: IntentResult) -> float:
    """Calculate penalty for off-topic sections."""
    pi = intent.primary_intent

    # Penalise type-1 sections for type-2/gestational/monogenic/prediabetes queries
    if pi in ("type2_diabetes", "gestational_diabetes", "prediabetes_criteria",
              "monogenic_diabetes"):
        if "Type 1 Diabetes" in section:
            return -0.15

    # Penalise gestational for non-gestational queries
    if pi != "gestational_diabetes" and "Gestational" in section:
        return -0.15

    # Penalise monogenic for non-monogenic queries
    if pi != "monogenic_diabetes" and "Monogenic" in section:
        return -0.15

    # Penalise general diagnostic sections for specific-type queries
    if pi in ("type1_diabetes", "type2_diabetes", "gestational_diabetes",
              "monogenic_diabetes"):
        if section in ("Screening and Diagnosis of Diabetes",
                       "Confirming the Diagnosis",
                       "Diagnosis of Prediabetes"):
            return -0.10

    # Penalise specific-type sections for general diagnostic queries
    if pi in ("diagnostic_threshold", "a1c_test", "fpg_test", "ogtt_test",
              "random_glucose"):
        if any(kw in section for kw in ("Type 1", "Gestational", "Monogenic")):
            return -0.10

    # Penalise comparison sections for non-comparison queries
    if pi not in ("test_comparison", "test_interference") and "Comparing" in section:
        return -0.10

    return 0.0
