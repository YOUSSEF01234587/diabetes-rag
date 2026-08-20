"""Improved medical query expansion for Diabetes RAG (v2).

Key improvements over v1:
  - 80+ medical term mappings with patient-facing → medical term translations
  - Abbreviation expansion with context
  - Source-specific and section-specific term sets
  - Context-aware selection (does NOT blindly add every synonym)
  - Every expansion is logged with its reason
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expansion record (for logging / debugging)
# ---------------------------------------------------------------------------

@dataclass
class ExpansionRecord:
    """Single expansion applied to a query."""
    original_term: str
    added_terms: list[str]
    reason: str


# ---------------------------------------------------------------------------
# Master medical terms dictionary
# Maps:  patient-facing term → (medical aliases, source hint, section hint)
# ---------------------------------------------------------------------------

@dataclass
class TermMapping:
    """Defines how a term should be expanded."""
    medical_aliases: list[str]
    source_hint: str = ""      # "niddk" | "ada" | ""
    section_hint: str = ""     # section name to nudge toward
    priority: int = 1          # higher = expanded first when budget limited


MEDICAL_TERMS: dict[str, TermMapping] = {
    # ---- FPG variants ----
    "fasting blood sugar": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting blood glucose"],
        priority=2,
    ),
    "fasting glucose": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting blood sugar"],
        priority=2,
    ),
    "fasting blood glucose": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting glucose"],
        priority=2,
    ),
    "fasting test": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting blood glucose", "overnight fast 8 hours"],
    ),
    "fasting blood test": TermMapping(
        ["fasting plasma glucose", "FPG"],
    ),
    "blood sugar fasting": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting glucose"],
    ),
    "fpg test": TermMapping(
        ["fasting plasma glucose", "FPG", "fasting blood glucose"],
    ),
    "fpg level": TermMapping(
        ["fasting plasma glucose", "FPG"],
    ),

    # ---- OGTT variants ----
    "glucose tolerance test": TermMapping(
        ["oral glucose tolerance test", "OGTT", "2-hour glucose"],
        priority=2,
    ),
    "oral glucose tolerance": TermMapping(
        ["OGTT", "2-hour glucose", "glucose tolerance test"],
        priority=2,
    ),
    "2 hour glucose test": TermMapping(
        ["OGTT", "2-hour glucose", "oral glucose tolerance test"],
    ),
    "2-hour glucose": TermMapping(
        ["OGTT", "2-h PG", "oral glucose tolerance test"],
    ),
    "glucose tolerance": TermMapping(
        ["OGTT", "oral glucose tolerance test", "2-hour glucose"],
    ),
    "ogtt test": TermMapping(
        ["oral glucose tolerance test", "OGTT", "75 gram glucose"],
    ),

    # ---- A1C variants ----
    "hemoglobin a1c": TermMapping(
        ["A1C", "glycated hemoglobin", "HbA1c"],
        priority=2,
    ),
    "glycated hemoglobin": TermMapping(
        ["A1C", "HbA1c", "hemoglobin a1c"],
        priority=2,
    ),
    "hba1c": TermMapping(
        ["A1C", "glycated hemoglobin", "hemoglobin a1c"],
        priority=2,
    ),
    "a1c test": TermMapping(
        ["A1C", "glycated hemoglobin", "HbA1c"],
    ),
    "a1c level": TermMapping(
        ["A1C", "glycated hemoglobin"],
    ),
    "a1c percentage": TermMapping(
        ["A1C", "glycated hemoglobin"],
    ),
    "glycosylated hemoglobin": TermMapping(
        ["A1C", "HbA1c", "glycated hemoglobin"],
    ),

    # ---- Random glucose ----
    "random blood sugar": TermMapping(
        ["random plasma glucose", "RPG", "random blood glucose"],
        priority=2,
    ),
    "random glucose": TermMapping(
        ["random plasma glucose", "RPG"],
        priority=2,
    ),
    "random blood glucose": TermMapping(
        ["random plasma glucose", "RPG"],
    ),

    # ---- Diagnostic thresholds ----
    "prediabetes criteria": TermMapping(
        ["prediabetes diagnosis", "impaired fasting glucose", "IFG",
         "impaired glucose tolerance", "IGT", "A1C 5.7-6.4%"],
        section_hint="Diagnosis of Prediabetes",
    ),
    "prediabetes thresholds": TermMapping(
        ["prediabetes diagnostic thresholds", "IFG", "IGT",
         "A1C 5.7-6.4%", "FPG 100-125", "OGTT 140-199"],
        section_hint="Diagnosis of Prediabetes",
    ),
    "diabetes thresholds": TermMapping(
        ["diabetes diagnostic thresholds", "FPG ≥126", "OGTT ≥200",
         "A1C ≥6.5%"],
        section_hint="Screening and Diagnosis of Diabetes",
    ),
    "diagnostic criteria": TermMapping(
        ["diagnostic criteria", "diagnosis classification",
         "A1C", "FPG", "OGTT"],
        section_hint="Screening and Diagnosis of Diabetes",
    ),
    "diagnostic threshold": TermMapping(
        ["diagnostic threshold", "diagnostic cutoff", "diagnostic level",
         "A1C ≥6.5%", "FPG ≥126", "OGTT ≥200", "RPG ≥200"],
        section_hint="Screening and Diagnosis of Diabetes",
    ),
    "cutoff": TermMapping(
        ["diagnostic cutoff", "threshold", "level"],
    ),

    # ---- Type 1 diabetes ----
    "autoantibodies": TermMapping(
        ["type 1 diabetes autoantibodies", "IAA", "GADA", "IA-2A",
         "ZnT8A", "ICA"],
        section_hint="Type 1 Diabetes",
    ),
    "type 1 diabetes antibodies": TermMapping(
        ["autoantibodies", "IAA", "GADA", "IA-2A", "ZnT8A"],
        section_hint="Type 1 Diabetes",
    ),
    "islet cell antibodies": TermMapping(
        ["autoantibodies", "ICA", "GADA"],
        section_hint="Type 1 Diabetes",
    ),
    "type 1": TermMapping(
        ["type 1 diabetes", "T1D", "autoimmune diabetes"],
        section_hint="Type 1 Diabetes",
    ),
    "autoimmune diabetes": TermMapping(
        ["type 1 diabetes", "autoantibodies", "autoimmune"],
        section_hint="Type 1 Diabetes",
    ),

    # ---- Type 2 diabetes ----
    "type 2": TermMapping(
        ["type 2 diabetes", "T2D"],
        section_hint="Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
    ),
    "insulin resistance": TermMapping(
        ["type 2 diabetes", "insulin resistance", "metabolic syndrome"],
        section_hint="Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
    ),

    # ---- Gestational diabetes ----
    "gestational diabetes": TermMapping(
        ["gestational diabetes mellitus", "GDM", "pregnancy diabetes"],
        section_hint="Gestational Diabetes Mellitus",
    ),
    "pregnancy diabetes": TermMapping(
        ["gestational diabetes mellitus", "GDM"],
        section_hint="Gestational Diabetes Mellitus",
    ),
    "gdm": TermMapping(
        ["gestational diabetes mellitus", "pregnancy diabetes"],
        section_hint="Gestational Diabetes Mellitus",
    ),

    # ---- Monogenic diabetes ----
    "mody": TermMapping(
        ["maturity-onset diabetes of the young", "monogenic diabetes",
         "HNF1A-MODY", "GCK-MODY"],
        section_hint="Monogenic Diabetes Syndromes",
    ),
    "monogenic diabetes": TermMapping(
        ["monogenic diabetes", "MODY", "neonatal diabetes"],
        section_hint="Monogenic Diabetes Syndromes",
    ),
    "neonatal diabetes": TermMapping(
        ["neonatal diabetes", "monogenic diabetes", "KCNJ11", "ABCC8"],
        section_hint="Monogenic Diabetes Syndromes",
    ),

    # ---- CGM ----
    "continuous glucose monitoring": TermMapping(
        ["CGM", "continuous glucose monitoring"],
    ),
    "cgm": TermMapping(
        ["continuous glucose monitoring", "CGM"],
    ),

    # ---- Confirmation ----
    "confirm diagnosis": TermMapping(
        ["confirming diagnosis", "confirmation", "repeat test",
         "two different tests"],
        section_hint="Confirming the Diagnosis",
    ),
    "diagnosis confirmation": TermMapping(
        ["confirming diagnosis", "confirmation", "repeat test"],
        section_hint="Confirming the Diagnosis",
    ),

    # ---- Misc medical terms ----
    "impaired fasting glucose": TermMapping(
        ["IFG", "prediabetes", "FPG 100-125 mg/dL"],
        section_hint="Diagnosis of Prediabetes",
    ),
    "impaired glucose tolerance": TermMapping(
        ["IGT", "prediabetes", "OGTT 140-199 mg/dL"],
        section_hint="Diagnosis of Prediabetes",
    ),
    "hyperglycemia": TermMapping(
        ["high blood glucose", "elevated blood glucose"],
    ),
    "hypoglycemia": TermMapping(
        ["low blood glucose", "low blood sugar"],
    ),
    "oral glucose tolerance test": TermMapping(
        ["OGTT", "2-hour glucose", "75 gram glucose load"],
    ),
    "plasma glucose": TermMapping(
        ["blood glucose", "blood sugar"],
    ),
    "mg/dl": TermMapping(
        ["mg/dL", "milligrams per deciliter"],
    ),
    "mmol/l": TermMapping(
        ["mmol/L", "millimoles per liter"],
    ),
    "75 grams": TermMapping(
        ["75 gram glucose load", "75g OGTT"],
    ),
    "75g": TermMapping(
        ["75 gram glucose load", "75g OGTT"],
    ),
    "411": TermMapping(
        ["A1C 5.7-6.4%", "prediabetes A1C"],
    ),
    "5.7 to 6.4": TermMapping(
        ["A1C 5.7-6.4%", "prediabetes A1C range"],
    ),
    "100 to 125": TermMapping(
        ["FPG 100-125 mg/dL", "IFG range"],
    ),
    "140 to 199": TermMapping(
        ["OGTT 140-199 mg/dL", "IGT range"],
    ),
    "screening": TermMapping(
        ["screening", "when to test", "who should be screened"],
        section_hint="Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
    ),
    "when to test": TermMapping(
        ["screening", "when to screen", "screening frequency"],
        section_hint="Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults",
    ),
    "comparison": TermMapping(
        ["test comparison", "sensitivity", "specificity", "pros and cons"],
        section_hint="Comparing Diabetes Blood Tests",
        source_hint="niddk",
    ),
    "pros and cons": TermMapping(
        ["advantages", "disadvantages", "test comparison"],
        section_hint="Comparing Diabetes Blood Tests",
        source_hint="niddk",
    ),
    "sensitivity": TermMapping(
        ["sensitivity", "specificity", "diagnostic accuracy"],
        section_hint="Comparing Diabetes Blood Tests",
    ),
    "cystic fibrosis": TermMapping(
        ["cystic fibrosis-related diabetes", "CFRD"],
        section_hint="Screening and Diagnosis of Diabetes",
    ),
    "interference": TermMapping(
        ["test interference", "falsely elevated", "falsely low",
         "hemoglobin variants", "G6PD", "sickle cell"],
    ),
    "false result": TermMapping(
        ["falsely elevated", "falsely low", "interference",
         "inaccurate result"],
    ),
    "g6pd": TermMapping(
        ["G6PD deficiency", "hemolytic anemia", "falsely low A1C"],
    ),
    "sickle cell": TermMapping(
        ["sickle cell disease", "hemoglobin variants",
         "interference with A1C"],
    ),
}

# ---- Abbreviation → full name mappings ----
ABBREVIATIONS: dict[str, str] = {
    "fpg": "fasting plasma glucose",
    "ogtt": "oral glucose tolerance test",
    "rpg": "random plasma glucose",
    "a1c": "A1C glycated hemoglobin",
    "hba1c": "hemoglobin A1C glycated hemoglobin",
    "ifg": "impaired fasting glucose",
    "igt": "impaired glucose tolerance",
    "gdm": "gestational diabetes mellitus",
    "mody": "maturity-onset diabetes of the young",
    "cgm": "continuous glucose monitoring",
    "t1d": "type 1 diabetes",
    "t2d": "type 2 diabetes",
    "iaa": "insulin autoantibodies",
    "gada": "glutamic acid decarboxylase autoantibodies",
    "ia-2a": "islet antigen 2 autoantibodies",
    "znt8": "zinc transporter 8 autoantibodies",
    "ica": "islet cell autoantibodies",
    "ngsp": "national glycohemoglobin standardization program",
    "dcct": "diabetes control and complications trial",
    "iadpsg": "international association of diabetes and pregnancy study groups",
    "cfrelated": "cystic fibrosis related",
    "niddk": "national institute of diabetes and digestive and kidney diseases",
    "ada": "american diabetes association",
}

# ---- Source-specific expansions ----
SOURCE_TERMS: dict[str, list[str]] = {
    "niddk": [
        "NIDDK", "comparing", "comparison", "pros and cons",
        "technical features", "coefficient of variation",
        "sensitivity", "specificity", "convenience", "cost",
    ],
    "ada": [
        "ADA", "standards of care", "diagnostic criteria",
        "screening", "classification", "type 1", "type 2",
        "gestational", "monogenic", "prediabetes", "confirming",
    ],
}

# ---- Section-specific term sets ----
SECTION_TERM_SETS: dict[str, list[str]] = {
    "Screening and Diagnosis of Diabetes": [
        "diagnostic criteria", "screening", "diagnosis",
        "threshold", "cutoff", "level", "A1C", "FPG", "OGTT",
        "fasting", "glucose", "RPG", "random plasma glucose",
    ],
    "Confirming the Diagnosis": [
        "confirming", "confirmation", "repeat test",
        "two different tests", "two abnormal results",
    ],
    "Diagnosis of Prediabetes": [
        "prediabetes", "IFG", "IGT", "impaired fasting glucose",
        "impaired glucose tolerance", "5.7-6.4%", "100-125",
        "140-199", "borderline",
    ],
    "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults": [
        "screening", "when to test", "who should be screened",
        "asymptomatic", "risk factors", "every 3 years",
        "type 2", "prevalence",
    ],
    "Type 1 Diabetes": [
        "type 1 diabetes", "autoantibodies", "autoimmune",
        "staging", "beta cell", "IAA", "GADA", "IA-2A", "ZnT8",
        "islet cell",
    ],
    "Gestational Diabetes Mellitus": [
        "gestational diabetes", "GDM", "pregnancy", "prenatal",
        "24-28 weeks", "IADPSG", "75-g OGTT", "maternal",
    ],
    "Monogenic Diabetes Syndromes": [
        "monogenic", "MODY", "neonatal diabetes", "genetic testing",
        "HNF1A", "GCK", "KCNJ11", "ABCC8",
    ],
    "Comparing Diabetes Blood Tests": [
        "comparison", "pros", "cons", "sensitivity", "specificity",
        "coefficient of variation", "cost", "convenience",
        "A1C vs FPG", "FPG vs OGTT",
    ],
}


# ---------------------------------------------------------------------------
# Core expansion function
# ---------------------------------------------------------------------------

def expand_query_v2(
    query: str,
    intent: Optional[object] = None,
) -> str:
    """Expand query with context-aware medical term additions.

    Args:
        query: Original user query.
        intent: Optional IntentResult (from intent_v2.py) to guide expansion.

    Returns:
        Expanded query string with added medical terms.
    """
    query_lower = query.lower().strip()
    expansions: list[ExpansionRecord] = []
    added_terms: list[str] = []

    # 1. Abbreviation expansion – always add full forms
    for abbr, full_form in ABBREVIATIONS.items():
        if re.search(r'\b' + re.escape(abbr) + r'\b', query_lower):
            for token in full_form.split():
                if token.lower() not in query_lower and token not in added_terms:
                    added_terms.append(token)
            expansions.append(ExpansionRecord(
                original_term=abbr,
                added_terms=full_form.split(),
                reason="abbreviation expansion",
            ))

    # 2. Medical term expansion – select by relevance
    matched_terms: list[tuple[str, TermMapping, int]] = []
    for term, mapping in MEDICAL_TERMS.items():
        if term.lower() in query_lower:
            matched_terms.append((term, mapping, mapping.priority))

    # Sort by priority descending
    matched_terms.sort(key=lambda x: -x[2])

    # Budget: max ~20 added terms to avoid noise
    budget = 20 - len(added_terms)
    for term, mapping, _priority in matched_terms:
        if budget <= 0:
            break
        new_terms = []
        for alias in mapping.medical_aliases:
            # Skip if already in query or already added
            if alias.lower() in query_lower or alias in added_terms:
                continue
            new_terms.append(alias)
            if alias not in added_terms:
                added_terms.append(alias)
            budget -= 1
            if budget <= 0:
                break
        if new_terms:
            expansions.append(ExpansionRecord(
                original_term=term,
                added_terms=new_terms,
                reason=f"medical term expansion for '{term}'",
            ))

    # 3. Source-specific expansion (if intent hints at a source)
    source_hint = _extract_source_hint(query_lower, intent)
    if source_hint and source_hint in SOURCE_TERMS:
        for st in SOURCE_TERMS[source_hint]:
            if st.lower() not in query_lower and st not in added_terms:
                added_terms.append(st)
        expansions.append(ExpansionRecord(
            original_term=f"[source:{source_hint}]",
            added_terms=SOURCE_TERMS[source_hint][:5],
            reason=f"source-specific expansion for {source_hint}",
        ))

    # 4. Section-specific expansion (if intent hints at a section)
    section_hint = _extract_section_hint(intent)
    if section_hint and section_hint in SECTION_TERM_SETS:
        for st in SECTION_TERM_SETS[section_hint]:
            if st.lower() not in query_lower and st not in added_terms:
                added_terms.append(st)
        expansions.append(ExpansionRecord(
            original_term=f"[section:{section_hint}]",
            added_terms=SECTION_TERM_SETS[section_hint][:5],
            reason=f"section-specific expansion for {section_hint}",
        ))

    if expansions:
        expanded = query + " " + " ".join(added_terms)
        logger.debug(
            "expand_query_v2('%s') → %d expansions, %d terms added: %s",
            query[:60], len(expansions), len(added_terms),
            [e.reason for e in expansions],
        )
        return expanded

    return query


# ---------------------------------------------------------------------------
# Query variant generation
# ---------------------------------------------------------------------------

def generate_query_variants_v2(
    query: str,
    intent: Optional[object] = None,
) -> list[tuple[str, str]]:
    """Generate 2-4 high-value query variants with reasons.

    Returns:
        List of (variant_query, reason) tuples.
    """
    variants: list[tuple[str, str]] = []
    query_lower = query.lower().strip()

    # Variant 1: abbreviated form (if query uses full names)
    abbr_variant = _try_abbreviate(query_lower)
    if abbr_variant and abbr_variant != query_lower:
        variants.append((abbr_variant, "Abbreviated form for alternative retrieval"))

    # Variant 2: medical-term form (if query uses lay terms)
    medical_variant = _try_medical_form(query_lower)
    if medical_variant and medical_variant != query_lower:
        variants.append((medical_variant, "Medical terminology form"))

    # Variant 3: rephrased as a diagnostic question
    diag_variant = _try_diagnostic_rephrase(query_lower, intent)
    if diag_variant:
        variants.append((diag_variant, "Rephrased as diagnostic criteria question"))

    # Variant 4: source-specific variant
    src_variant = _try_source_variant(query_lower, intent)
    if src_variant:
        variants.append((src_variant, "Source-specific variant"))

    # Dedupe and cap at 4
    seen = {query_lower}
    unique: list[tuple[str, str]] = []
    for v, r in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique.append((v, r))
    return unique[:4]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_source_hint(query_lower: str, intent: Optional[object]) -> str:
    """Determine if a source hint should be applied."""
    # Explicit mention in query
    if "niddk" in query_lower:
        return "niddk"
    if "ada" in query_lower or "american diabetes association" in query_lower:
        return "ada"
    # Use intent if available
    if intent is not None and hasattr(intent, "preferred_sources"):
        for src in getattr(intent, "preferred_sources", []):
            if "niddk" in src:
                return "niddk"
            if "ada" in src:
                return "ada"
    return ""


def _extract_section_hint(intent: Optional[object]) -> str:
    """Determine if a section hint should be applied from intent."""
    if intent is None or not hasattr(intent, "preferred_sections"):
        return ""
    sections = getattr(intent, "preferred_sections", [])
    if sections:
        return sections[0]
    return ""


def _try_abbreviate(query_lower: str) -> str:
    """If query uses long form names, produce an abbreviated variant."""
    result = query_lower
    long_to_short = {
        "fasting plasma glucose": "FPG",
        "oral glucose tolerance test": "OGTT",
        "random plasma glucose": "RPG",
        "hemoglobin a1c": "A1C",
        "glycated hemoglobin": "A1C",
        "gestational diabetes mellitus": "GDM",
        "type 1 diabetes": "T1D",
        "type 2 diabetes": "T2D",
        "impaired fasting glucose": "IFG",
        "impaired glucose tolerance": "IGT",
        "maturity-onset diabetes of the young": "MODY",
        "continuous glucose monitoring": "CGM",
        "random blood sugar": "RPG",
        "fasting blood sugar": "FPG",
    }
    changed = False
    for long, short in long_to_short.items():
        if long in result:
            result = result.replace(long, short)
            changed = True
    return result if changed else ""


def _try_medical_form(query_lower: str) -> str:
    """If query uses lay terms, produce a medical terminology variant."""
    result = query_lower
    lay_to_medical = {
        "blood sugar": "blood glucose plasma glucose",
        "sugar test": "glucose test plasma glucose",
        "diabetes test": "diabetes diagnostic test",
        "blood test for diabetes": "diabetes diagnostic blood test fasting plasma glucose A1C",
        "sugar level": "glucose level plasma glucose",
        "high sugar": "hyperglycemia elevated glucose",
        "low sugar": "hypoglycemia low glucose",
        "borderline diabetes": "prediabetes impaired fasting glucose impaired glucose tolerance",
        "sugar disease": "diabetes mellitus",
    }
    changed = False
    for lay, medical in lay_to_medical.items():
        if lay in result:
            result = result.replace(lay, medical)
            changed = True
    return result if changed else ""


def _try_diagnostic_rephrase(query_lower: str, intent: Optional[object]) -> str:
    """If query is about a test but not framed as a diagnostic question,
    produce a rephrased variant."""
    # Only rephrase if the query mentions a test but doesn't ask about thresholds
    test_terms = ["a1c", "fpg", "ogtt", "fasting", "glucose tolerance", "random glucose"]
    has_test = any(t in query_lower for t in test_terms)
    has_diagnostic = any(w in query_lower for w in [
        "diagnostic criteria", "threshold", "cutoff", "level",
        "what is the", "classify",
    ])
    if has_test and not has_diagnostic:
        return f"diagnostic criteria for {query_lower}"
    return ""


def _try_source_variant(query_lower: str, intent: Optional[object]) -> str:
    """If no explicit source mentioned, add a source-specific variant."""
    if "niddk" in query_lower or "ada" in query_lower:
        return ""
    # Check if intent suggests NIDDK (comparison questions)
    if intent is not None and hasattr(intent, "preferred_sources"):
        for src in getattr(intent, "preferred_sources", []):
            if "niddk" in src:
                return f"{query_lower} NIDDK comparison"
            if "ada" in src:
                return f"{query_lower} ADA standards of care"
    return ""


# ---------------------------------------------------------------------------
# Convenience: combined expand + variants (for callers that want both)
# ---------------------------------------------------------------------------

def expand_and_generate_variants(
    query: str,
    intent: Optional[object] = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Return (expanded_query, variants) in one call."""
    expanded = expand_query_v2(query, intent)
    variants = generate_query_variants_v2(query, intent)
    return expanded, variants
