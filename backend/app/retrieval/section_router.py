"""Section Router - maps queries to the correct ADA/NIDDK document section.

Uses the ADA Standards of Care 2026 document hierarchy to route
queries based on keyword matching, medical terminology, and intent signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SectionNode:
    """A node in the section hierarchy tree."""
    name: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    medical_terms: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)
    page_range: tuple[int, int] = (0, 0)
    source_id: str = "ada_soc_2026_diagnosis"


# ---------------------------------------------------------------------------
# Complete ADA Section Hierarchy (ADA Standards of Care 2026, Ch. 2)
# ---------------------------------------------------------------------------
ADA_SECTION_TREE: dict[str, SectionNode] = {
    "Diagnostic Tests for Diabetes": SectionNode(
        name="Diagnostic Tests for Diabetes",
        parent=None,
        children=[
            "Screening and Diagnosis of Diabetes",
            "Confirming the Diagnosis",
            "Technical Considerations",
        ],
        siblings=[],
        keywords=[
            "diagnostic", "test", "diagnosis", "criteria", "threshold",
            "screening", "confirm", "classification",
        ],
        medical_terms=[
            "a1c", "fpg", "ogtt", "random plasma glucose",
            "hemoglobin a1c", "fasting plasma glucose",
            "oral glucose tolerance test", "2-hour plasma glucose",
            "hba1c", "ifg", "igt", "impaired fasting glucose",
            "impaired glucose tolerance",
        ],
        intent_signals=[
            "diagnostic_criteria", "diagnostic_tests",
            "test_threshold", "cutoff",
        ],
        page_range=(1, 4),
    ),
    "Screening and Diagnosis of Diabetes": SectionNode(
        name="Screening and Diagnosis of Diabetes",
        parent="Diagnostic Tests for Diabetes",
        children=[
            "A1C Test",
            "FPG Test",
            "OGTT",
            "Random Plasma Glucose",
        ],
        siblings=[
            "Confirming the Diagnosis",
            "Technical Considerations",
        ],
        keywords=[
            "screening", "diagnosis", "diagnose", "threshold",
            "criteria", "cut point", "cutpoint", "cutoff",
            "a1c test", "fpg test", "ogtt",
            "fasting glucose", "plasma glucose",
            "point-of-care", "laboratory test",
            "ngsp", "dcct", "ifcc",
            "table 2.1", "table 2.2", "table 2.3",
        ],
        medical_terms=[
            "a1c", "fpg", "ogtt", "hba1c", "hemoglobin a1c",
            "fasting plasma glucose", "oral glucose tolerance test",
            "2-hour plasma glucose", "random plasma glucose",
            "ifg", "igt", "impaired fasting glucose",
            "impaired glucose tolerance",
            "ngsp", "dcct reference assay",
            "point-of-care testing", "clia",
            "hemoglobin variants", "erythropoietin",
            "erythrocyte turnover", "anemia",
            "glucose-6-phosphate dehydrogenase deficiency",
            "retinopathy", "cgm", "continuous glucose monitoring",
        ],
        intent_signals=[
            "diagnostic_criteria", "a1c", "fpg", "ogtt",
            "random_glucose", "test_interference",
            "test_comparison", "table_lookup",
        ],
        page_range=(2, 3),
    ),
    "A1C Test": SectionNode(
        name="A1C Test",
        parent="Screening and Diagnosis of Diabetes",
        children=[],
        siblings=["FPG Test", "OGTT", "Random Plasma Glucose"],
        keywords=[
            "a1c", "hba1c", "hemoglobin", "glycated",
            "ngsp", "dcct", "point-of-care",
        ],
        medical_terms=[
            "hemoglobin a1c", "glycohemoglobin",
            "national glycohemoglobin standardization program",
            "hemoglobin variants", "thalassemia",
            "erythropoietin", "erythrocyte turnover",
            "fructosamine", "glycated albumin",
        ],
        intent_signals=["a1c", "hba1c"],
        page_range=(2, 3),
    ),
    "FPG Test": SectionNode(
        name="FPG Test",
        parent="Screening and Diagnosis of Diabetes",
        children=[],
        siblings=["A1C Test", "OGTT", "Random Plasma Glucose"],
        keywords=[
            "fpg", "fasting", "fasting plasma glucose",
            "fasting glucose", "fasting requirement",
        ],
        medical_terms=[
            "fasting plasma glucose", "impaired fasting glucose",
            "ifg", "fasting definition", "8-hour fasting",
            "glycolysis", "sample stability",
            "coefficient of variation", "diurnal variation",
        ],
        intent_signals=["fpg", "fasting_glucose"],
        page_range=(2, 3),
    ),
    "OGTT": SectionNode(
        name="OGTT",
        parent="Screening and Diagnosis of Diabetes",
        children=[],
        siblings=["A1C Test", "FPG Test", "Random Plasma Glucose"],
        keywords=[
            "ogtt", "oral glucose tolerance",
            "2-hour", "75g glucose", "glucose load",
        ],
        medical_terms=[
            "oral glucose tolerance test", "2-hour plasma glucose",
            "impaired glucose tolerance", "igt",
            "75 g anhydrous glucose", "who protocol",
            "sample stability", "coefficient of variation",
        ],
        intent_signals=["ogtt"],
        page_range=(2, 3),
    ),
    "Random Plasma Glucose": SectionNode(
        name="Random Plasma Glucose",
        parent="Screening and Diagnosis of Diabetes",
        children=[],
        siblings=["A1C Test", "FPG Test", "OGTT"],
        keywords=[
            "random", "random plasma glucose", "symptoms",
            "hyperglycemic crisis",
        ],
        medical_terms=[
            "random plasma glucose", "hyperglycemic crisis",
            "polyuria", "polydipsia", "unexplained weight loss",
        ],
        intent_signals=["random_glucose"],
        page_range=(2, 2),
    ),
    "Confirming the Diagnosis": SectionNode(
        name="Confirming the Diagnosis",
        parent="Diagnostic Tests for Diabetes",
        children=[],
        siblings=[
            "Screening and Diagnosis of Diabetes",
            "Technical Considerations",
        ],
        keywords=[
            "confirm", "confirmation", "repeat test",
            "two tests", "discordant", "repeat testing",
        ],
        medical_terms=[
            "confirmatory testing", "repeat testing",
            "unequivocal hyperglycemia", "single test",
            "two abnormal results", "discordant results",
        ],
        intent_signals=[
            "diagnosis_confirmation", "confirmatory",
        ],
        page_range=(4, 4),
    ),
    "Technical Considerations": SectionNode(
        name="Technical Considerations",
        parent="Diagnostic Tests for Diabetes",
        children=[],
        siblings=[
            "Screening and Diagnosis of Diabetes",
            "Confirming the Diagnosis",
        ],
        keywords=[
            "technical", "preanalytical", "sample handling",
            "interference", "assay",
        ],
        medical_terms=[
            "preanalytical stability", "hemolysis",
            "hypertriglyceridemia", "hyperbilirubinemia",
            "glycolysis", "sample processing",
        ],
        intent_signals=["technical", "preanalytical"],
        page_range=(3, 3),
    ),
    "Type 1 Diabetes": SectionNode(
        name="Type 1 Diabetes",
        parent=None,
        children=[],
        siblings=[
            "Prediabetes and Type 2 Diabetes",
            "Drug-Induced or Chemotherapy-Induced Diabetes",
            "Monogenic Diabetes Syndromes",
            "Gestational Diabetes Mellitus",
        ],
        keywords=[
            "type 1", "autoimmune", "autoantibody",
            "staging", "stage 1", "stage 2", "stage 3",
            "hlA", "genetic risk",
        ],
        medical_terms=[
            "type 1 diabetes", "autoimmune diabetes",
            "islet autoantibodies", "gad65", "ia-2",
            "znt8", "ica", "insulin autoantibodies",
            "hla haplotype", "dr3", "dr4",
            "stage 1 presymptomatic", "stage 2 presymptomatic",
            "stage 3 overt", "diabetic ketoacidosis",
        ],
        intent_signals=["type1", "autoimmune"],
        page_range=(5, 9),
    ),
    "Prediabetes and Type 2 Diabetes": SectionNode(
        name="Prediabetes and Type 2 Diabetes",
        parent=None,
        children=[
            "Screening and Testing for Prediabetes and Type 2 Diabetes",
            "Diagnosis of Prediabetes",
            "Diagnosis of Type 2 Diabetes",
            "Risk-Based Screening",
            "Testing Interval",
        ],
        siblings=[
            "Type 1 Diabetes",
            "Drug-Induced or Chemotherapy-Induced Diabetes",
            "Monogenic Diabetes Syndromes",
            "Gestational Diabetes Mellitus",
        ],
        keywords=[
            "prediabetes", "type 2", "type2", "ifg", "igt",
            "risk factors", "screening", "risk-based",
        ],
        medical_terms=[
            "prediabetes", "type 2 diabetes",
            "impaired fasting glucose", "impaired glucose tolerance",
            "metabolic syndrome", "obesity", "bmi",
            "family history", "sedentary lifestyle",
        ],
        intent_signals=["prediabetes", "type2", "screening"],
        page_range=(10, 18),
    ),
    "Screening and Testing for Prediabetes and Type 2 Diabetes": SectionNode(
        name="Screening and Testing for Prediabetes and Type 2 Diabetes",
        parent="Prediabetes and Type 2 Diabetes",
        children=[],
        siblings=[
            "Diagnosis of Prediabetes",
            "Diagnosis of Type 2 Diabetes",
            "Risk-Based Screening",
            "Testing Interval",
        ],
        keywords=[
            "screening", "testing", "who to screen",
            "when to screen", "risk assessment",
            "screening frequency", "screening interval",
            "how often", "screen for", "at risk",
            "at-risk", "risk factors", "screened",
        ],
        medical_terms=[
            "ada risk test", "screening recommendations",
            "overweight", "obese", "bmi",
            "racial", "ethnic", "african american",
            "hispanic", "asian american",
            "children", "adolescents",
            "hiv", "antiretroviral",
            "screening", "who should be tested",
        ],
        intent_signals=["screening"],
        page_range=(10, 12),
    ),
    "Diagnosis of Prediabetes": SectionNode(
        name="Diagnosis of Prediabetes",
        parent="Prediabetes and Type 2 Diabetes",
        children=[
            "Diagnosis of Type 2 Diabetes",
            "Risk-Based Screening",
        ],
        siblings=[
            "Screening and Testing for Prediabetes and Type 2 Diabetes",
            "Testing Interval",
        ],
        keywords=[
            "prediabetes", "ifg", "igt", "criteria",
            "diagnostic", "threshold", "a1c prediabetes",
        ],
        medical_terms=[
            "impaired fasting glucose", "impaired glucose tolerance",
            "a1c 5.7-6.4%", "fpg 100-125",
            "2-h pg 140-199",
            "risk progression", "diabetes prevention",
        ],
        intent_signals=[
            "prediabetes_criteria", "prediabetes",
        ],
        page_range=(10, 12),
    ),
    "Diagnosis of Type 2 Diabetes": SectionNode(
        name="Diagnosis of Type 2 Diabetes",
        parent="Diagnosis of Prediabetes",
        children=[],
        siblings=["Risk-Based Screening"],
        keywords=[
            "type 2 diagnosis", "type 2 criteria",
            "characteristics", "pathophysiology",
        ],
        medical_terms=[
            "insulin resistance", "beta cell dysfunction",
            "type 2 diabetes", "characteristics",
        ],
        intent_signals=["type2"],
        page_range=(10, 12),
    ),
    "Risk-Based Screening": SectionNode(
        name="Risk-Based Screening",
        parent="Diagnosis of Prediabetes",
        children=[],
        siblings=["Diagnosis of Type 2 Diabetes"],
        keywords=[
            "risk", "risk factors", "risk assessment",
            "screening", "overweight", "obese",
        ],
        medical_terms=[
            "bmi", "family history", "gestational diabetes history",
            "polycystic ovary syndrome", "pcos",
            "hypertension", "dyslipidemia",
        ],
        intent_signals=["screening", "risk"],
        page_range=(11, 12),
    ),
    "Testing Interval": SectionNode(
        name="Testing Interval",
        parent="Diagnosis of Prediabetes",
        children=[],
        siblings=[
            "Diagnosis of Prediabetes",
            "Diagnosis of Type 2 Diabetes",
        ],
        keywords=[
            "interval", "frequency", "how often",
            "retesting", "follow-up testing",
        ],
        medical_terms=[
            "testing interval", "retesting frequency",
            "annual testing", "3-6 months",
        ],
        intent_signals=["testing_interval"],
        page_range=(17, 18),
    ),
    "Drug-Induced or Chemotherapy-Induced Diabetes": SectionNode(
        name="Drug-Induced or Chemotherapy-Induced Diabetes",
        parent=None,
        children=[],
        siblings=[
            "Type 1 Diabetes",
            "Prediabetes and Type 2 Diabetes",
            "Monogenic Diabetes Syndromes",
            "Gestational Diabetes Mellitus",
        ],
        keywords=[
            "drug-induced", "chemotherapy", "steroid",
            "medication-induced",
        ],
        medical_terms=[
            "glucocorticoids", "steroid-induced diabetes",
            "atypical antipsychotics",
            "chemotherapy-related hyperglycemia",
        ],
        intent_signals=["drug_induced"],
        page_range=(14, 14),
    ),
    "Monogenic Diabetes Syndromes": SectionNode(
        name="Monogenic Diabetes Syndromes",
        parent=None,
        children=[],
        siblings=[
            "Type 1 Diabetes",
            "Prediabetes and Type 2 Diabetes",
            "Drug-Induced or Chemotherapy-Induced Diabetes",
            "Gestational Diabetes Mellitus",
        ],
        keywords=[
            "monogenic", "mody", "genetic",
            "neonatal", "syndrome",
        ],
        medical_terms=[
            "monogenic diabetes", "mody",
            "maturity-onset diabetes of the young",
            "hnf1a", "gck", "hnf4a",
            "neonatal diabetes", "genetic testing",
            "sulfonylurea", "keratinocyte",
        ],
        intent_signals=["monogenic", "mody", "genetic"],
        page_range=(14, 15),
    ),
    "Gestational Diabetes Mellitus": SectionNode(
        name="Gestational Diabetes Mellitus",
        parent=None,
        children=[],
        siblings=[
            "Type 1 Diabetes",
            "Prediabetes and Type 2 Diabetes",
            "Drug-Induced or Chemotherapy-Induced Diabetes",
            "Monogenic Diabetes Syndromes",
        ],
        keywords=[
            "gestational", "pregnancy", "pregnant",
            "gdm", "prenatal", "maternal",
        ],
        medical_terms=[
            "gestational diabetes mellitus", "gdm",
            "iadpsg", "one-step", "two-step",
            "50g glt", "75g ogtt", "100g ogtt",
            "carpenter-coustan", "national diabetes data group",
            "macrosomia", "neonatal",
            "24-28 weeks gestation",
        ],
        intent_signals=["gestational", "gdm", "pregnancy"],
        page_range=(16, 18),
    ),
    "References": SectionNode(
        name="References",
        parent=None,
        children=[],
        siblings=[],
        keywords=["references", "bibliography"],
        medical_terms=[],
        intent_signals=[],
        page_range=(19, 23),
    ),
    # NIDDK source
    "Comparing Diabetes Blood Tests": SectionNode(
        name="Comparing Diabetes Blood Tests",
        parent=None,
        children=[],
        siblings=[],
        keywords=[
            "comparing", "comparison", "pros", "cons",
            "advantages", "disadvantages", "niddk",
        ],
        medical_terms=[
            "a1c", "fpg", "ogtt",
            "sensitivity", "specificity",
            "cost", "reproducibility", "variability",
            "sample stability", "preanalytical",
            "within-patient variability",
        ],
        intent_signals=["test_comparison"],
        page_range=(1, 1),
        source_id="niddk_diabetes_prediabetes_tests",
    ),
}


# ---------------------------------------------------------------------------
# Intent-to-Section mapping
# ---------------------------------------------------------------------------
INTENT_SECTION_MAP: dict[str, list[tuple[str, float]]] = {
    "diagnostic_criteria": [
        ("Screening and Diagnosis of Diabetes", 0.6),
        ("Diagnostic Tests for Diabetes", 0.2),
        ("Confirming the Diagnosis", 0.15),
        ("Comparing Diabetes Blood Tests", 0.05),
    ],
    "a1c": [
        ("Screening and Diagnosis of Diabetes", 0.55),
        ("A1C Test", 0.25),
        ("Comparing Diabetes Blood Tests", 0.15),
        ("Technical Considerations", 0.05),
    ],
    "fpg": [
        ("Screening and Diagnosis of Diabetes", 0.55),
        ("FPG Test", 0.25),
        ("Comparing Diabetes Blood Tests", 0.15),
        ("Technical Considerations", 0.05),
    ],
    "ogtt": [
        ("Screening and Diagnosis of Diabetes", 0.55),
        ("OGTT", 0.25),
        ("Comparing Diabetes Blood Tests", 0.15),
        ("Technical Considerations", 0.05),
    ],
    "random_glucose": [
        ("Screening and Diagnosis of Diabetes", 0.6),
        ("Random Plasma Glucose", 0.3),
        ("Confirming the Diagnosis", 0.1),
    ],
    "test_interference": [
        ("Screening and Diagnosis of Diabetes", 0.7),
        ("Technical Considerations", 0.2),
        ("A1C Test", 0.1),
    ],
    "test_comparison": [
        ("Comparing Diabetes Blood Tests", 0.5),
        ("Screening and Diagnosis of Diabetes", 0.3),
        ("A1C Test", 0.1),
        ("FPG Test", 0.05),
        ("OGTT", 0.05),
    ],
    "diagnosis_confirmation": [
        ("Confirming the Diagnosis", 0.65),
        ("Screening and Diagnosis of Diabetes", 0.2),
        ("Diagnostic Tests for Diabetes", 0.15),
    ],
    "type1": [
        ("Type 1 Diabetes", 0.85),
        ("Screening and Diagnosis of Diabetes", 0.1),
        ("Diagnostic Tests for Diabetes", 0.05),
    ],
    "type2": [
        ("Diagnosis of Prediabetes", 0.35),
        ("Diagnosis of Type 2 Diabetes", 0.25),
        ("Prediabetes and Type 2 Diabetes", 0.15),
        ("Screening and Testing for Prediabetes and Type 2 Diabetes", 0.15),
        ("Risk-Based Screening", 0.1),
    ],
    "prediabetes_criteria": [
        ("Diagnosis of Prediabetes", 0.55),
        ("Screening and Diagnosis of Diabetes", 0.25),
        ("Comparing Diabetes Blood Tests", 0.15),
        ("Screening and Testing for Prediabetes and Type 2 Diabetes", 0.05),
    ],
    "screening": [
        ("Screening and Testing for Prediabetes and Type 2 Diabetes", 0.35),
        ("Risk-Based Screening", 0.25),
        ("Screening and Diagnosis of Diabetes", 0.25),
        ("Diagnosis of Prediabetes", 0.15),
    ],
    "gestational": [
        ("Gestational Diabetes Mellitus", 0.75),
        ("Screening and Diagnosis of Diabetes", 0.1),
        ("Confirming the Diagnosis", 0.1),
        ("Diagnosis of Prediabetes", 0.05),
    ],
    "monogenic": [
        ("Monogenic Diabetes Syndromes", 0.8),
        ("Type 1 Diabetes", 0.1),
        ("Diagnosis of Prediabetes", 0.05),
        ("Diagnosis of Type 2 Diabetes", 0.05),
    ],
    "special_population": [
        ("Screening and Testing for Prediabetes and Type 2 Diabetes", 0.3),
        ("Gestational Diabetes Mellitus", 0.25),
        ("Drug-Induced or Chemotherapy-Induced Diabetes", 0.15),
        ("Screening and Diagnosis of Diabetes", 0.15),
        ("Technical Considerations", 0.15),
    ],
    "drug_induced": [
        ("Drug-Induced or Chemotherapy-Induced Diabetes", 0.85),
        ("Screening and Diagnosis of Diabetes", 0.15),
    ],
    "table_lookup": [
        ("Screening and Diagnosis of Diabetes", 0.4),
        ("Comparing Diabetes Blood Tests", 0.3),
        ("Diagnostic Tests for Diabetes", 0.15),
        ("Diagnosis of Prediabetes", 0.15),
    ],
    "unsupported": [],
}


# ---------------------------------------------------------------------------
# SectionRouter
# ---------------------------------------------------------------------------
class SectionRouter:
    """Routes a query to candidate sections with confidence scores.

    The router combines three signals:
      1. Keyword / medical-term matching against section metadata
      2. Intent-based prior probabilities
      3. Parent/child/sibling structural hints

    Returns a list of ``(section_name, score)`` tuples sorted descending
    by score, normalised to [0.0, 1.0].
    """

    def __init__(
        self,
        tree: dict[str, SectionNode] | None = None,
        intent_map: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        self.tree = tree or ADA_SECTION_TREE
        self.intent_map = intent_map or INTENT_SECTION_MAP

        # Pre-compute a flat term index for fast lookup
        self._term_index: dict[str, list[str]] = {}
        for name, node in self.tree.items():
            for term in node.keywords + node.medical_terms:
                key = term.lower()
                self._term_index.setdefault(key, []).append(name)

        # Section bonus weights
        self._keyword_weight = 0.55
        self._intent_weight = 0.35
        self._structure_weight = 0.10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route_query(
        self,
        query: str,
        intent: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return scored candidate sections for *query*.

        Parameters
        ----------
        query:
            The user's natural-language question.
        intent:
            Optional pre-detected intent label (e.g. ``"a1c"``,
            ``"gestational"``).

        Returns
        -------
        list[tuple[str, float]]
            ``(section_name, score)`` pairs sorted best-first, scores in
            [0.0, 1.0].
        """
        query_lower = query.lower()
        query_tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", query_lower))

        # --- 1. Keyword / medical-term scoring --------------------------
        keyword_scores: dict[str, float] = {}
        for token in query_tokens:
            for term, sections in self._term_index.items():
                if token == term or token in term or term in token:
                    for sec in sections:
                        keyword_scores[sec] = keyword_scores.get(sec, 0.0) + 1.0
                # Also try bigram overlap for multi-word terms
                if " " in term and all(
                    w in query_lower for w in term.split()
                ):
                    for sec in sections:
                        keyword_scores[sec] = keyword_scores.get(sec, 0.0) + 1.5

        # Boost exact phrase matches
        for name, node in self.tree.items():
            all_terms = node.keywords + node.medical_terms
            for term in all_terms:
                if len(term) > 2 and term.lower() in query_lower:
                    keyword_scores[name] = keyword_scores.get(name, 0.0) + 2.0

        # --- 2. Intent-based scoring ------------------------------------
        intent_scores: dict[str, float] = {}
        if intent and intent in self.intent_map:
            for sec, prior in self.intent_map[intent]:
                intent_scores[sec] = intent_scores.get(sec, 0.0) + prior

        # Fuzzy intent detection from query text
        detected_intents = self._detect_intents_from_query(query_lower)
        for det_intent, conf in detected_intents:
            if det_intent in self.intent_map:
                for sec, prior in self.intent_map[det_intent]:
                    intent_scores[sec] = (
                        intent_scores.get(sec, 0.0) + prior * conf * 0.5
                    )

        # --- 3. Structural boosting (parent/child/sibling) --------------
        structure_scores: dict[str, float] = {}
        # If keyword scoring favours a section, boost its children
        top_keyword = (
            max(keyword_scores, key=keyword_scores.get)
            if keyword_scores
            else None
        )
        if top_keyword and top_keyword in self.tree:
            node = self.tree[top_keyword]
            for child in node.children:
                structure_scores[child] = structure_scores.get(child, 0.0) + 1.0
            for sibling in node.siblings:
                structure_scores[sibling] = (
                    structure_scores.get(sibling, 0.0) + 0.3
                )
            if node.parent and node.parent in self.tree:
                structure_scores[node.parent] = (
                    structure_scores.get(node.parent, 0.0) + 0.5
                )

        # Also boost parent of top intent-scored section
        top_intent = (
            max(intent_scores, key=intent_scores.get)
            if intent_scores
            else None
        )
        if top_intent and top_intent in self.tree:
            node = self.tree[top_intent]
            for child in node.children:
                structure_scores[child] = (
                    structure_scores.get(child, 0.0) + 0.5
                )

        # --- 4. Combine scores ------------------------------------------
        all_sections = set(self.tree.keys())
        combined: dict[str, float] = {}
        for sec in all_sections:
            combined[sec] = (
                self._keyword_weight * keyword_scores.get(sec, 0.0)
                + self._intent_weight * intent_scores.get(sec, 0.0)
                + self._structure_weight * structure_scores.get(sec, 0.0)
            )

        # Filter out zero-score and Reference sections
        scored = [
            (sec, sc)
            for sec, sc in combined.items()
            if sc > 0.0 and sec != "References"
        ]

        if not scored:
            return []

        # Normalise to [0, 1]
        max_score = max(sc for _, sc in scored)
        if max_score > 0:
            scored = [(sec, round(sc / max_score, 4)) for sec, sc in scored]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detect_intents_from_query(
        self, query_lower: str
    ) -> list[tuple[str, float]]:
        """Heuristic intent detection from query text."""
        intents: list[tuple[str, float]] = []

        patterns: list[tuple[str, list[str], float]] = [
            ("gestational", [
                "gestational", "pregnancy", "pregnant", "gdm",
                "prenatal", "maternal", "iadpsg",
            ], 0.9),
            ("monogenic", [
                "monogenic", "mody", "neonatal diabetes",
                "genetic", "genetic test",
            ], 0.9),
            ("type1", [
                "type 1", "autoimmune", "autoantibod",
                "staging", "stage 1", "stage 2", "stage 3",
                "hlA",
            ], 0.85),
            ("type2", [
                "type 2", "type2",
            ], 0.85),
            ("prediabetes_criteria", [
                "prediabetes", "impaired fasting",
                "impaired glucose tolerance",
            ], 0.85),
            ("screening", [
                "screening", "who should be tested",
                "when to test", "screen for",
                "risk assessment", "overweight",
                "risk factors",
            ], 0.8),
            ("test_comparison", [
                "compare", "comparison", "advantage",
                "disadvantage", "pros and cons",
                "differ", "versus", " vs ",
                "better", "worse", "sensitivity",
                "specificity", "reproducibility",
                "coefficient of variation",
                "within-patient",
            ], 0.8),
            ("diagnosis_confirmation", [
                "confirm", "confirmation", "repeat",
                "single test", "two tests",
                "unequivocal",
            ], 0.8),
            ("diagnostic_criteria", [
                "diagnostic", "criteria", "threshold",
                "cut point", "cutoff", "diagnos",
                "how is .* diagnosed",
            ], 0.75),
            ("a1c", [
                "a1c", "hba1c", "hemoglobin a1c",
                "glycated hemoglobin",
            ], 0.9),
            ("fpg", [
                "fpg", "fasting plasma glucose",
                "fasting glucose", "fasting blood sugar",
            ], 0.9),
            ("ogtt", [
                "ogtt", "oral glucose tolerance",
                "2-hour", "75g glucose",
            ], 0.9),
            ("random_glucose", [
                "random plasma glucose", "random glucose",
            ], 0.9),
            ("test_interference", [
                "interfer", "affect.*result",
                "hemoglobin variant", "variant",
                "alternative.*test",
            ], 0.7),
            ("table_lookup", [
                "table 2\\.", "table \\d",
                "complete.*table", "threshold table",
            ], 0.85),
            ("unsupported", [
                "cancer", "lung cancer", "metformin",
                "medication", "daily carbohydrate",
                "exercise.*insulin", "personal",
                "my a1c", "my glucose", "my fasting",
            ], 0.9),
        ]

        for intent_label, keywords, confidence in patterns:
            for kw in keywords:
                if re.search(kw, query_lower):
                    intents.append((intent_label, confidence))
                    break

        return intents

    def get_section_info(self, section_name: str) -> SectionNode | None:
        """Return the SectionNode for a given section name."""
        return self.tree.get(section_name)

    def get_children(self, section_name: str) -> list[str]:
        """Return child section names."""
        node = self.tree.get(section_name)
        return node.children if node else []

    def get_parent(self, section_name: str) -> str | None:
        """Return parent section name."""
        node = self.tree.get(section_name)
        return node.parent if node else None

    def get_siblings(self, section_name: str) -> list[str]:
        """Return sibling section names."""
        node = self.tree.get(section_name)
        return node.siblings if node else []
