"""Metadata enrichment - detects sections, subsections, and structural elements."""
import re
from typing import Optional

# Known section patterns for ADA document
ADA_SECTION_PATTERNS = [
    (r"^Screening and Diagnosis of Diabetes$", "Screening and Diagnosis of Diabetes", "section"),
    (r"^Diagnostic Tests for Diabetes$", "Diagnostic Tests for Diabetes", "section"),
    (r"^A1C Test$", "A1C Test", "subsection"),
    (r"^FPG Test$", "FPG Test", "subsection"),
    (r"^OGTT$", "OGTT", "subsection"),
    (r"^Random Plasma Glucose$", "Random Plasma Glucose", "subsection"),
    (r"^Confirming the Diagnosis$", "Confirming the Diagnosis", "section"),
    (r"^Technical Considerations$", "Technical Considerations", "section"),
    (r"^TYPE 1 DIABETES$", "Type 1 Diabetes", "section"),
    (r"^PREDIABETES AND TYPE 2 DIABETES$", "Prediabetes and Type 2 Diabetes", "section"),
    (r"^Screening and Testing for Prediabetes and Type 2 Diabetes", "Screening and Testing for Prediabetes and Type 2 Diabetes", "section"),
    (r"^Diagnosis of Prediabetes$", "Diagnosis of Prediabetes", "section"),
    (r"^Diagnosis of Type 2 Diabetes$", "Diagnosis of Type 2 Diabetes", "subsection"),
    (r"^Risk-Based Screening$", "Risk-Based Screening", "subsection"),
    (r"^Testing Interval$", "Testing Interval", "subsection"),
    (r"^Drug-Induced or Chemotherapy-Induced Diabetes$", "Drug-Induced or Chemotherapy-Induced Diabetes", "section"),
    (r"^MONOGENIC DIABETES SYNDROMES$", "Monogenic Diabetes Syndromes", "section"),
    (r"^GESTATIONAL DIABETES MELLITUS$", "Gestational Diabetes Mellitus", "section"),
    (r"^References$", "References", "section"),
]

NIDDK_SECTIONS = {
    1: "Comparing Diabetes Blood Tests",
}

# Recommendation pattern
RECOMMENDATION_PATTERN = re.compile(r"^(\d+\.\d+[a-z]?)\s")


def detect_sections(pages: list[dict], source_id: str) -> list[dict]:
    """Detect and annotate sections for each page.
    
    Iterates pages grouped by source_id so each document starts with
    a fresh section state.  The ``source_id`` parameter is kept for
    backward-compatibility but is ignored in favour of per-page metadata.
    """
    from collections import OrderedDict

    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for page in pages:
        sid = page.get("source_id", "unknown")
        grouped.setdefault(sid, []).append(page)

    for sid, group in grouped.items():
        current_section = "Introduction"
        current_subsection = None

        for page in group:
            lines = [l["text"] for l in page.get("structured_lines", [])]

            if sid == "ada_soc_2026_diagnosis":
                for line in lines:
                    stripped = line.strip()
                    for pattern, name, level in ADA_SECTION_PATTERNS:
                        if re.match(pattern, stripped, re.IGNORECASE):
                            if level == "section":
                                current_section = name
                                current_subsection = None
                            elif level == "subsection":
                                current_subsection = name

            elif sid == "niddk_diabetes_prediabetes_tests":
                current_section = NIDDK_SECTIONS.get(page["page_pdf"], "Comparing Diabetes Blood Tests")

            page["section"] = current_section
            page["subsection"] = current_subsection

    return pages


def detect_tables(pages: list[dict]) -> list[dict]:
    """Flag pages that contain table-like content."""
    for page in pages:
        text = page["text"]
        page["has_table_keywords"] = bool(
            re.search(r"Table\s+\d+\.\d+", text)
            or re.search(r"Test\s+Uses\s+Technical", text)
        )
        page["is_reference_page"] = bool(
            page["section"] == "References" if "section" in page else False
        )
    return pages


def detect_recommendations(pages: list[dict]) -> list[dict]:
    """Count recommendations on each page."""
    for page in pages:
        text = page["text"]
        recs = RECOMMENDATION_PATTERN.findall(text)
        page["num_recommendations"] = len(recs)
        page["recommendation_ids"] = recs[:10]
    return pages


def enrich_metadata(pages: list[dict]) -> list[dict]:
    """Apply all metadata enrichment to pages."""
    if not pages:
        return pages

    pages = detect_sections(pages, "")
    pages = detect_tables(pages)
    pages = detect_recommendations(pages)
    return pages
