"""Page-to-section truth mapping for the ADA Standards of Care 2026 document.

Since section detection during ingestion assigns pages 10-18 all to
'Diagnosis of Prediabetes', this mapping provides the TRUE section for
each page based on manual inspection of the PDF headers.
"""
import logging
logger = logging.getLogger(__name__)

# Page → section mapping (ADA SOC 2026, pages 1-18)
ADA_PAGE_SECTIONS = {
    1: ("Diagnostic Tests for Diabetes", ""),
    2: ("Screening and Diagnosis of Diabetes", ""),
    3: ("Screening and Diagnosis of Diabetes", "A1C Test"),
    4: ("Confirming the Diagnosis", ""),
    5: ("Type 1 Diabetes", ""),
    6: ("Type 1 Diabetes", ""),
    7: ("Type 1 Diabetes", ""),
    8: ("Type 1 Diabetes", ""),
    9: ("Type 1 Diabetes", ""),
    10: ("Diagnosis of Prediabetes", ""),
    11: ("Screening and Testing for Prediabetes and Type 2 Diabetes", ""),
    12: ("Screening and Testing for Prediabetes and Type 2 Diabetes", "Testing Interval"),
    13: ("Drug-Induced or Chemotherapy-Induced Diabetes", ""),
    14: ("Posttransplantation Diabetes", ""),
    15: ("Monogenic Diabetes Syndromes", ""),
    16: ("Monogenic Diabetes Syndromes", "Gestational Diabetes Mellitus"),
    17: ("Gestational Diabetes Mellitus", ""),
    18: ("Gestational Diabetes Mellitus", ""),
    19: ("References", ""),
    20: ("References", ""),
    21: ("References", ""),
    22: ("References", ""),
    23: ("References", ""),
}

# Page → section for NIDDK
NIDDK_PAGE_SECTIONS = {
    1: ("Comparing Diabetes Blood Tests", ""),
}

# Test set expected_sections → page ranges (for relevance checking)
# Includes both short names (from true_section metadata) and full names (from test set)
SECTION_PAGE_RANGES = {
    "Screening and Diagnosis of Diabetes": [2, 3],
    "Diagnostic Tests for Diabetes": [1, 2],
    "Confirming the Diagnosis": [4],
    "Type 1 Diabetes": [5, 6, 7, 8, 9],
    "Diagnosis of Prediabetes": [10],
    "Screening and Testing for Prediabetes and Type 2 Diabetes": [11, 12],
    "Screening and Testing for Prediabetes and Type 2 Diabetes in Asymptomatic Adults": [11, 12],
    "Drug-Induced or Chemotherapy-Induced Diabetes": [12, 13],
    "Cystic Fibrosis-Related Diabetes": [13],
    "Pancreatic Diabetes": [13],
    "Posttransplantation Diabetes": [14],
    "Monogenic Diabetes Syndromes": [15, 16],
    "Gestational Diabetes Mellitus": [16, 17, 18],
    "Comparing Diabetes Blood Tests": [1],
}


def get_true_section(page_pdf: int, source_id: str = "ada_soc_2026_diagnosis") -> str:
    """Get the true section for a page, regardless of metadata."""
    if "niddk" in source_id.lower():
        return NIDDK_PAGE_SECTIONS.get(page_pdf, ("Comparing Diabetes Blood Tests", ""))[0]
    return ADA_PAGE_SECTIONS.get(page_pdf, ("Unknown", ""))[0]


def is_relevant_section(retrieved_section: str, expected_sections: list[str], retrieved_page: int = 0) -> bool:
    """Check if a retrieved chunk is relevant, considering page-level truth.
    
    This is a more lenient check than exact section match:
    - Exact section name match → relevant
    - Prefix match (retrieved is prefix of expected, or vice versa) → relevant
    - Page falls within the expected section's page range → relevant
    """
    if not expected_sections:
        return False
    
    # Exact section match
    if retrieved_section in expected_sections:
        return True
    
    # Prefix match: handle section name truncation/variation
    for expected_sec in expected_sections:
        if retrieved_section.startswith(expected_sec[:30]) or expected_sec.startswith(retrieved_section[:30]):
            return True
    
    # Page-based relevance: check if the retrieved page belongs to any expected section
    if retrieved_page > 0:
        for expected_sec in expected_sections:
            page_range = SECTION_PAGE_RANGES.get(expected_sec, [])
            if retrieved_page in page_range:
                return True
    
    return False


def update_chunk_true_sections(chromadb_path: str):
    """Add 'true_section' metadata field to all chunks based on page truth mapping."""
    import chromadb
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_collection("diabetes_rag")
    
    all_data = collection.get(include=["metadatas"])
    ids = all_data["ids"]
    metas = all_data["metadatas"]
    
    updates = 0
    for cid, meta in zip(ids, metas):
        page = meta.get("page_pdf", meta.get("page_document", 0))
        source = meta.get("source_id", "")
        true_section = get_true_section(page, source)
        
        if meta.get("true_section") != true_section:
            meta["true_section"] = true_section
            collection.update(ids=[cid], metadatas=[meta])
            updates += 1
    
    logger.info(f"Updated {updates}/{len(ids)} chunks with true_section metadata")
    return updates
