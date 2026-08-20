"""Table-aware extraction for medical documents.

Extracts tables as structured data while preserving relationships
between row labels, column headers, cell values, and units.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# --- Table detection patterns ---

# ADA table patterns: "Table 2.1", "Table 2.2", etc.
_ADA_TABLE_PATTERN = re.compile(r'Table\s+(\d+\.\d+)(?:[—–:\-]\s*(.+?))?$', re.MULTILINE)

# NIDDK table header
_NIDDK_TABLE_PATTERN = re.compile(r'Comparing Diabetes Blood Tests', re.IGNORECASE)

# Test names for NIDDK splitting
_NIDDK_TEST_MARKERS = [
    ("FPG Test", "Fasting Plasma Glucose"),
    ("OGTT", "Oral Glucose Tolerance Test"),
    ("A1C Test", "A1C"),
    ("A1C", "A1C"),
    ("RPG Test", "Random Plasma Glucose"),
    ("Random Plasma Glucose", "Random Plasma Glucose"),
    ("Random blood sugar", "Random Plasma Glucose"),
]

# Threshold patterns
_THRESHOLD_PATTERN = re.compile(
    r'(?:>=|≥|>|<|<=|≤)\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L|%|mg|mmol)',
    re.IGNORECASE
)

# Units
_UNITS = {
    "mg/dL": "mg/dL",
    "mmol/L": "mmol/L",
    "%": "%",
    "mg": "mg",
    "mmol": "mmol",
}


def detect_table_regions(page_text: str, structured_lines: list = None) -> list[dict]:
    """Detect table regions within a page.
    
    Returns list of table regions with:
    - table_title: str
    - start_line: int
    - end_line: int
    - table_type: 'ada' | 'niddk' | 'generic'
    """
    tables = []

    # Find ADA tables by title
    for match in _ADA_TABLE_PATTERN.finditer(page_text):
        table_num = match.group(1)
        title_suffix = match.group(2) or ""
        title = f"Table {table_num}"
        if title_suffix:
            title += f" — {title_suffix.strip()}"

        # Find approximate line position
        pos = match.start()
        line_num = page_text[:pos].count('\n')

        tables.append({
            "table_title": title,
            "table_number": table_num,
            "start_line": max(0, line_num - 1),
            "end_line": min(line_num + 50, page_text.count('\n')),
            "table_type": "ada",
        })

    # Find NIDDK table
    if _NIDDK_TABLE_PATTERN.search(page_text):
        tables.append({
            "table_title": "Comparing Diabetes Blood Tests",
            "table_number": "niddk_1",
            "start_line": 0,
            "end_line": page_text.count('\n'),
            "table_type": "niddk",
        })

    return tables


def extract_table_structure(text: str, table_type: str = "generic") -> dict:
    """Extract structured representation of a medical table.
    
    Returns dict with:
    - title: table title
    - headers: list of column headers
    - rows: list of row dicts
    - structured_text: retrieval-friendly text representation
    - thresholds: list of threshold dicts
    """
    result = {
        "title": "",
        "headers": [],
        "rows": [],
        "structured_text": "",
        "thresholds": [],
        "raw_text": text,
    }

    lines = text.strip().split('\n')

    if table_type == "niddk":
        result = _extract_niddk_table(text, lines)
    else:
        result = _extract_ada_table(text, lines)

    return result


def _extract_niddk_table(text: str, lines: list) -> dict:
    """Extract NIDDK comparison table structure."""
    result = {
        "title": "Comparing Diabetes Blood Tests",
        "headers": ["Test", "Uses", "Technical Features", "Pros", "Cons"],
        "rows": [],
        "structured_text": "",
        "thresholds": [],
        "raw_text": text,
    }

    # Split by test markers
    current_test = None
    current_test_name = None
    current_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check for test marker
        matched = False
        for marker, full_name in _NIDDK_TEST_MARKERS:
            if line_stripped.startswith(marker) or marker.lower() in line_stripped.lower():
                if current_test and current_lines:
                    _process_niddk_section(result, current_test, current_test_name, current_lines)
                current_test = marker
                current_test_name = full_name
                current_lines = []
                matched = True
                break

        if not matched:
            current_lines.append(line_stripped)

    # Process last section
    if current_test and current_lines:
        _process_niddk_section(result, current_test, current_test_name, current_lines)

    # Build structured text
    structured_parts = []
    for row in result["rows"]:
        test = row.get("test_name", "")
        thresholds = row.get("thresholds", [])
        for t in thresholds:
            structured_parts.append(
                f"NIDDK {test} | {t.get('condition', '')} | {t.get('value', '')} | {t.get('unit', '')}"
            )
        if row.get("uses"):
            structured_parts.append(f"NIDDK {test} | Uses | {row['uses']}")
        if row.get("pros"):
            structured_parts.append(f"NIDDK {test} | Pros | {row['pros']}")
        if row.get("cons"):
            structured_parts.append(f"NIDDK {test} | Cons | {row['cons']}")

    result["structured_text"] = "\n".join(structured_parts)
    return result


def _process_niddk_section(result: dict, test_marker: str, test_name: str, lines: list):
    """Process a single test section from NIDDK table."""
    text = " ".join(lines)

    row = {
        "test_marker": test_marker,
        "test_name": test_name,
        "uses": "",
        "pros": "",
        "cons": "",
        "thresholds": [],
    }

    # Extract thresholds
    for match in _THRESHOLD_PATTERN.finditer(text):
        value = match.group(1)
        unit = match.group(2)
        # Try to find the condition (e.g., "diabetes", "prediabetes")
        context = text[max(0, match.start()-50):match.start()]
        condition = "unknown"
        if "diabetes" in context.lower():
            condition = "diabetes"
        elif "prediabetes" in context.lower() or "impaired" in context.lower():
            condition = "prediabetes"

        row["thresholds"].append({
            "condition": condition,
            "value": value,
            "unit": _UNITS.get(unit, unit),
            "operator": ">=",
        })

    # Extract pros/cons
    pros_match = re.search(r'Pros?\s*(.*?)(?:Cons?|$)', text, re.IGNORECASE | re.DOTALL)
    cons_match = re.search(r'Cons?\s*(.*?)(?:$)', text, re.IGNORECASE | re.DOTALL)

    if pros_match:
        row["pros"] = pros_match.group(1).strip()[:200]
    if cons_match:
        row["cons"] = cons_match.group(1).strip()[:200]

    # Extract uses
    uses_match = re.search(r'(?:Use|Purpose|Screening)\s*(.*?)(?:Technical|Pros|Cons|$)', text, re.IGNORECASE | re.DOTALL)
    if uses_match:
        row["uses"] = uses_match.group(1).strip()[:200]

    result["rows"].append(row)


def _extract_ada_table(text: str, lines: list) -> dict:
    """Extract ADA table structure."""
    result = {
        "title": "",
        "headers": [],
        "rows": [],
        "structured_text": "",
        "thresholds": [],
        "raw_text": text,
    }

    # Find table title
    title_match = _ADA_TABLE_PATTERN.search(text)
    if title_match:
        result["title"] = f"Table {title_match.group(1)}"
        if title_match.group(2):
            result["title"] += f" — {title_match.group(2).strip()}"

    # Extract thresholds from text
    for match in _THRESHOLD_PATTERN.finditer(text):
        value = match.group(1)
        unit = match.group(2)
        context = text[max(0, match.start()-80):match.start()+20]

        condition = "unknown"
        test_type = "unknown"
        if "diabetes" in context.lower():
            condition = "diabetes"
        elif "prediabetes" in context.lower() or "impaired" in context.lower():
            condition = "prediabetes"

        if "a1c" in context.lower() or "hba1c" in context.lower():
            test_type = "A1C"
        elif "fpg" in context.lower() or "fasting" in context.lower():
            test_type = "FPG"
        elif "ogtt" in context.lower() or "2-h" in context.lower() or "2h" in context.lower():
            test_type = "OGTT"
        elif "rpg" in context.lower() or "random" in context.lower():
            test_type = "RPG"

        result["thresholds"].append({
            "condition": condition,
            "test_type": test_type,
            "value": value,
            "unit": _UNITS.get(unit, unit),
            "operator": ">=",
        })

    # Build structured text from thresholds
    structured_parts = []
    for t in result["thresholds"]:
        structured_parts.append(
            f"ADA {result['title']} | {t['test_type']} | {t['condition']} | {t['operator']}{t['value']} {t['unit']}"
        )

    result["structured_text"] = "\n".join(structured_parts)
    return result


def create_table_chunks(table_data: dict, source_id: str, page_pdf: int,
                         section: str, subsection: str = "") -> list[dict]:
    """Create retrieval-optimized chunks from structured table data.
    
    Returns list of chunks, each representing one row/cell of the table.
    """
    chunks = []

    # Create one chunk per threshold/rule
    for i, threshold in enumerate(table_data.get("thresholds", [])):
        test_type = threshold.get("test_type", threshold.get("test_name", ""))
        condition = threshold.get("condition", "")
        value = threshold.get("value", "")
        unit = threshold.get("unit", "")
        operator = threshold.get("operator", ">=")

        text = (
            f"{table_data['title']} | "
            f"{test_type} for {condition}: {operator}{value} {unit}"
        )

        chunks.append({
            "text": text.strip(),
            "source_id": source_id,
            "page_pdf": page_pdf,
            "section": section,
            "subsection": subsection,
            "has_table": True,
            "table_title": table_data.get("title", ""),
            "table_type": table_data.get("table_type", "ada"),
            "threshold_value": f"{operator}{value} {unit}",
            "threshold_test": test_type,
            "threshold_condition": condition,
        })

    # If no thresholds, create a general table chunk
    if not table_data.get("thresholds") and table_data.get("structured_text"):
        chunks.append({
            "text": table_data["structured_text"],
            "source_id": source_id,
            "page_pdf": page_pdf,
            "section": section,
            "subsection": subsection,
            "has_table": True,
            "table_title": table_data.get("title", ""),
            "table_type": table_data.get("table_type", "generic"),
        })

    # Create a summary chunk from all rows
    if table_data.get("rows"):
        row_texts = []
        for row in table_data["rows"]:
            test_name = row.get("test_name", row.get("test_marker", ""))
            uses = row.get("uses", "")
            pros = row.get("pros", "")
            cons = row.get("cons", "")
            row_text = f"{test_name}"
            if uses:
                row_text += f": {uses}"
            if pros:
                row_text += f" (Pros: {pros[:100]})"
            if cons:
                row_text += f" (Cons: {cons[:100]})"
            row_texts.append(row_text)

        chunks.append({
            "text": f"{table_data['title']} — " + " | ".join(row_texts),
            "source_id": source_id,
            "page_pdf": page_pdf,
            "section": section,
            "subsection": subsection,
            "has_table": True,
            "table_title": table_data.get("title", ""),
            "table_type": table_data.get("table_type", "niddk"),
            "is_table_summary": True,
        })

    return chunks


def extract_tables_from_page(page_text: str, structured_lines: list = None,
                              source_id: str = "", page_pdf: int = 0,
                              section: str = "", subsection: str = "") -> list[dict]:
    """Extract and structure tables from a page.
    
    Returns list of structured table chunks.
    """
    table_regions = detect_table_regions(page_text, structured_lines)

    all_table_chunks = []
    for region in table_regions:
        # Extract text for this table region
        lines = page_text.split('\n')
        start = max(0, region["start_line"])
        end = min(len(lines), region["end_line"])
        table_text = '\n'.join(lines[start:end])

        # Extract structure
        table_data = extract_table_structure(table_text, region["table_type"])
        table_data["table_type"] = region["table_type"]
        table_data["title"] = region.get("table_title", table_data.get("title", ""))

        # Create chunks
        chunks = create_table_chunks(
            table_data, source_id, page_pdf, section, subsection
        )
        all_table_chunks.extend(chunks)

    return all_table_chunks
