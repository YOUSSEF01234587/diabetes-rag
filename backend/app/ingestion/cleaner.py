"""Text cleaning pipeline for medical documents.

Separates retrieval_text (clean content for embeddings/BM25)
from display_text (original evidence) and metadata (URLs, DOIs, citations).
"""
import re
from typing import Optional

# --- Noise patterns ---

# URLs (http/https/www)
_URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\'\)\]]+'
    r'|www\.[^\s<>\"\'\)\]]+',
    re.IGNORECASE
)

# DOI strings
_DOI_PATTERN = re.compile(
    r'(?:doi:\s*|https?://doi\.org/)?10\.\d{4,9}/[^\s<>\"\'\)\]]+',
    re.IGNORECASE
)

# Copyright/license blocks
_COPYRIGHT_PATTERN = re.compile(
    r'(?:©|Copyright)\s*\d{4}.*?(?:\.|$)',
    re.IGNORECASE
)

# Page headers/footers: "Diabetes Care Volume XX, Supplement Y, Month Year"
_HEADER_FOOTER_PATTERN = re.compile(
    r'Diabetes Care\s+Volume\s+\d+.*?(?:\d{4}|Supplement\s+\d+)',
    re.IGNORECASE
)

# Journal navigation: page numbers like "S27" or "S49" at line boundaries
_PAGE_NUM_PATTERN = re.compile(
    r'^\s*S\d{1,3}\s*$',
    re.MULTILINE
)

# Repeated whitespace/newlines
_MULTI_WHITESPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINES = re.compile(r'\n{3,}')

# Standalone reference numbers like "(1)" "(2,3)" "(15,16)"
_STANDALONE_REFS = re.compile(r'^\s*\(\d+(?:,\s*\d+)*\)\s*$', re.MULTILINE)

# Recommendation IDs like "2.6" at start of line (keep the text, remove just the ID prefix when it's noise)
_REC_ID = re.compile(r'^(\d+\.\d+[a-z]?)\s+', re.MULTILINE)

# Bracketed references like "[1]" "[2,3]" "[15-20]"
_BRACKET_REFS = re.compile(r'\[[\d,\s\-]+\]')

# Navigation artifacts
_NAV_ARTIFACTS = re.compile(
    r'(?:Downloaded from|Published online|Advance online|EPub ahead of|'
    r'Official journal of|The journal of|An ADA Publication|'
    r'American Diabetes Association|Diabetes Care\s+diabetesjournals\.org)',
    re.IGNORECASE
)

# Journal footer: "diabetesjournals.org/care Diagnosis and Classification of Diabetes"
_JOURNAL_FOOTER = re.compile(
    r'diabetesjournals\.org/care\s+Diagnosis and Classification of Diabetes',
    re.IGNORECASE
)

# --- NEW: Enhanced noise patterns ---

# Download stamps: "by guest on 18 August 2026" or similar
_DOWNLOAD_STAMP = re.compile(
    r'by\s+guest\s+on\s+\d{1,2}\s+\w+\s+\d{4}',
    re.IGNORECASE
)

# PDF download URLs with timestamps (including zero-width space and soft-hyphenated versions)
_PDF_DOWNLOAD_URL = re.compile(
    r'diabetesjournals\.[\u00ad\u200b]*org/[\w/\-\.\u00ad\u200b]+\.pdf',
    re.IGNORECASE
)

# Zero-width spaces and soft hyphens
_ZERO_WIDTH_CHARS = re.compile(r'[\u00ad\u200b\u200c\u200d\ufeff]')

# Soft hyphen (U+00AD) followed by optional whitespace
_SOFT_HYPHEN = re.compile(r'\u00ad\s*')

# Split words from soft hyphens: "diag nosis" -> "diagnosis" (common pattern)
_SPLIT_WORDS = re.compile(
    r'\b(diag|diag­|diag\u00ad)\s+(nosis|noses|nosed)\b'
    r'|(in)\s+(dividuals|dividual)\b'
    r'|(cla)\s+(ssification|ssifications)\b'
    r'|(sickle)\s+(cell)\b'
    r'|(fast)\s+(ing)\b'
    r'|(fasting)\s+(plasma)\b'
    r'|(inter)\s+(ference|ferences)\b'
    r'|(ana)\s+(lytical)\b'
    r'|(analytical)\s+(problem)\b',
    re.IGNORECASE
)

# Copyright/reproduction boilerplate blocks (multi-line)
_COPYRIGHT_BLOCK = re.compile(
    r'Readers may use this work.*?third-party site or platform\.'
    r'|This publication and its contents.*?prior written permission\.'
    r'|Requests to reuse.*?diabetes\s*\.org\.'
    r'|More information is available at.*?license\.',
    re.IGNORECASE | re.DOTALL
)

# Permissions email
_PERMISSIONS_EMAIL = re.compile(
    r'permissions@diabetes\s*\.org',
    re.IGNORECASE
)

# Citation boilerplate
_CITATION_BOILERPLATE = re.compile(
    r'Suggested citation:.*?(?:Diabetes Care\s+\d{4}.*?\d+\-\d+)',
    re.IGNORECASE | re.DOTALL
)

# Duality of interest boilerplate
_DUALITY_BOILERPLATE = re.compile(
    r'Duality of interest information.*?(?:available at|http)',
    re.IGNORECASE | re.DOTALL
)

# ADA member list boilerplate
_MEMBER_LIST_BOILERPLATE = re.compile(
    r'\*A complete list of members.*?(?:https?://|doi\.org)',
    re.IGNORECASE | re.DOTALL
)

# Empty parentheses or brackets with only whitespace
_EMPTY_BRACKETS = re.compile(r'\(\s*\)|\[\s*\]|\{\s*\}')


def clean_for_retrieval(text: str) -> dict:
    """Clean text for retrieval while preserving medical content.
    
    Returns dict with:
    - retrieval_text: cleaned content optimized for embeddings/BM25
    - display_text: cleaned content for user display
    - removed_urls: list of extracted URLs
    - removed_dois: list of extracted DOIs
    - removed_noise: list of removed noise patterns
    """
    removed_urls = []
    removed_dois = []
    removed_noise = []

    # Step 1: Strip zero-width characters and soft hyphens BEFORE any other processing
    text = _ZERO_WIDTH_CHARS.sub('', text)
    text = _SOFT_HYPHEN.sub('', text)

    # Extract URLs before removing
    urls = _URL_PATTERN.findall(text)
    removed_urls = [u.strip() for u in urls if u.strip()]

    # Extract DOIs before removing
    dois = _DOI_PATTERN.findall(text)
    removed_dois = [d.strip() for d in dois if d.strip()]

    retrieval = text

    # Step 2: Remove copyright/reproduction boilerplate blocks (multi-line)
    if _COPYRIGHT_BLOCK.search(retrieval):
        removed_noise.append("copyright_block")
        retrieval = _COPYRIGHT_BLOCK.sub('', retrieval)

    # Step 3: Remove citation boilerplate
    if _CITATION_BOILERPLATE.search(retrieval):
        removed_noise.append("citation_boilerplate")
        retrieval = _CITATION_BOILERPLATE.sub('', retrieval)

    # Step 4: Remove duality of interest boilerplate
    if _DUALITY_BOILERPLATE.search(retrieval):
        removed_noise.append("duality_boilerplate")
        retrieval = _DUALITY_BOILERPLATE.sub('', retrieval)

    # Step 5: Remove member list boilerplate
    if _MEMBER_LIST_BOILERPLATE.search(retrieval):
        removed_noise.append("member_list")
        retrieval = _MEMBER_LIST_BOILERPLATE.sub('', retrieval)

    # Step 6: Remove download stamps
    if _DOWNLOAD_STAMP.search(retrieval):
        removed_noise.append("download_stamp")
        retrieval = _DOWNLOAD_STAMP.sub('', retrieval)

    # Step 7: Remove PDF download URLs
    if _PDF_DOWNLOAD_URL.search(retrieval):
        removed_noise.append("pdf_download_url")
        retrieval = _PDF_DOWNLOAD_URL.sub('', retrieval)

    # Step 8: Remove URL patterns
    retrieval = _URL_PATTERN.sub('', retrieval)

    # Step 9: Remove DOI patterns from retrieval text (keep in metadata)
    retrieval = _DOI_PATTERN.sub('', retrieval)

    # Step 10: Remove permissions email
    if _PERMISSIONS_EMAIL.search(retrieval):
        removed_noise.append("permissions_email")
        retrieval = _PERMISSIONS_EMAIL.sub('', retrieval)

    # Step 11: Remove copyright blocks
    if _COPYRIGHT_PATTERN.search(retrieval):
        removed_noise.append("copyright")
        retrieval = _COPYRIGHT_PATTERN.sub('', retrieval)

    # Step 12: Remove journal headers/footers
    if _HEADER_FOOTER_PATTERN.search(retrieval):
        removed_noise.append("header_footer")
        retrieval = _HEADER_FOOTER_PATTERN.sub('', retrieval)

    # Step 13: Remove standalone page numbers
    if _PAGE_NUM_PATTERN.search(retrieval):
        removed_noise.append("page_numbers")
        retrieval = _PAGE_NUM_PATTERN.sub('', retrieval)

    # Step 14: Remove navigation artifacts
    nav_matches = _NAV_ARTIFACTS.findall(retrieval)
    if nav_matches:
        removed_noise.append("navigation")
        retrieval = _NAV_ARTIFACTS.sub('', retrieval)

    # Step 15: Remove journal footers
    if _JOURNAL_FOOTER.search(retrieval):
        removed_noise.append("journal_footer")
        retrieval = _JOURNAL_FOOTER.sub('', retrieval)

    # Step 16: Remove empty brackets
    retrieval = _EMPTY_BRACKETS.sub('', retrieval)

    # Step 16: Clean up whitespace
    retrieval = _MULTI_WHITESPACE.sub(' ', retrieval)
    retrieval = _MULTI_NEWLINES.sub('\n\n', retrieval)
    retrieval = retrieval.strip()

    # Display text gets same treatment
    display = retrieval

    return {
        "retrieval_text": retrieval,
        "display_text": display,
        "removed_urls": removed_urls,
        "removed_dois": removed_dois,
        "removed_noise": removed_noise,
    }


def extract_source_metadata(text: str, source_meta: Optional[dict] = None) -> dict:
    """Extract citation/source metadata from text.
    
    Returns clean metadata dict with DOI, URLs, organization, etc.
    """
    meta = {}

    # Extract DOIs
    dois = _DOI_PATTERN.findall(text)
    meta["dois"] = [d.strip() for d in dois if d.strip()]

    # Extract URLs
    urls = _URL_PATTERN.findall(text)
    meta["urls"] = [u.strip() for u in urls if u.strip()]

    # Use provided source metadata
    if source_meta:
        meta["source_title"] = source_meta.get("title", "")
        meta["short_title"] = source_meta.get("short_title", "")
        meta["organization"] = source_meta.get("organization", "")
        meta["year"] = source_meta.get("year", 0)
        meta["doi"] = source_meta.get("doi", "")
        meta["official_url"] = source_meta.get("official_url", "")
        meta["authority"] = source_meta.get("authority", "high")
        meta["document_type"] = source_meta.get("document_type", "")

    return meta


def clean_page_text(text: str, structured_lines: list = None) -> dict:
    """Clean a full page of text.
    
    Args:
        text: Raw page text from PDF
        structured_lines: Optional font-enriched lines from parser
    
    Returns:
        dict with cleaned text and metadata
    """
    cleaning_result = clean_for_retrieval(text)

    # Detect headers/footers from structured lines if available
    header_lines = []
    footer_lines = []
    content_lines = []

    if structured_lines:
        for line in structured_lines:
            text_line = line.get("text", "")
            font_size = line.get("font_size", 0)
            is_bold = line.get("is_bold", False)

            # Headers typically: small font, centered, or at top of page
            if (font_size < 7 and text_line.strip()) or \
               _HEADER_FOOTER_PATTERN.search(text_line) or \
               _PAGE_NUM_PATTERN.match(text_line):
                if is_bold and font_size >= 8:
                    content_lines.append(line)  # Bold headings are content
                else:
                    header_lines.append(text_line)
            else:
                content_lines.append(line)

    return {
        "retrieval_text": cleaning_result["retrieval_text"],
        "display_text": cleaning_result["display_text"],
        "removed_urls": cleaning_result["removed_urls"],
        "removed_dois": cleaning_result["removed_dois"],
        "removed_noise": cleaning_result["removed_noise"],
        "header_footer_count": len(header_lines),
        "has_structured_lines": bool(structured_lines),
    }
