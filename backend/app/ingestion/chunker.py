"""Structure-aware chunker for medical documents."""
import re
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 1 token per 4 chars for English)."""
    return max(1, len(text) // 4)


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _make_chunk_id(source_id: str, page: int, idx: int) -> str:
    raw = f"{source_id}_p{page}_c{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _is_table_block(text: str) -> bool:
    """Detect if text block is a table or structured list."""
    lines = text.split("\n")
    if len(lines) < 3:
        return False
    bullet_count = sum(1 for l in lines if re.match(r"^[•\-\*\d]+[\.\)]\s", l.strip()))
    if bullet_count >= 3:
        return True
    if "PROS" in text and "CONS" in text:
        return True
    if "Technical Features" in text:
        return True
    return False


def _split_niddk_table(text: str) -> list[str]:
    """Split NIDDK comparison table by test sections."""
    sections = []
    current = []
    lines = text.split("\n")

    test_markers = ["FPG Test", "OGTT", "A1C Test", "RPG Test", "Comparing Diabetes"]
    current_test = None

    for line in lines:
        stripped = line.strip()
        is_marker = False
        for marker in test_markers:
            if marker in stripped and len(stripped) < 80:
                is_marker = True
                if current:
                    sections.append("\n".join(current))
                    current = []
                current_test = marker
                break
        current.append(line)

    if current:
        sections.append("\n".join(current))

    return [s.strip() for s in sections if s.strip()]


def chunk_page(page: dict, max_tokens: int = 400, overlap_tokens: int = 50) -> list[dict]:
    """Chunk a single page into semantically meaningful pieces."""
    text = page.get("text", "").strip()
    if not text:
        return []

    source_id = page.get("source_id", "unknown")
    page_pdf = page.get("page_pdf", 0)
    section = page.get("section", "")
    subsection = page.get("subsection")

    is_niddk = "niddk" in source_id.lower()
    is_table = _is_table_block(text) or page.get("has_table_keywords", False)

    if is_niddk and is_table:
        logger.info(f"NIDDK table detected on page {page_pdf}, splitting by test section")
        paragraphs = _split_niddk_table(text)
        overlap_tokens = 0
    else:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = _split_by_sentences(text)

    chunks = []
    current_text = ""
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        para = _clean_text(para)
        if not para:
            continue

        para_tokens = _estimate_tokens(para)

        if para_tokens > max_tokens:
            if current_text:
                chunks.append(_build_chunk(
                    current_text, source_id, page_pdf, section, subsection,
                    page, chunk_idx, max_tokens
                ))
                chunk_idx += 1
                current_text = ""
                current_tokens = 0

            sub_chunks = _split_long_text(para, max_tokens, overlap_tokens)
            for sc in sub_chunks:
                chunks.append(_build_chunk(
                    sc, source_id, page_pdf, section, subsection,
                    page, chunk_idx, max_tokens
                ))
                chunk_idx += 1
            continue

        if current_tokens + para_tokens > max_tokens and current_text:
            chunks.append(_build_chunk(
                current_text, source_id, page_pdf, section, subsection,
                page, chunk_idx, max_tokens
            ))
            chunk_idx += 1

            if overlap_tokens > 0 and current_text:
                overlap_text = _get_tail_overlap(current_text, overlap_tokens)
                current_text = overlap_text + " " + para
                current_tokens = _estimate_tokens(current_text)
            else:
                current_text = para
                current_tokens = para_tokens
        else:
            if current_text:
                current_text += " " + para
            else:
                current_text = para
            current_tokens += para_tokens

    if current_text.strip():
        chunks.append(_build_chunk(
            current_text, source_id, page_pdf, section, subsection,
            page, chunk_idx, max_tokens
        ))

    return chunks


def _split_by_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current = ""
    for part in parts:
        if _estimate_tokens(current + " " + part) > 200 and current:
            result.append(current.strip())
            current = part
        else:
            current = current + " " + part if current else part
    if current.strip():
        result.append(current.strip())
    return result if result else [text]


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a long text into chunks respecting sentence boundaries."""
    words = text.split()
    chunks = []
    current_words = []
    current_tokens = 0

    for word in words:
        word_tokens = _estimate_tokens(word)
        if current_tokens + word_tokens > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            if overlap_tokens > 0:
                overlap_words = _get_tail_overlap_words(current_words, overlap_tokens)
                current_words = overlap_words + [word]
                current_tokens = _estimate_tokens(" ".join(current_words))
            else:
                current_words = [word]
                current_tokens = word_tokens
        else:
            current_words.append(word)
            current_tokens += word_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _get_tail_overlap(text: str, overlap_tokens: int) -> str:
    """Get the tail portion of text for overlap."""
    words = text.split()
    overlap_words = _get_tail_overlap_words(words, overlap_tokens)
    return " ".join(overlap_words)


def _get_tail_overlap_words(words: list[str], overlap_tokens: int) -> list[str]:
    """Get words from the tail that fit within overlap token budget."""
    result = []
    token_count = 0
    for word in reversed(words):
        wt = _estimate_tokens(word)
        if token_count + wt > overlap_tokens:
            break
        result.insert(0, word)
        token_count += wt
    return result


def _build_chunk(
    text: str, source_id: str, page_pdf: int, section: str,
    subsection: Optional[str], page: dict, chunk_idx: int, max_tokens: int
) -> dict:
    """Build a chunk dict with full metadata."""
    chunk_id = _make_chunk_id(source_id, page_pdf, chunk_idx)

    return {
        "chunk_id": chunk_id,
        "text": text.strip(),
        "token_estimate": _estimate_tokens(text),
        "source_id": source_id,
        "source_title": page.get("source_title", ""),
        "short_title": page.get("short_title", ""),
        "organization": page.get("organization", ""),
        "document_type": page.get("document_type", ""),
        "page_pdf": page_pdf,
        "page_document": page.get("page_document", page_pdf),
        "section": section,
        "subsection": subsection,
        "doi": page.get("doi"),
        "official_url": page.get("official_url"),
        "year": page.get("year"),
        "authority": page.get("authority", "high"),
        "pdf_file": page.get("pdf_file", ""),
        "has_table": page.get("has_table_keywords", False),
        "is_reference": page.get("is_reference_page", False),
        "num_recommendations": page.get("num_recommendations", 0),
    }


def chunk_documents(pages: list[dict], max_tokens: int = 400, overlap_tokens: int = 50) -> list[dict]:
    """Chunk all pages into text chunks with metadata.

    Source-aware chunking:
    - ADA: uses configured max_tokens (default 700)
    - NIDDK: uses smaller max_tokens (300) to split table by test section
    """
    all_chunks = []

    for page in pages:
        if page.get("is_reference_page", False):
            continue

        source_id = page.get("source_id", "")
        is_niddk = "niddk" in source_id.lower()

        if is_niddk:
            page_chunks = chunk_page(page, max_tokens=300, overlap_tokens=0)
        else:
            page_chunks = chunk_page(page, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        all_chunks.extend(page_chunks)

    return all_chunks


def create_parent_child_chunks(
    pages: list[dict],
    child_max_tokens: int = 300,
    parent_max_tokens: int = 700,
    overlap_tokens: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Create parent and child chunks for hierarchical retrieval.

    Child chunks: small, fine-grained for search
    Parent chunks: larger, provide context for answers

    Returns (child_chunks, parent_chunks)
    """
    parent_chunks = []
    child_chunks = []

    for page in pages:
        if page.get("is_reference_page", False):
            continue

        source_id = page.get("source_id", "unknown")
        page_pdf = page.get("page_pdf", 0)
        section = page.get("section", "")
        subsection = page.get("subsection")
        text = page.get("text", "").strip()

        if not text:
            continue

        is_niddk = "niddk" in source_id.lower()
        is_table = _is_table_block(text) or page.get("has_table_keywords", False)

        if is_niddk and is_table:
            paragraphs = _split_niddk_table(text)
            overlap_tokens = 0
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if len(paragraphs) <= 1:
                paragraphs = _split_by_sentences(text)

        parent_idx = 0
        for para in paragraphs:
            para = _clean_text(para)
            if not para:
                continue

            parent_id = _make_chunk_id(source_id, page_pdf, parent_idx) + "_p"
            parent_chunk = {
                "chunk_id": parent_id,
                "text": para,
                "token_estimate": _estimate_tokens(para),
                "source_id": source_id,
                "source_title": page.get("source_title", ""),
                "short_title": page.get("short_title", ""),
                "organization": page.get("organization", ""),
                "page_pdf": page_pdf,
                "page_document": page.get("page_document", page_pdf),
                "section": section,
                "subsection": subsection,
                "doi": page.get("doi"),
                "official_url": page.get("official_url"),
                "year": page.get("year"),
                "authority": page.get("authority", "high"),
                "is_parent": True,
            }
            parent_chunks.append(parent_chunk)

            child_para_chunks = []
            para_tokens = _estimate_tokens(para)
            if para_tokens > child_max_tokens:
                child_para_chunks = _split_long_text(para, child_max_tokens, overlap_tokens)
            else:
                child_para_chunks = [para]

            for ci, child_text in enumerate(child_para_chunks):
                child_id = _make_chunk_id(source_id, page_pdf, len(child_chunks))
                child_chunk = {
                    "chunk_id": child_id,
                    "text": child_text,
                    "token_estimate": _estimate_tokens(child_text),
                    "source_id": source_id,
                    "source_title": page.get("source_title", ""),
                    "short_title": page.get("short_title", ""),
                    "organization": page.get("organization", ""),
                    "page_pdf": page_pdf,
                    "page_document": page.get("page_document", page_pdf),
                    "section": section,
                    "subsection": subsection,
                    "doi": page.get("doi"),
                    "official_url": page.get("official_url"),
                    "year": page.get("year"),
                    "authority": page.get("authority", "high"),
                    "is_parent": False,
                    "parent_chunk_id": parent_id,
                }
                child_chunks.append(child_chunk)

            parent_idx += 1

    logger.info(
        f"Parent/child chunks: {len(child_chunks)} children, "
        f"{len(parent_chunks)} parents"
    )
    return child_chunks, parent_chunks


def generate_chunk_report(chunks: list[dict], output_path) -> dict:
    """Generate a chunk report."""
    import json
    from collections import Counter

    if not chunks:
        return {"total_chunks": 0}

    doc_counts = Counter(c["source_id"] for c in chunks)
    section_counts = Counter(c["section"] for c in chunks)
    token_lengths = [c["token_estimate"] for c in chunks]
    text_lengths = [len(c["text"]) for c in chunks]
    pages = set()
    for c in chunks:
        pages.add((c["source_id"], c["page_pdf"]))

    empty_chunks = sum(1 for c in chunks if not c["text"].strip())
    table_chunks = sum(1 for c in chunks if c.get("has_table"))

    report = {
        "total_chunks": len(chunks),
        "chunks_per_document": dict(doc_counts),
        "unique_pages": len(pages),
        "pages_per_document": {},
        "sections": dict(section_counts),
        "avg_token_length": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "min_token_length": min(token_lengths) if token_lengths else 0,
        "max_token_length": max(token_lengths) if token_lengths else 0,
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "empty_chunks": empty_chunks,
        "table_chunks": table_chunks,
    }

    pages_per_doc = {}
    for sid, pg in pages:
        pages_per_doc.setdefault(sid, set()).add(pg)
    report["pages_per_document"] = {k: len(v) for k, v in pages_per_doc.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
