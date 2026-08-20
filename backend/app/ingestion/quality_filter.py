"""Quality filter: deterministic rules for noise detection and removal in ingestion.

Every removed item is optionally logged for debugging.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_noise_chunk(chunk: dict) -> tuple[bool, str]:
    """Determine if a chunk is noise that should be filtered.

    Returns (is_noise, reason).
    """
    text = chunk.get("text", "").strip()

    if len(text) < 30:
        return True, "too_short"

    token_est = chunk.get("token_estimate", len(text) // 4)
    if token_est < 8:
        return True, "too_few_tokens"

    if text.count("http") > 2:
        return True, "url_dominant"

    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    url_chars = sum(len(u) for u in urls)
    if url_chars > len(text) * 0.3:
        return True, "url_heavy"

    if chunk.get("is_reference", False):
        return True, "reference_page"

    nav_pattern = re.compile(
        r'Downloaded from|Published online|Advance online|'
        r'Official journal of|American Diabetes Association',
        re.IGNORECASE
    )
    if nav_pattern.search(text) and len(text) < 100:
        return True, "navigation_noise"

    page_num_pattern = re.compile(r'^\s*S\d{1,3}\s*$')
    if page_num_pattern.match(text):
        return True, "page_number_only"

    section = chunk.get("section", "")
    if section == "References" and token_est < 50:
        return True, "reference_noise"

    return False, ""


def filter_chunks(chunks: list[dict], log_removals: bool = False) -> tuple[list[dict], list[dict]]:
    """Filter noise chunks from a list.

    Returns (kept_chunks, removed_chunks).
    """
    kept = []
    removed = []

    for chunk in chunks:
        is_noise, reason = is_noise_chunk(chunk)
        if is_noise:
            chunk["_removal_reason"] = reason
            removed.append(chunk)
            if log_removals:
                logger.debug(f"Removed chunk {chunk.get('chunk_id', '?')}: {reason}")
        else:
            kept.append(chunk)

    return kept, removed


def dedup_chunks(chunks: list[dict], text_threshold: float = 0.90) -> list[dict]:
    """Remove near-duplicate chunks based on text similarity."""
    if not chunks:
        return []

    seen_texts = []
    deduped = []

    for chunk in chunks:
        text = chunk.get("text", "").lower().strip()
        words = set(text.split())
        is_dup = False

        for prev_words in seen_texts:
            if not prev_words or not words:
                continue
            intersection = len(words & prev_words)
            union = len(words | prev_words)
            if union > 0 and intersection / union >= text_threshold:
                is_dup = True
                break

        if not is_dup:
            deduped.append(chunk)
            seen_texts.append(words)

    return deduped
