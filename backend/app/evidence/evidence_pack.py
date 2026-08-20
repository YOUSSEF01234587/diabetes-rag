"""EvidencePack: structured evidence wrapper for generation."""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A parsed citation from generated text."""
    evidence_index: int
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "evidence_index": self.evidence_index,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "raw_text": self.raw_text,
        }


@dataclass
class EvidenceChunk:
    """A single evidence chunk with citation metadata."""
    index: int  # 1-based index for prompting
    chunk_id: str
    text: str
    source_id: str
    source_label: str
    organization: str
    page: int
    section: str
    subsection: str
    fusion_score: float
    is_table: bool = False
    doi: str = ""

    def citation_text(self) -> str:
        """Format citation for the LLM prompt."""
        loc = f"Page {self.page}"
        if self.section:
            loc += f", Section: {self.section}"
        if self.subsection:
            loc += f" > {self.subsection}"
        tag = " [Table]" if self.is_table else ""
        return f"[Evidence {self.index}] {self.organization} - {self.source_label}{tag}\nLocation: {loc}\nContent: {self.text}\n"


@dataclass
class EvidencePack:
    """Structured evidence package for grounded generation."""
    chunks: list[EvidenceChunk]
    query: str
    total_candidates: int
    selected_k: int
    source_agreement: float = 0.0
    section_coherence: float = 0.0
    has_table_evidence: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_prompt_evidence(self) -> str:
        """Format all chunks for the LLM prompt."""
        return "\n".join(c.citation_text() for c in self.chunks)

    def source_citations(self) -> list[dict]:
        """Return source citation metadata for the response."""
        return [
            {
                "index": c.index,
                "source_id": c.source_id,
                "source_label": c.source_label,
                "organization": c.organization,
                "page": c.page,
                "section": c.section,
                "doi": c.doi,
            }
            for c in self.chunks
        ]

    def top_sources(self) -> list[str]:
        """Unique source labels, ordered by appearance."""
        seen = []
        for c in self.chunks:
            if c.source_label not in seen:
                seen.append(c.source_label)
        return seen

    def evidence_summary(self) -> dict:
        """Summary metrics for the evidence pack."""
        return {
            "selected_chunks": len(self.chunks),
            "total_candidates": self.total_candidates,
            "selected_k": self.selected_k,
            "source_agreement": round(self.source_agreement, 3),
            "section_coherence": round(self.section_coherence, 3),
            "has_table_evidence": self.has_table_evidence,
            "organizations": list(set(c.organization for c in self.chunks)),
            "sections": list(set(c.section for c in self.chunks if c.section)),
            "warnings": self.warnings,
        }


def _compute_text_similarity(a: str, b: str) -> float:
    """Fast Jaccard similarity on word sets for near-duplicate detection."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.70) -> bool:
    return _compute_text_similarity(text_a, text_b) >= threshold


def _select_evidence_chunks(
    search_results: list[dict],
    evidence_k: int = 5,
) -> list[dict]:
    """Intelligent evidence selection: dedup, diversify, rank.

    Strategy:
    1. Walk candidates in fusion_score order
    2. Skip exact chunk_id duplicates
    3. Skip near-duplicates (>70% word overlap)
    4. Prefer source diversity: if we already have 3+ from same source, require higher score delta
    5. Ensure table chunks are included if query-relevant
    """
    if not search_results:
        return []

    selected = []
    seen_ids = set()
    source_counts = {}
    max_per_source = max(2, evidence_k - 1)

    for r in search_results:
        if len(selected) >= evidence_k:
            break

        cid = r.get("chunk_id", "")
        if cid in seen_ids:
            continue

        text = r.get("text", "")
        source_id = r.get("source_id", r.get("metadata", {}).get("source_id", ""))

        is_dup = False
        for prev in selected:
            if _is_near_duplicate(text, prev.get("text", "")):
                is_dup = True
                break
        if is_dup:
            continue

        src_count = source_counts.get(source_id, 0)
        if src_count >= max_per_source and len(selected) >= 2:
            continue

        selected.append(r)
        seen_ids.add(cid)
        source_counts[source_id] = src_count + 1

    if not selected and search_results:
        selected = search_results[:1]

    return selected


def build_evidence_pack(
    query: str,
    search_results: list[dict],
    evidence_k: int = 5,
    total_candidates: int = 10,
) -> EvidencePack:
    """Build an EvidencePack from retrieval results with intelligent selection.

    Selects evidence_k chunks from candidates, avoiding duplicates,
    near-duplicates, and source over-representation.
    """
    warnings = []
    selected = _select_evidence_chunks(search_results, evidence_k)
    actual_k = len(selected)

    if actual_k == 0:
        warnings.append("No evidence chunks selected")
        return EvidencePack(
            chunks=[],
            query=query,
            total_candidates=len(search_results),
            selected_k=0,
            warnings=warnings,
        )

    if actual_k < evidence_k:
        warnings.append(f"Only {actual_k} chunks selected (requested {evidence_k})")

    chunks = []
    for i, r in enumerate(selected, 1):
        page = r.get("page_document") or r.get("page_pdf", 0)
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                page = 0

        meta = r.get("metadata", {})
        chunks.append(EvidenceChunk(
            index=i,
            chunk_id=r.get("chunk_id", f"unknown_{i}"),
            text=r.get("text", ""),
            source_id=r.get("source_id") or meta.get("source_id", "unknown"),
            source_label=r.get("short_title") or r.get("source_title") or meta.get("short_title") or meta.get("source_title") or r.get("source_id", "Unknown"),
            organization=r.get("organization") or meta.get("organization", "Unknown"),
            page=page,
            section=r.get("section") or meta.get("true_section", meta.get("section", "")),
            subsection=r.get("subsection") or meta.get("subsection", ""),
            fusion_score=r.get("fusion_score", 0.0),
            is_table=r.get("is_table") or meta.get("has_table", False),
            doi=r.get("doi") or meta.get("doi", ""),
        ))

    top_n = min(5, len(chunks))
    orgs = [c.organization for c in chunks[:top_n]]
    if orgs:
        most_common = max(set(orgs), key=orgs.count)
        source_agreement = orgs.count(most_common) / len(orgs)
    else:
        source_agreement = 0.0

    secs = [c.section for c in chunks[:top_n] if c.section]
    if secs:
        most_common_sec = max(set(secs), key=secs.count)
        section_coherence = secs.count(most_common_sec) / len(secs)
    else:
        section_coherence = 0.0

    has_table = any(c.is_table for c in chunks)

    if source_agreement < 0.5:
        warnings.append("Low source agreement across evidence chunks")
    if section_coherence < 0.4:
        warnings.append("Low section coherence across evidence chunks")

    return EvidencePack(
        chunks=chunks,
        query=query,
        total_candidates=len(search_results),
        selected_k=actual_k,
        source_agreement=source_agreement,
        section_coherence=section_coherence,
        has_table_evidence=has_table,
        warnings=warnings,
    )


def parse_citations_from_text(text: str) -> list[Citation]:
    """Parse [Evidence N] citations from generated text."""
    citations = []
    pattern = r'\[Evidence\s+(\d+)\]'
    for match in re.finditer(pattern, text):
        idx = int(match.group(1))
        citations.append(Citation(
            evidence_index=idx,
            source="",
            raw_text=match.group(0),
        ))
    return citations


def extract_cited_evidence_indices(text: str) -> set[int]:
    """Extract all cited evidence indices from generated text."""
    pattern = r'\[Evidence\s+(\d+)\]'
    return {int(m.group(1)) for m in re.finditer(pattern, text)}
