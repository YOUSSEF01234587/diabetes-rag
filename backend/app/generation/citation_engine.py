"""Citation engine: builds citations programmatically from evidence metadata."""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A structured citation for a clinical claim."""
    evidence_index: int
    source_id: str
    source_title: str
    organization: str
    page: int
    section: str
    subsection: str = ""
    doi: str = ""
    official_url: str = ""
    is_table: bool = False
    raw_evidence_text: str = ""

    def to_display(self) -> str:
        """Format for user-facing display."""
        parts = [self.source_title]
        if self.section:
            parts.append(self.section)
        if self.page:
            parts.append(f"p. {self.page}")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "evidence_index": self.evidence_index,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "organization": self.organization,
            "page": self.page,
            "section": self.section,
            "subsection": self.subsection,
            "doi": self.doi,
            "official_url": self.official_url,
            "is_table": self.is_table,
        }


@dataclass
class CitationValidationResult:
    """Result of validating citations in generated text."""
    total_citations_mentioned: int
    valid_citations: int
    invalid_citations: int
    hallucinated_indices: list[int]
    uncited_evidence: list[int]
    issues: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def passed(self) -> bool:
        return self.invalid_citations == 0 and len(self.hallucinated_indices) == 0

    def to_dict(self) -> dict:
        return {
            "total_citations_mentioned": self.total_citations_mentioned,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "hallucinated_indices": self.hallucinated_indices,
            "uncited_evidence": self.uncited_evidence,
            "issues": self.issues,
            "score": round(self.score, 3),
        }


def build_citations_from_evidence(evidence_pack) -> list[Citation]:
    """Build programmatic citations from the evidence pack.

    These citations are derived from actual retrieval metadata,
    NOT from what the LLM claims in its response.

    Duplicates (same source_id + page) are merged into single citations
    to avoid showing redundant entries to the user.
    """
    raw_citations = []
    for chunk in evidence_pack.chunks:
        raw_citations.append(Citation(
            evidence_index=chunk.index,
            source_id=chunk.source_id,
            source_title=chunk.source_label,
            organization=chunk.organization,
            page=chunk.page,
            section=chunk.section,
            subsection=chunk.subsection,
            doi=chunk.doi,
            is_table=chunk.is_table,
            raw_evidence_text=chunk.text[:200],
        ))

    return _deduplicate_citations(raw_citations)


def _deduplicate_citations(citations: list[Citation]) -> list[Citation]:
    """Deduplicate citations by source_id + page.

    Merges multiple evidence chunks from the same page into a single citation.
    Preserves the lowest evidence_index for each group (for [Evidence N] traceability).
    Returns deduplicated citations sorted by evidence_index.
    """
    if not citations:
        return []

    groups: dict[tuple, list[Citation]] = {}
    for c in citations:
        key = (c.source_id, c.page)
        if key not in groups:
            groups[key] = []
        groups[key].append(c)

    deduped = []
    for key, group in groups.items():
        primary = min(group, key=lambda c: c.evidence_index)
        if len(group) > 1:
            evidence_indices = sorted(c.evidence_index for c in group)
            primary.raw_evidence_text = (
                f"[From {len(group)} chunks: Evidence {', '.join(str(i) for i in evidence_indices)}] "
                + primary.raw_evidence_text
            )
        deduped.append(primary)

    deduped.sort(key=lambda c: c.evidence_index)
    return deduped


def validate_answer_citations(answer_text: str, evidence_pack) -> CitationValidationResult:
    """Validate that citations in the generated text match actual evidence.

    Checks:
    1. Every [Evidence N] in the text references an existing evidence chunk
    2. No hallucinated evidence indices exist
    """
    pattern = r'\[Evidence\s+(\d+)\]'
    mentioned = sorted({int(m.group(1)) for m in re.finditer(pattern, answer_text)})

    valid_indices = {c.index for c in evidence_pack.chunks}
    hallucinated = [idx for idx in mentioned if idx not in valid_indices]
    uncited = sorted(valid_indices - set(mentioned))

    valid_count = len(mentioned) - len(hallucinated)
    invalid_count = len(hallucinated)

    issues = []
    if hallucinated:
        issues.append(f"Hallucinated citations: {[f'Evidence {i}' for i in hallucinated]}")
    if uncited and len(uncited) == len(valid_indices):
        issues.append("No evidence was cited in the response")
    elif uncited:
        issues.append(f"Uncited evidence chunks: {[f'Evidence {i}' for i in uncited]}")

    if len(mentioned) == 0:
        score = 0.0
    else:
        score = valid_count / len(mentioned) if mentioned else 0.0
        if hallucinated:
            score = max(0.0, score - len(hallucinated) * 0.2)

    return CitationValidationResult(
        total_citations_mentioned=len(mentioned),
        valid_citations=valid_count,
        invalid_citations=invalid_count,
        hallucinated_indices=hallucinated,
        uncited_evidence=uncited,
        issues=issues,
        score=score,
    )
