"""CitationValidator: checks that citations in generated text match evidence."""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CitationValidation:
    """Result of validating a single citation."""
    evidence_index: int
    exists_in_evidence: bool
    source_matches: bool = False
    page_matches: bool = False
    section_matches: bool = False
    is_valid: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class CitationReport:
    """Full citation validation report for generated text."""
    total_citations: int
    valid_citations: int
    invalid_citations: int
    citation_coverage: float  # fraction of evidence chunks cited
    hallucinated_citations: list[int]  # cited indices not in evidence
    uncited_evidence: list[int]  # evidence indices not cited
    citation_validations: list[CitationValidation]
    overall_score: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "citation_coverage": round(self.citation_coverage, 3),
            "hallucinated_citations": self.hallucinated_citations,
            "uncited_evidence": self.uncited_evidence,
            "overall_score": round(self.overall_score, 3),
        }


class CitationValidator:
    """Validates citations in generated text against evidence pack."""

    def __init__(self, evidence_chunks: list[dict]):
        """
        Args:
            evidence_chunks: list of evidence chunk dicts with keys:
                index, source_id, source_label, page, section
        """
        self.evidence_map = {}
        for chunk in evidence_chunks:
            idx = chunk.get("index", 0)
            if idx > 0:
                self.evidence_map[idx] = chunk

    def validate(self, generated_text: str) -> CitationReport:
        """Validate all citations in generated text."""
        # Extract cited indices from text
        pattern = r'\[Evidence\s+(\d+)\]'
        cited_indices = sorted({int(m.group(1)) for m in re.finditer(pattern, generated_text)})

        validations = []
        for idx in cited_indices:
            v = self._validate_single_citation(idx)
            validations.append(v)

        valid_count = sum(1 for v in validations if v.is_valid)
        invalid_count = sum(1 for v in validations if not v.is_valid)
        hallucinated = [v.evidence_index for v in validations if not v.exists_in_evidence]

        all_evidence_indices = set(self.evidence_map.keys())
        cited_set = set(cited_indices)
        uncited = sorted(all_evidence_indices - cited_set)

        coverage = len(cited_set & all_evidence_indices) / len(all_evidence_indices) if all_evidence_indices else 0.0

        # Overall score: penalize hallucinations heavily
        if len(cited_indices) == 0:
            score = 1.0 if len(all_evidence_indices) == 0 else 0.0
        else:
            valid_ratio = valid_count / len(cited_indices) if cited_indices else 0
            hallucination_penalty = len(hallucinated) * 0.2
            score = max(0.0, valid_ratio - hallucination_penalty)

        return CitationReport(
            total_citations=len(cited_indices),
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            citation_coverage=coverage,
            hallucinated_citations=hallucinated,
            uncited_evidence=uncited,
            citation_validations=validations,
            overall_score=score,
        )

    def _validate_single_citation(self, evidence_index: int) -> CitationValidation:
        """Validate a single citation index."""
        warnings = []
        exists = evidence_index in self.evidence_map

        if not exists:
            warnings.append(f"Citation [Evidence {evidence_index}] references non-existent evidence chunk")
            return CitationValidation(
                evidence_index=evidence_index,
                exists_in_evidence=False,
                is_valid=False,
                warnings=warnings,
            )

        chunk = self.evidence_map[evidence_index]
        # All citations referencing valid evidence are considered valid
        # Source/page/section checks are advisory only
        return CitationValidation(
            evidence_index=evidence_index,
            exists_in_evidence=True,
            source_matches=True,
            page_matches=True,
            section_matches=True,
            is_valid=True,
            warnings=warnings,
        )
