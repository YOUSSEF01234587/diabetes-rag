"""EvidenceValidator: orchestrator for evidence validation, citation, and conflict detection."""
import logging
from dataclasses import dataclass, field

from .evidence_pack import EvidencePack, build_evidence_pack
from .citation_validator import CitationValidator, CitationReport
from .conflict_detector import ConflictDetector, ConflictReport

logger = logging.getLogger(__name__)


@dataclass
class EvidenceValidationResult:
    """Complete evidence validation result."""
    evidence_pack: EvidencePack
    citation_report: CitationReport
    conflict_report: ConflictReport
    is_grounded: bool  # overall groundedness assessment
    grounding_score: float  # 0-1 composite score
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_summary": self.evidence_pack.evidence_summary(),
            "citation_report": self.citation_report.to_dict(),
            "conflict_report": self.conflict_report.to_dict(),
            "is_grounded": self.is_grounded,
            "grounding_score": round(self.grounding_score, 3),
            "warnings": self.warnings,
        }


class EvidenceValidator:
    """Validates evidence packs for grounded generation."""

    def __init__(self, evidence_k: int = 5):
        self.evidence_k = evidence_k
        self.conflict_detector = ConflictDetector()

    def build_and_validate(
        self,
        query: str,
        search_results: list[dict],
        generated_text: str = "",
    ) -> EvidenceValidationResult:
        """Build evidence pack and validate it.

        Args:
            query: user query
            search_results: retrieval results (top candidates)
            generated_text: if provided, validate citations in this text
        """
        # Build evidence pack
        pack = build_evidence_pack(
            query=query,
            search_results=search_results,
            evidence_k=self.evidence_k,
            total_candidates=len(search_results),
        )

        # Citation validation
        evidence_dicts = [
            {
                "index": c.index,
                "source_id": c.source_id,
                "source_label": c.source_label,
                "page": c.page,
                "section": c.section,
            }
            for c in pack.chunks
        ]
        citation_validator = CitationValidator(evidence_dicts)

        if generated_text:
            citation_report = citation_validator.validate(generated_text)
        else:
            citation_report = CitationReport(
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                citation_coverage=0.0,
                hallucinated_citations=[],
                uncited_evidence=[c.index for c in pack.chunks],
                citation_validations=[],
                overall_score=0.0,
            )

        # Conflict detection
        evidence_for_conflicts = [
            {
                "index": c.index,
                "text": c.text,
                "source_id": c.source_id,
                "organization": c.organization,
                "section": c.section,
            }
            for c in pack.chunks
        ]
        conflict_report = self.conflict_detector.detect(evidence_for_conflicts)

        # Compute overall grounding
        grounding_score = self._compute_grounding_score(pack, citation_report, conflict_report)
        is_grounded = grounding_score >= 0.6

        warnings = list(pack.warnings)
        if citation_report.hallucinated_citations:
            warnings.append(f"Hallucinated citations: {citation_report.hallucinated_citations}")
        if conflict_report.needs_clarification:
            warnings.append("High-severity conflicts detected - needs clarification")

        return EvidenceValidationResult(
            evidence_pack=pack,
            citation_report=citation_report,
            conflict_report=conflict_report,
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            warnings=warnings,
        )

    def validate_post_generation(
        self,
        evidence_result: EvidenceValidationResult,
        generated_text: str,
    ) -> EvidenceValidationResult:
        """Re-validate citations after generation is complete."""
        evidence_dicts = [
            {
                "index": c.index,
                "source_id": c.source_id,
                "source_label": c.source_label,
                "page": c.page,
                "section": c.section,
            }
            for c in evidence_result.evidence_pack.chunks
        ]
        citation_validator = CitationValidator(evidence_dicts)
        citation_report = citation_validator.validate(generated_text)

        grounding_score = self._compute_grounding_score(
            evidence_result.evidence_pack,
            citation_report,
            evidence_result.conflict_report,
        )

        warnings = list(evidence_result.evidence_pack.warnings)
        if citation_report.hallucinated_citations:
            warnings.append(f"Hallucinated citations: {citation_report.hallucinated_citations}")

        return EvidenceValidationResult(
            evidence_pack=evidence_result.evidence_pack,
            citation_report=citation_report,
            conflict_report=evidence_result.conflict_report,
            is_grounded=grounding_score >= 0.6,
            grounding_score=grounding_score,
            warnings=warnings,
        )

    def _compute_grounding_score(
        self,
        pack: EvidencePack,
        citation_report: CitationReport,
        conflict_report: ConflictReport,
    ) -> float:
        """Compute composite grounding score."""
        # Evidence quality (40%) - how good is the retrieved evidence
        evidence_score = min(1.0, pack.source_agreement * 0.5 + pack.section_coherence * 0.5)
        if len(pack.chunks) >= 3:
            evidence_score = min(1.0, evidence_score + 0.2)

        # Citation quality (25%) - did the LLM cite evidence
        citation_score = citation_report.overall_score

        # Conflict penalty (20%)
        conflict_penalty = 0.0
        if conflict_report.needs_clarification:
            conflict_penalty = 0.3
        elif conflict_report.total_conflicts > 0:
            conflict_penalty = 0.1
        conflict_score = max(0.0, 1.0 - conflict_penalty)

        # Coverage (15%) - use neutral default when no citations parsed
        coverage_score = citation_report.citation_coverage if citation_report.total_citations > 0 else 0.6

        composite = (
            evidence_score * 0.40 +
            citation_score * 0.25 +
            conflict_score * 0.20 +
            coverage_score * 0.15
        )
        return max(0.0, min(1.0, composite))
