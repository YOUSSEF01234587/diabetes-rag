"""Evidence-grounded generation module."""
from .evidence_pack import EvidencePack, EvidenceChunk, build_evidence_pack
from .citation_validator import CitationValidator
from .conflict_detector import ConflictDetector
from .evidence_validator import EvidenceValidator

__all__ = [
    "EvidencePack",
    "EvidenceChunk",
    "build_evidence_pack",
    "CitationValidator",
    "ConflictDetector",
    "EvidenceValidator",
]
