"""ConflictDetector: detects contradictions and conflicts between evidence chunks."""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Conflict:
    """A detected conflict between evidence chunks."""
    chunk_a_index: int
    chunk_b_index: int
    conflict_type: str  # "threshold", "population", "source", "numerical"
    description: str
    severity: str  # "low", "medium", "high"


@dataclass
class ConflictReport:
    """Full conflict detection report."""
    total_conflicts: int
    conflicts: list[Conflict]
    has_population_conflict: bool
    has_threshold_conflict: bool
    has_source_disagreement: bool
    needs_clarification: bool  # True if high-severity conflicts exist

    def to_dict(self) -> dict:
        return {
            "total_conflicts": self.total_conflicts,
            "has_population_conflict": self.has_population_conflict,
            "has_threshold_conflict": self.has_threshold_conflict,
            "has_source_disagreement": self.has_source_disagreement,
            "needs_clarification": self.needs_clarification,
            "conflict_details": [
                {
                    "type": c.conflict_type,
                    "chunks": [c.chunk_a_index, c.chunk_b_index],
                    "description": c.description,
                    "severity": c.severity,
                }
                for c in self.conflicts
            ],
        }


# Patterns that indicate different populations
POPULATION_PATTERNS = {
    "pregnant": r'\bpregnant|pregnancy|gestational|antenatal\b',
    "pediatric": r'\bpediatric|child|children|adolescent|youth\b',
    "adult": r'\badult[s]?\b',
    "elderly": r'\belderly|older adult|geriatric|aged\b',
    "obese": r'\bobes|obesity|bmi\s*[>≥]|overweight\b',
    "high_risk": r'\bhigh.risk|family history|genetic\b',
    "asymptomatic": r'\basymptomatic\b',
}

# Patterns for different test types
TEST_PATTERNS = {
    "a1c": r'\ba1c|hemoglobin a1c|hba1c\b',
    "fg": r'\bfasting glucose|fasting blood glucose|fbg\b',
    "ogtt": r'\bogtt|oral glucose tolerance\b',
    "rgt": r'\brandom glucose\b',
    "any_test": r'\btest|testing|screening\b',
}


class ConflictDetector:
    """Detects conflicts and contradictions between evidence chunks."""

    def __init__(self):
        self.conflicts = []

    def detect(self, evidence_chunks: list[dict]) -> ConflictReport:
        """Run all conflict detection on evidence chunks."""
        self.conflicts = []

        self._detect_population_conflicts(evidence_chunks)
        self._detect_threshold_conflicts(evidence_chunks)
        self._detect_source_disagreements(evidence_chunks)
        self._detect_numerical_conflicts(evidence_chunks)

        has_pop = any(c.conflict_type == "population" for c in self.conflicts)
        has_thresh = any(c.conflict_type == "threshold" for c in self.conflicts)
        has_source = any(c.conflict_type == "source" for c in self.conflicts)
        needs_clarification = any(c.severity == "high" for c in self.conflicts)

        return ConflictReport(
            total_conflicts=len(self.conflicts),
            conflicts=self.conflicts,
            has_population_conflict=has_pop,
            has_threshold_conflict=has_thresh,
            has_source_disagreement=has_source,
            needs_clarification=needs_clarification,
        )

    def _detect_population_conflicts(self, chunks: list[dict]) -> None:
        """Check if chunks refer to different populations."""
        populations_by_chunk = {}
        for chunk in chunks:
            idx = chunk.get("index", 0)
            text = chunk.get("text", "").lower()
            pops = set()
            for pop_name, pattern in POPULATION_PATTERNS.items():
                if re.search(pattern, text):
                    pops.add(pop_name)
            populations_by_chunk[idx] = pops

        # Compare pairs
        indices = sorted(populations_by_chunk.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_a, idx_b = indices[i], indices[j]
                pops_a = populations_by_chunk[idx_a]
                pops_b = populations_by_chunk[idx_b]

                if pops_a and pops_b and pops_a != pops_b:
                    overlap = pops_a & pops_b
                    if not overlap:
                        self.conflicts.append(Conflict(
                            chunk_a_index=idx_a,
                            chunk_b_index=idx_b,
                            conflict_type="population",
                            description=f"Chunks refer to different populations: {pops_a} vs {pops_b}",
                            severity="high",
                        ))

    def _detect_threshold_conflicts(self, chunks: list[dict]) -> None:
        """Check for conflicting numeric thresholds."""
        threshold_pattern = r'(\d+\.?\d*)\s*(mg/dL|mmol/L|%)'
        thresholds_by_chunk = {}
        for chunk in chunks:
            idx = chunk.get("index", 0)
            text = chunk.get("text", "")
            matches = re.findall(threshold_pattern, text, re.IGNORECASE)
            if matches:
                thresholds_by_chunk[idx] = [(float(v), u) for v, u in matches]

        indices = sorted(thresholds_by_chunk.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_a, idx_b = indices[i], indices[j]
                vals_a = set(thresholds_by_chunk[idx_a])
                vals_b = set(thresholds_by_chunk[idx_b])

                # Same unit but different values → possible conflict
                units_a = {u for _, u in vals_a}
                units_b = {u for _, u in vals_b}
                common_units = units_a & units_b
                for unit in common_units:
                    nums_a = {v for v, u in vals_a if u == unit}
                    nums_b = {v for v, u in vals_b if u == unit}
                    if nums_a != nums_b and nums_a and nums_b:
                        self.conflicts.append(Conflict(
                            chunk_a_index=idx_a,
                            chunk_b_index=idx_b,
                            conflict_type="threshold",
                            description=f"Different {unit} thresholds: {sorted(nums_a)} vs {sorted(nums_b)}",
                            severity="medium",
                        ))

    def _detect_source_disagreements(self, chunks: list[dict]) -> None:
        """Check if ADA and NIDDK sources give different information."""
        sources_by_chunk = {}
        for chunk in chunks:
            idx = chunk.get("index", 0)
            org = chunk.get("organization", "").upper()
            if "ADA" in org or "AMERICAN" in org:
                sources_by_chunk[idx] = "ADA"
            elif "NIDDK" in org or "NIH" in org:
                sources_by_chunk[idx] = "NIDDK"
            else:
                sources_by_chunk[idx] = "OTHER"

        # Different organizations covering same topic
        orgs = set(sources_by_chunk.values())
        if "ADA" in orgs and "NIDDK" in orgs:
            ada_chunks = [i for i, o in sources_by_chunk.items() if o == "ADA"]
            niddk_chunks = [i for i, o in sources_by_chunk.items() if o == "NIDDK"]
            if ada_chunks and niddk_chunks:
                self.conflicts.append(Conflict(
                    chunk_a_index=ada_chunks[0],
                    chunk_b_index=niddk_chunks[0],
                    conflict_type="source",
                    description="ADA and NIDDK sources may use different terminology or thresholds",
                    severity="low",
                ))

    def _detect_numerical_conflicts(self, chunks: list[dict]) -> None:
        """Detect specific numerical conflicts (e.g., A1C thresholds)."""
        a1c_pattern = r'a1c\s*[≥>=>]+\s*(\d+\.?\d*)\s*%'

        a1c_by_chunk = {}
        for chunk in chunks:
            idx = chunk.get("index", 0)
            text = chunk.get("text", "").lower()
            matches = re.findall(a1c_pattern, text)
            if matches:
                a1c_by_chunk[idx] = [float(m) for m in matches]

        indices = sorted(a1c_by_chunk.keys())
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_a, idx_b = indices[i], indices[j]
                vals_a = set(a1c_by_chunk[idx_a])
                vals_b = set(a1c_by_chunk[idx_b])
                if vals_a != vals_b and vals_a and vals_b:
                    self.conflicts.append(Conflict(
                        chunk_a_index=idx_a,
                        chunk_b_index=idx_b,
                        conflict_type="numerical",
                        description=f"Conflicting A1C thresholds: {sorted(vals_a)}% vs {sorted(vals_b)}%",
                        severity="high",
                    ))
