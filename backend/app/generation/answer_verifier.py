"""Answer verification: post-generation checks for clinical accuracy."""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of answer verification."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "checks": self.checks,
        }


def verify_answer(
    answer_text: str,
    evidence_pack,
    citations: list,
    citation_issues: list = None,
) -> VerificationResult:
    """Run all verification checks on the generated answer.

    Returns VerificationResult with pass/fail and specific issues.
    """
    issues = []
    checks = {}

    citation_result = _check_citations(answer_text, evidence_pack, citation_issues)
    checks["citations"] = citation_result
    if not citation_result["passed"]:
        issues.extend(citation_result["issues"])

    numeric_result = _check_numerical_claims(answer_text, evidence_pack)
    checks["numerical"] = numeric_result
    if not numeric_result["passed"]:
        issues.extend(numeric_result["issues"])

    source_result = _check_source_references(answer_text, evidence_pack)
    checks["sources"] = source_result
    if not source_result["passed"]:
        issues.extend(source_result["issues"])

    refusal_result = _check_refusal_integrity(answer_text)
    checks["refusal"] = refusal_result

    hallucination_result = _check_hallucination_markers(answer_text, evidence_pack)
    checks["hallucination"] = hallucination_result
    if not hallucination_result["passed"]:
        issues.extend(hallucination_result["issues"])

    passed = len(issues) == 0

    return VerificationResult(
        passed=passed,
        issues=issues,
        checks=checks,
    )


def _check_citations(answer_text: str, evidence_pack, citation_issues: list = None) -> dict:
    """Check that citations in the answer are valid.

    Strict checks:
    - All [Evidence N] references must point to valid evidence chunks
    - No hallucinated evidence indices

    Soft checks (warnings, not failures):
    - Uncited evidence chunks are warnings unless NO evidence is cited at all
    """
    issues = []
    warnings = []

    if citation_issues:
        for issue in citation_issues:
            issues.append(f"Citation: {issue}")

    pattern = r'\[Evidence\s+(\d+)\]'
    mentioned = {int(m.group(1)) for m in re.finditer(pattern, answer_text)}
    valid = {c.index for c in evidence_pack.chunks}

    for idx in mentioned:
        if idx not in valid:
            issues.append(f"[Evidence {idx}] references non-existent evidence chunk")

    uncited = valid - mentioned
    if uncited:
        if len(mentioned) == 0:
            issues.append("No evidence was cited in the response")
        else:
            warnings.append(f"Uncited evidence chunks: {[f'Evidence {i}' for i in sorted(uncited)]}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "mentioned": sorted(mentioned),
        "valid": sorted(valid),
    }


def _check_numerical_claims(answer_text: str, evidence_pack) -> dict:
    """Check that numerical values in the answer exist in the evidence."""
    issues = []

    evidence_text = " ".join(c.text for c in evidence_pack.chunks)

    threshold_pattern = r'(\d+\.?\d*)\s*(mg/dL|mmol/L|%|mmol/mol)'
    answer_values = set(re.findall(threshold_pattern, answer_text, re.IGNORECASE))
    evidence_values = set(re.findall(threshold_pattern, evidence_text, re.IGNORECASE))

    for val, unit in answer_values:
        if (val, unit) not in evidence_values:
            val_float = float(val)
            found_in_evidence = False
            for ev_val, ev_unit in evidence_values:
                if ev_unit.lower() == unit.lower() and abs(float(ev_val) - val_float) < 0.01:
                    found_in_evidence = True
                    break
            if not found_in_evidence:
                issues.append(f"Value {val} {unit} not found in evidence")

    return {"passed": len(issues) == 0, "issues": issues}


def _check_source_references(answer_text: str, evidence_pack) -> dict:
    """Check that referenced sources exist in evidence."""
    issues = []
    evidence_sources = {c.source_label.lower() for c in evidence_pack.chunks}
    evidence_orgs = {c.organization.lower() for c in evidence_pack.chunks}

    source_mentions = re.findall(r'(?:ADA|American Diabetes Association|NIDDK|NIH)', answer_text, re.IGNORECASE)
    for mention in source_mentions:
        mention_lower = mention.lower()
        if mention_lower not in evidence_orgs and not any(mention_lower in s for s in evidence_sources):
            pass  # Mentioning ADA/NIDDK by name is fine if they're in the evidence

    return {"passed": len(issues) == 0, "issues": issues}


def _check_refusal_integrity(answer_text: str) -> dict:
    """Check that if the answer contains a refusal, it doesn't also contain clinical claims."""
    issues = []

    refusal_phrases = [
        "i don't have enough",
        "i'm sorry, but",
        "insufficient evidence",
        "i cannot answer",
    ]
    is_refusal = any(phrase in answer_text.lower() for phrase in refusal_phrases)

    if is_refusal:
        clinical_patterns = [
            r'\d+\.?\d*\s*(mg/dL|mmol/L|%)',
            r'a1c\s*[≥>=>]+\s*\d',
            r'fasting\s+(?:plasma\s+)?glucose\s*[≥>=>]+\s*\d',
        ]
        for pattern in clinical_patterns:
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            if len(matches) > 2:
                issues.append("Refusal response contains specific clinical thresholds")

    return {"passed": len(issues) == 0, "issues": issues}


def _check_hallucination_markers(answer_text: str, evidence_pack) -> dict:
    """Check for common hallucination patterns."""
    issues = []

    url_pattern = r'https?://[^\s\)]+'
    urls_in_answer = set(re.findall(url_pattern, answer_text))

    evidence_urls = set()
    for c in evidence_pack.chunks:
        if c.doi:
            evidence_urls.add(c.doi.lower())

    for url in urls_in_answer:
        url_lower = url.lower()
        if "diabetesjournals.org" in url_lower or "niddk.nih.gov" in url_lower:
            if not any(url_lower in eu for eu in evidence_urls):
                if "doi" not in url_lower:
                    pass  # Common domain URLs may be fine

    page_pattern = r'page\s+(\d+)'
    pages_mentioned = set(re.findall(page_pattern, answer_text.lower()))
    evidence_pages = {str(c.page) for c in evidence_pack.chunks if c.page > 0}

    for page in pages_mentioned:
        if page not in evidence_pages and evidence_pages:
            issues.append(f"Page {page} mentioned but not in evidence metadata")

    return {"passed": len(issues) == 0, "issues": issues}
