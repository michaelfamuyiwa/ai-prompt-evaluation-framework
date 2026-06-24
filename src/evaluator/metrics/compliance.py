"""Instruction compliance evaluation metric."""

from typing import Dict, Any, List
import logging
import re
from ..base_evaluator import BaseMetric

logger = logging.getLogger(__name__)

# Common instruction patterns
NEGATION_PATTERNS = [r"don't\s+", r"do not\s+", r"never\s+", r"avoid\s+"]
FORMAT_PATTERNS = {
    "json": r"\{.*\}",
    "list": r"^\s*[-*].*$|^\s*\d+\.",
    "markdown": r"^\s*#+\s",
    "code_block": r"```",
}


class InstructionComplianceMetric(BaseMetric):
    """Evaluates compliance with explicit instructions in the prompt."""

    def __init__(self, weight: float = 0.20, threshold: float = 3.5):
        super().__init__("instruction_compliance", weight, threshold)

    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate instruction compliance.

        Args:
            prompt: Original prompt containing instructions
            response: Model's response
            ground_truth: Not used

        Returns:
            Dict with compliance score and details
        """
        try:
            compliance_scores = []
            violations = []

            # Check negation instructions ("don't", "never")
            negation_violations = self._check_negations(prompt, response)
            if negation_violations:
                violations.extend(negation_violations)
                compliance_scores.append(2.0)  # Major penalty
            else:
                compliance_scores.append(5.0)

            # Check format instructions
            format_compliance = self._check_format(prompt, response)
            compliance_scores.append(format_compliance)
            if format_compliance < 4.0:
                violations.append("Format requirements not met")

            # Check length constraints
            length_score = self._check_length_constraints(prompt, response)
            compliance_scores.append(length_score)

            avg_score = sum(compliance_scores) / len(compliance_scores)

            return {
                "score": self.validate_score(avg_score),
                "explanation": f"Compliance score: {avg_score:.1f}/5" + (
                    f" - {len(violations)} violation(s)" if violations else " - Fully compliant"
                ),
                "method": "instruction_parsing",
                "details": {
                    "compliance_checks": len(compliance_scores),
                    "violations": violations[:3],  # Top 3 violations
                    "format_compliance": format_compliance,
                    "length_compliance": length_score,
                },
            }
        except Exception as e:
            logger.error(f"Error in compliance evaluation: {e}")
            return {
                "score": 3.0,
                "explanation": f"Evaluation error: {str(e)}",
                "method": "error",
                "details": {},
            }

    def _check_negations(self, prompt: str, response: str) -> List[str]:
        """Check for negation instruction violations."""
        violations = []
        for pattern in NEGATION_PATTERNS:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                # Extract what should be avoided
                for match in matches:
                    avoided_word = prompt.lower().split(match)[-1].split()[0] if prompt.lower().split(match)[-1].split() else ""
                    if avoided_word and avoided_word in response.lower():
                        violations.append(f"Violated: should not mention '{avoided_word}'")
        return violations

    def _check_format(self, prompt: str, response: str) -> float:
        """Check if response meets format requirements."""
        score = 5.0
        for format_type, pattern in FORMAT_PATTERNS.items():
            if format_type in prompt.lower():
                if not re.search(pattern, response, re.MULTILINE):
                    score -= 1.0
        return self.validate_score(score)

    def _check_length_constraints(self, prompt: str, response: str) -> float:
        """Check length constraints (short, long, specific word count)."""
        score = 5.0
        words = len(response.split())

        if "short" in prompt.lower() and words > 100:
            score -= 1.5
        elif "brief" in prompt.lower() and words > 150:
            score -= 1.0
        elif "long" in prompt.lower() and words < 200:
            score -= 1.5
        elif "detailed" in prompt.lower() and words < 150:
            score -= 1.0

        return self.validate_score(score)
