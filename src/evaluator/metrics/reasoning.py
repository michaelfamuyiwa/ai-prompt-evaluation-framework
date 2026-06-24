"""Reasoning depth evaluation metric."""

from typing import Dict, Any
import logging
from ..base_evaluator import BaseMetric

logger = logging.getLogger(__name__)

REASONING_INDICATORS = [
    "because",
    "therefore",
    "thus",
    "consequently",
    "as a result",
    "however",
    "although",
    "while",
    "in contrast",
    "for example",
    "specifically",
    "in particular",
    "this suggests",
    "which implies",
    "evidence",
    "research shows",
]


class ReasoningDepthMetric(BaseMetric):
    """Evaluates the depth and quality of reasoning in the response."""

    def __init__(self, weight: float = 0.15, threshold: float = 3.0):
        super().__init__("reasoning_depth", weight, threshold)

    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate reasoning depth based on linguistic markers and structure.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Not used for this metric

        Returns:
            Dict with score and reasoning depth analysis
        """
        try:
            response_lower = response.lower()
            words = response.split()
            sentences = response.split(".")

            # Count reasoning indicators
            reasoning_count = sum(
                response_lower.count(indicator) for indicator in REASONING_INDICATORS
            )

            # Score based on multiple factors
            length_score = min(2.0, len(words) / 100.0)  # Points for length
            reasoning_score = min(2.0, reasoning_count / 3.0)  # Points for reasoning markers
            sentence_score = min(1.0, len(sentences) / 5.0)  # Points for structural complexity

            total_score = length_score + reasoning_score + sentence_score
            total_score = self.validate_score(total_score)

            return {
                "score": total_score,
                "explanation": f"Reasoning depth: {reasoning_count} logical connectors detected",
                "method": "linguistic_analysis",
                "details": {
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "reasoning_indicators": reasoning_count,
                    "indicators_found": [ind for ind in REASONING_INDICATORS if ind in response_lower][
                        :5
                    ],
                },
            }
        except Exception as e:
            logger.error(f"Error in reasoning depth evaluation: {e}")
            return {
                "score": 2.0,
                "explanation": f"Evaluation error: {str(e)}",
                "method": "error",
                "details": {},
            }
