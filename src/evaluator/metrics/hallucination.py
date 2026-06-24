"""Hallucination severity evaluation metric."""

from typing import Dict, Any
import logging
from sentence_transformers import util
from ..base_evaluator import BaseMetric
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


class HallucinationMetric(BaseMetric):
    """Detects hallucinations by measuring semantic drift from ground truth."""

    def __init__(self, weight: float = 0.20, threshold: float = 2.0):
        super().__init__("hallucination_severity", weight, threshold)

    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate hallucination severity.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Reference answer

        Returns:
            Dict with score (0 = no hallucination, 5 = severe hallucination) and details
        """
        if not ground_truth:
            # Without ground truth, analyze internal consistency
            return self._evaluate_consistency(response)

        try:
            response_embedding = get_embeddings(response)
            truth_embedding = get_embeddings(ground_truth)
            prompt_embedding = get_embeddings(prompt)

            # Semantic drift: distance from ground truth
            accuracy_similarity = float(util.pytorch_cos_sim(response_embedding, truth_embedding)[0][0])

            # Topic coherence: how related response is to prompt
            topic_similarity = float(util.pytorch_cos_sim(response_embedding, prompt_embedding)[0][0])

            # Hallucination score (inverse of accuracy)
            hallucination_score = (1.0 - accuracy_similarity) * 5.0

            # Penalize off-topic responses
            if topic_similarity < 0.3:
                hallucination_score = min(5.0, hallucination_score + 1.5)

            return {
                "score": self.validate_score(5.0 - hallucination_score),  # Higher score = less hallucination
                "explanation": f"Hallucination severity: {hallucination_score:.1f}/5",
                "method": "semantic_drift",
                "details": {
                    "accuracy_similarity": round(accuracy_similarity, 3),
                    "topic_coherence": round(topic_similarity, 3),
                    "hallucination_index": round(hallucination_score, 3),
                },
            }
        except Exception as e:
            logger.error(f"Error in hallucination evaluation: {e}")
            return {
                "score": 2.0,
                "explanation": f"Evaluation error: {str(e)}",
                "method": "error",
                "details": {},
            }

    def _evaluate_consistency(self, response: str) -> Dict[str, Any]:
        """Evaluate internal consistency without ground truth."""
        # Check for obvious contradictions in the response
        words = response.lower().split()
        contradiction_phrases = [
            ("but not", "but"),
            ("however no", "however"),
            ("yet not", "yet"),
        ]

        contradiction_count = 0
        for phrase, _ in contradiction_phrases:
            contradiction_count += response.lower().count(phrase)

        # Penalty for contradictions
        contradiction_score = min(2.0, contradiction_count * 0.5)
        score = 4.0 - contradiction_score  # Default to 4.0, penalize contradictions

        return {
            "score": self.validate_score(score),
            "explanation": "Internal consistency evaluation (no ground truth)",
            "method": "consistency_check",
            "details": {"contradiction_count": contradiction_count},
        }
