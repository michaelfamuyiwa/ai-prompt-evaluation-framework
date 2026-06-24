"""Factual accuracy evaluation metric using semantic similarity."""

from typing import Dict, Any
import logging
from sentence_transformers import util
from ..base_evaluator import BaseMetric
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


class FactualAccuracyMetric(BaseMetric):
    """Evaluates factual accuracy using semantic similarity to ground truth."""

    def __init__(self, weight: float = 0.25, threshold: float = 3.5):
        super().__init__("factual_accuracy", weight, threshold)

    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate factual accuracy using semantic similarity.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Reference answer

        Returns:
            Dict with score and explanation
        """
        if not ground_truth:
            logger.warning("No ground truth provided for factual accuracy evaluation")
            return {
                "score": 3.0,
                "explanation": "No ground truth available for comparison",
                "method": "N/A",
                "details": {},
            }

        try:
            # Get embeddings
            response_embedding = get_embeddings(response)
            truth_embedding = get_embeddings(ground_truth)

            # Calculate semantic similarity
            similarity = float(util.pytorch_cos_sim(response_embedding, truth_embedding)[0][0])

            # Map similarity [0-1] to score [0-5]
            # 0.9+ = 5.0, 0.7-0.9 = 4.0, 0.5-0.7 = 3.0, etc.
            if similarity >= 0.9:
                score = 5.0
            elif similarity >= 0.75:
                score = 4.0
            elif similarity >= 0.6:
                score = 3.0
            elif similarity >= 0.4:
                score = 2.0
            else:
                score = 1.0

            return {
                "score": self.validate_score(score),
                "explanation": f"Semantic similarity to ground truth: {similarity:.2%}",
                "method": "cosine_similarity",
                "details": {"similarity_score": round(similarity, 3), "threshold_used": 0.6},
            }
        except Exception as e:
            logger.error(f"Error in factual accuracy evaluation: {e}")
            return {
                "score": 2.0,
                "explanation": f"Evaluation error: {str(e)}",
                "method": "error",
                "details": {},
            }
