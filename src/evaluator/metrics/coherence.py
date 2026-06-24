"""Coherence evaluation metric."""

from typing import Dict, Any
import logging
from sentence_transformers import util
from ..base_evaluator import BaseMetric
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


class CoherenceMetric(BaseMetric):
    """Evaluates semantic coherence and consistency within the response."""

    def __init__(self, weight: float = 0.10, threshold: float = 3.5):
        super().__init__("coherence", weight, threshold)

    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate coherence of the response.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Not used

        Returns:
            Dict with coherence score and analysis
        """
        try:
            sentences = [
                s.strip() for s in response.split(".") if s.strip()
            ]  # Split into sentences

            if len(sentences) < 2:
                # Single sentence response
                return {
                    "score": 4.0,
                    "explanation": "Single sentence response - coherence not fully assessable",
                    "method": "sentence_analysis",
                    "details": {"sentence_count": len(sentences)},
                }

            # Calculate coherence by measuring similarity between consecutive sentences
            coherence_scores = []
            for i in range(len(sentences) - 1):
                try:
                    emb1 = get_embeddings(sentences[i])
                    emb2 = get_embeddings(sentences[i + 1])
                    similarity = float(util.pytorch_cos_sim(emb1, emb2)[0][0])
                    coherence_scores.append(similarity)
                except:
                    continue

            if not coherence_scores:
                return {
                    "score": 3.0,
                    "explanation": "Could not compute sentence coherence",
                    "method": "error",
                    "details": {},
                }

            # Map average similarity to score
            avg_coherence = sum(coherence_scores) / len(coherence_scores)

            if avg_coherence >= 0.7:
                score = 5.0
            elif avg_coherence >= 0.55:
                score = 4.0
            elif avg_coherence >= 0.4:
                score = 3.0
            elif avg_coherence >= 0.25:
                score = 2.0
            else:
                score = 1.0

            return {
                "score": self.validate_score(score),
                "explanation": f"Semantic coherence: {avg_coherence:.2%} average sentence similarity",
                "method": "sentence_similarity",
                "details": {
                    "sentence_count": len(sentences),
                    "coherence_index": round(avg_coherence, 3),
                    "min_coherence": round(min(coherence_scores), 3),
                    "max_coherence": round(max(coherence_scores), 3),
                },
            }
        except Exception as e:
            logger.error(f"Error in coherence evaluation: {e}")
            return {
                "score": 2.0,
                "explanation": f"Evaluation error: {str(e)}",
                "method": "error",
                "details": {},
            }
