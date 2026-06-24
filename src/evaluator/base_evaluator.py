"""Base class for all evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, name: str, weight: float = 1.0, threshold: float = 3.0):
        """
        Initialize a metric.

        Args:
            name: Name of the metric
            weight: Relative weight for aggregation (0-1)
            threshold: Minimum acceptable score (0-5)
        """
        self.name = name
        self.weight = weight
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, prompt: str, response: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Evaluate a single response.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Reference answer (if applicable)

        Returns:
            Dict with 'score' (0-5), 'explanation', and metadata
        """
        pass

    def validate_score(self, score: float) -> float:
        """Ensure score is within valid range [0, 5]."""
        return max(0.0, min(5.0, float(score)))
