"""Main evaluation framework orchestrating all metrics."""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .metrics import (
    FactualAccuracyMetric,
    HallucinationMetric,
    ReasoningDepthMetric,
    InstructionComplianceMetric,
    CoherenceMetric,
    SafetyMetric,
)

logger = logging.getLogger(__name__)


class AIPromptEvaluationFramework:
    """Main framework for evaluating LLM responses across multiple dimensions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation framework.

        Args:
            config: Optional configuration dict with metric weights and thresholds
        """
        self.config = config or {}
        self._initialize_metrics()
        logger.info("AI Prompt Evaluation Framework initialized")

    def _initialize_metrics(self):
        """Initialize all evaluation metrics."""
        config = self.config.get("metrics", {})

        self.metrics = [
            FactualAccuracyMetric(
                weight=config.get("factual_accuracy", {}).get("weight", 0.25),
                threshold=config.get("factual_accuracy", {}).get("threshold", 3.5),
            ),
            HallucinationMetric(
                weight=config.get("hallucination_severity", {}).get("weight", 0.20),
                threshold=config.get("hallucination_severity", {}).get("threshold", 2.0),
            ),
            ReasoningDepthMetric(
                weight=config.get("reasoning_depth", {}).get("weight", 0.15),
                threshold=config.get("reasoning_depth", {}).get("threshold", 3.0),
            ),
            InstructionComplianceMetric(
                weight=config.get("instruction_compliance", {}).get("weight", 0.20),
                threshold=config.get("instruction_compliance", {}).get("threshold", 3.5),
            ),
            CoherenceMetric(
                weight=config.get("coherence", {}).get("weight", 0.10),
                threshold=config.get("coherence", {}).get("threshold", 3.5),
            ),
            SafetyMetric(
                weight=config.get("safety", {}).get("weight", 0.10),
                threshold=config.get("safety", {}).get("threshold", 4.0),
            ),
        ]

    def evaluate_response(
        self, prompt: str, response: str, ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single LLM response across all metrics.

        Args:
            prompt: Original prompt
            response: Model's response
            ground_truth: Reference answer (optional)

        Returns:
            Comprehensive evaluation report
        """
        start_time = time.time()
        logger.info(f"Starting evaluation of response (length: {len(response)} chars)")

        metrics_scores = {}
        metrics_details = {}
        failed_metrics = []

        # Run each metric
        for metric in self.metrics:
            try:
                result = metric.evaluate(prompt, response, ground_truth)
                metrics_scores[metric.name] = result["score"]
                metrics_details[metric.name] = result
            except Exception as e:
                logger.error(f"Metric {metric.name} failed: {e}")
                metrics_scores[metric.name] = 2.0  # Default low score on error
                failed_metrics.append(metric.name)

        # Calculate weighted average
        total_weight = sum(m.weight for m in self.metrics)
        weighted_sum = sum(
            metrics_scores[m.name] * m.weight for m in self.metrics
        )
        average_alignment_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Determine status
        status = "PASSED (Gold Standard)" if average_alignment_score >= 4.0 else "NEEDS REVIEW"
        if average_alignment_score < 2.5:
            status = "REJECTED"

        elapsed_time = time.time() - start_time

        report = {
            "timestamp": datetime.now().isoformat(),
            "prompt_analyzed": prompt,
            "response_analyzed": response,
            "metrics_scores": metrics_scores,
            "metrics_details": metrics_details,
            "average_alignment_score": round(average_alignment_score, 2),
            "status": status,
            "evaluation_time_seconds": round(elapsed_time, 2),
            "failed_metrics": failed_metrics,
        }

        logger.info(f"Evaluation complete: {status} (score: {average_alignment_score:.2f})")
        return report

    def batch_evaluate(
        self, evaluations: List[Dict[str, str]], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses.

        Args:
            evaluations: List of dicts with 'prompt', 'response', optional 'ground_truth'
            show_progress: Show progress bar

        Returns:
            List of evaluation reports
        """
        results = []
        iterator = evaluations

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(evaluations, desc="Evaluating responses")
            except ImportError:
                pass

        for item in iterator:
            report = self.evaluate_response(
                prompt=item["prompt"],
                response=item["response"],
                ground_truth=item.get("ground_truth"),
            )
            results.append(report)

        return results
