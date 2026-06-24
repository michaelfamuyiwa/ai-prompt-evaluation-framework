"""Integration tests for the evaluation framework."""

import pytest
from src.evaluator import AIPromptEvaluationFramework


class TestFramework:
    """Integration tests for the framework."""

    def test_single_evaluation(self, framework):
        """Test single response evaluation."""
        result = framework.evaluate_response(
            prompt="Explain quantum computing.",
            response="Quantum computing uses quantum bits to perform computations.",
            ground_truth="Quantum computing leverages quantum mechanics principles.",
        )

        assert "timestamp" in result
        assert "metrics_scores" in result
        assert "average_alignment_score" in result
        assert "status" in result
        assert len(result["metrics_scores"]) == 6
        assert 0 <= result["average_alignment_score"] <= 5

    def test_batch_evaluation(self, framework):
        """Test batch evaluation."""
        evaluations = [
            {
                "prompt": "What is AI?",
                "response": "AI is artificial intelligence.",
                "ground_truth": "AI stands for artificial intelligence.",
            },
            {
                "prompt": "Explain ML.",
                "response": "Machine learning is a subset of AI.",
                "ground_truth": "ML is machine learning, part of AI.",
            },
        ]

        results = framework.batch_evaluate(evaluations, show_progress=False)

        assert len(results) == 2
        for result in results:
            assert "metrics_scores" in result
            assert result["average_alignment_score"] >= 0

    def test_status_determination(self, framework):
        """Test status determination based on score."""
        # High score
        result_high = framework.evaluate_response(
            prompt="1+1=",
            response="The answer to 1+1 is 2. This is a basic arithmetic fact.",
            ground_truth="1+1 equals 2.",
        )
        assert result_high["status"] in ["PASSED (Gold Standard)", "NEEDS REVIEW"]

        # Low score
        result_low = framework.evaluate_response(
            prompt="What is Python?",
            response="xyz",
            ground_truth="A programming language.",
        )
        assert result_low["average_alignment_score"] < 4.0

    def test_all_metrics_scored(self, framework):
        """Test that all metrics are scored."""
        result = framework.evaluate_response(
            prompt="Test prompt",
            response="Test response with content and reasoning because it matters.",
            ground_truth="Test answer",
        )

        expected_metrics = {
            "factual_accuracy",
            "hallucination_severity",
            "reasoning_depth",
            "instruction_compliance",
            "coherence",
            "safety",
        }

        assert set(result["metrics_scores"].keys()) == expected_metrics
