"""Unit tests for evaluation metrics."""

import pytest
from src.evaluator.metrics import (
    FactualAccuracyMetric,
    HallucinationMetric,
    ReasoningDepthMetric,
    InstructionComplianceMetric,
    CoherenceMetric,
    SafetyMetric,
)


class TestFactualAccuracyMetric:
    """Tests for factual accuracy metric."""

    def test_high_similarity_response(self):
        """Test response with high semantic similarity to ground truth."""
        metric = FactualAccuracyMetric()
        result = metric.evaluate(
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            ground_truth="Paris is the capital of France.",
        )
        assert result["score"] >= 4.0, f"Expected high score, got {result['score']}"
        assert "similarity" in result["explanation"].lower()

    def test_no_ground_truth(self):
        """Test behavior when ground truth is not provided."""
        metric = FactualAccuracyMetric()
        result = metric.evaluate(
            prompt="Test", response="Test response", ground_truth=None
        )
        assert result["score"] == 3.0
        assert "No ground truth" in result["explanation"]


class TestHallucinationMetric:
    """Tests for hallucination metric."""

    def test_accurate_response(self):
        """Test response with minimal hallucination."""
        metric = HallucinationMetric()
        result = metric.evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
            ground_truth="Python is a high-level programming language.",
        )
        assert result["score"] >= 3.0

    def test_consistency_check_without_ground_truth(self):
        """Test internal consistency when ground truth is missing."""
        metric = HallucinationMetric()
        result = metric.evaluate(
            prompt="Test", response="This is consistent. However it is good.", ground_truth=None
        )
        assert "consistency" in result["method"].lower()


class TestReasoningDepthMetric:
    """Tests for reasoning depth metric."""

    def test_high_reasoning_depth(self):
        """Test response with logical connectors and depth."""
        metric = ReasoningDepthMetric()
        result = metric.evaluate(
            prompt="Why is water important?",
            response="Water is important because it is essential for life. Therefore, organisms depend on it. For example, plants use water for photosynthesis.",
            ground_truth=None,
        )
        assert result["score"] >= 3.0
        assert result["details"]["reasoning_indicators"] > 0

    def test_low_reasoning_depth(self):
        """Test response with minimal reasoning."""
        metric = ReasoningDepthMetric()
        result = metric.evaluate(
            prompt="Why?", response="Yes.", ground_truth=None
        )
        assert result["score"] < 3.0


class TestInstructionComplianceMetric:
    """Tests for instruction compliance metric."""

    def test_negation_violation(self):
        """Test detection of negation instruction violations."""
        metric = InstructionComplianceMetric()
        result = metric.evaluate(
            prompt="Explain the concept without mentioning Java.",
            response="This is like Java but more efficient.",
            ground_truth=None,
        )
        assert result["score"] < 4.0
        assert len(result["details"]["violations"]) > 0

    def test_compliant_response(self):
        """Test fully compliant response."""
        metric = InstructionComplianceMetric()
        result = metric.evaluate(
            prompt="Explain Python.",
            response="Python is a programming language known for readability.",
            ground_truth=None,
        )
        assert result["score"] >= 3.5


class TestCoherenceMetric:
    """Tests for coherence metric."""

    def test_coherent_response(self):
        """Test coherent multi-sentence response."""
        metric = CoherenceMetric()
        result = metric.evaluate(
            prompt="Test",
            response="Sentence one about topics. Sentence two continues the discussion. Sentence three expands further.",
            ground_truth=None,
        )
        assert "coherence" in result["method"].lower()
        assert result["details"]["sentence_count"] == 3

    def test_single_sentence(self):
        """Test single sentence response."""
        metric = CoherenceMetric()
        result = metric.evaluate(
            prompt="Test", response="Single sentence response.", ground_truth=None
        )
        assert result["score"] == 4.0


class TestSafetyMetric:
    """Tests for safety metric."""

    def test_safe_response(self):
        """Test safe response."""
        metric = SafetyMetric()
        result = metric.evaluate(
            prompt="Test", response="This is a safe and helpful response.", ground_truth=None
        )
        assert result["score"] == 5.0
        assert len(result["details"]["issues_found"]) == 0

    def test_unsafe_response(self):
        """Test detection of unsafe content."""
        metric = SafetyMetric()
        result = metric.evaluate(
            prompt="Test",
            response="This response contains violence and illegal activity.",
            ground_truth=None,
        )
        assert result["score"] < 5.0
        assert len(result["details"]["issues_found"]) > 0
