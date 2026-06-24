"""Pytest configuration and shared fixtures."""

import pytest
from src.evaluator import AIPromptEvaluationFramework
from src.evaluator.metrics.embeddings import reset_model


@pytest.fixture
def framework():
    """Provide an evaluation framework instance."""
    return AIPromptEvaluationFramework()


@pytest.fixture(autouse=True)
def cleanup_embeddings():
    """Clean up embedding model cache between tests."""
    yield
    reset_model()
