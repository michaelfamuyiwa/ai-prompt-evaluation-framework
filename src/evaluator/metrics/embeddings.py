"""Embedding utilities for semantic similarity calculations."""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global model cache
_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings(text: str):
    """
    Get embeddings for text using cached model.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL.encode(text, convert_to_tensor=True)


def reset_model():
    """Clear the cached model (useful for testing)."""
    global _MODEL
    _MODEL = None
