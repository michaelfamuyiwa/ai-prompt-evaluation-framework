# Architecture Overview

## System Design

The AI Prompt Evaluation Framework is built with a modular, extensible architecture:

```
┌─────────────────────────────────────────────────────┐
│         AIPromptEvaluationFramework                 │
│         (Main Orchestrator)                         │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐           ┌─────▼─────┐
    │Metrics[]│           │Batch      │
    │         │           │Processor  │
    └────┬────┘           └─────┬─────┘
         │                       │
    ┌────┴────────────┐         │
    │                 │         │
┌───▼────┐ ┌────▼────┐ ...  ┌──▼───┐
│Factual │ │Halluc.  │      │Output│
│Accuracy│ │Severity │      │Genr. │
└────────┘ └─────────┘      └──────┘
```

## Components

### 1. **AIPromptEvaluationFramework** (framework.py)
- **Role**: Central orchestrator
- **Responsibilities**:
  - Initialize metrics
  - Coordinate evaluation pipeline
  - Aggregate metric scores
  - Determine overall status
  - Handle single and batch evaluations

### 2. **BaseMetric** (base_evaluator.py)
- **Role**: Abstract base class for all metrics
- **Contract**: All metrics implement `evaluate()`
- **Features**:
  - Score validation (0-5 range)
  - Weight and threshold configuration
  - Consistent return format

### 3. **Metric Classes** (metrics/)
Each metric is independent and implements BaseMetric:

| Metric | Method | Key Feature |
|--------|--------|-------------|
| FactualAccuracy | Cosine similarity | Semantic matching to ground truth |
| Hallucination | Semantic drift | Topic coherence + accuracy |
| ReasoningDepth | Linguistic analysis | Connector detection |
| Compliance | Pattern matching | Instruction adherence |
| Coherence | Sentence similarity | Internal consistency |
| Safety | Pattern matching | Harmful content detection |

### 4. **BatchProcessor** (batch_processor.py)
- **Role**: Handle large-scale evaluations
- **Responsibilities**:
  - Process multiple items
  - Generate JSON/Markdown output
  - Create summary reports
  - Save to disk

### 5. **EmbeddingCache** (metrics/embeddings.py)
- **Role**: Shared embedding model
- **Feature**: Global cache to avoid reloading
- **Model**: all-MiniLM-L6-v2 (384 dims)

## Data Flow

### Single Evaluation
```
Input (prompt, response, ground_truth)
    ↓
Framework.evaluate_response()
    ↓
For each Metric:
  Metric.evaluate() → {score, explanation, details}
    ↓
Aggregate Scores:
  weighted_avg = Σ(metric.score × metric.weight)
    ↓
Determine Status:
  score >= 4.0 → "PASSED"
  score < 2.5  → "REJECTED"
  else         → "NEEDS REVIEW"
    ↓
Output: Evaluation Report
```

### Batch Evaluation
```
Input: List of {prompt, response, ground_truth}
    ↓
BatchProcessor.process_batch()
    ↓
For each item:
  Framework.evaluate_response() → report
    ↓
Collect all reports
    ↓
Generate outputs:
  - JSON: Raw results
  - Markdown: Human-readable report
  - Summary: Aggregated statistics
    ↓
Save to output_dir
```

## Metric Scoring System

### Score Scale
- **5.0**: Excellent
- **4.0**: Good
- **3.0**: Acceptable
- **2.0**: Poor
- **1.0**: Very Poor
- **0.0**: Critical Failure

### Weighted Average
```
Final Score = Σ(metric_score × metric_weight) / Σ(weights)
```

Default weights:
- Factual Accuracy: 0.25
- Hallucination: 0.20
- Compliance: 0.20
- Reasoning: 0.15
- Coherence: 0.10
- Safety: 0.10

## Configuration

### YAML-Based Configuration (config/metrics_config.yaml)

```yaml
metrics:
  factual_accuracy:
    weight: 0.25
    threshold: 3.5
    method: "semantic_similarity"
```

### Runtime Configuration

```python
config = {
    "metrics": {
        "factual_accuracy": {"weight": 0.3},
        "safety": {"weight": 0.05}
    }
}
evaluator = AIPromptEvaluationFramework(config=config)
```

## Testing Architecture

### Test Pyramid
```
        Integration Tests
        (test_framework.py)
       /              \
      /                \
     /  Unit Tests      \
    /  (test_metrics.py) \
```

### Test Coverage
- **Unit Tests**: Individual metric correctness
- **Integration Tests**: Framework behavior with all metrics
- **Fixtures**: Reusable test data and framework instances

## Extensibility

### Adding New Metrics

1. Create new class inheriting from `BaseMetric`
2. Implement `evaluate()` method
3. Register in framework `_initialize_metrics()`
4. Add unit tests

### Adding New Output Formats

1. Extend `ReportGenerator` class
2. Implement format-specific generation
3. Register in `BatchProcessor._save_results()`

## Performance Considerations

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Size**: ~33MB (lightweight)
- **Latency**: ~100ms per encode
- **Caching**: Global instance cached per session

### Batch Processing
- **Batch Size**: 50 (configurable)
- **Parallel Workers**: 4 (planned)
- **Memory**: ~500MB for typical batch

### Optimization Tips
1. Reuse evaluator instance
2. Batch-process instead of individual evals
3. Cache embeddings for repeated text
4. Consider GPU acceleration for large batches

## Error Handling

### Metric-Level
- Individual metric failure doesn't stop evaluation
- Failed metrics default to score 2.0
- Errors logged and reported

### Framework-Level
- Graceful degradation
- Comprehensive logging
- Detailed error messages in reports

## Logging

Structured logging throughout:

```python
logger = logging.getLogger(__name__)
logger.info("Evaluation started", extra={"response_length": 500})
logger.error("Metric failed", exc_info=True)
```

Configure logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
