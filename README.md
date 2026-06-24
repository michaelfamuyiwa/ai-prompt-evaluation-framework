# AI Prompt Evaluation Framework 🤖

A modular, production-ready Python framework for scoring LLM responses across 6 core evaluation dimensions: **factual accuracy**, **hallucination severity**, **reasoning depth**, **instruction compliance**, **coherence**, and **safety**.

**Built for RLHF training pipelines and professional-grade response evaluation.**

---

## ✨ Key Features

- **🎯 6 Independent Metrics**: Each metric is isolated, testable, and configurable
  - Factual Accuracy (semantic similarity)
  - Hallucination Detection (semantic drift)
  - Reasoning Depth (linguistic analysis)
  - Instruction Compliance (pattern matching)
  - Coherence (sentence-level similarity)
  - Safety (content filtering)

- **⚙️ Configurable Weights & Thresholds**: Customize metric importance and decision boundaries

- **📊 Batch Processing**: Evaluate 50+ responses efficiently with JSON and Markdown output

- **📈 Comprehensive Reporting**: Detailed metrics, scores, and aggregated statistics

- **✅ Production Ready**: Comprehensive tests, logging, error handling

- **🧪 Fully Tested**: Unit and integration tests for all metrics

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/michaelfamuyiwa/ai-prompt-evaluation-framework.git
cd ai-prompt-evaluation-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.evaluator import AIPromptEvaluationFramework

# Initialize
evaluator = AIPromptEvaluationFramework()

# Evaluate a response
report = evaluator.evaluate_response(
    prompt="Explain quantum computing",
    response="Quantum computing uses quantum bits...",
    ground_truth="Quantum computing leverages quantum mechanics..."
)

print(f"Score: {report['average_alignment_score']}/5.0")
print(f"Status: {report['status']}")
print(f"Metrics: {report['metrics_scores']}")
```

### Batch Processing

```python
from src.evaluator import BatchProcessor

processor = BatchProcessor(batch_size=50)

evaluations = [
    {
        "prompt": "What is AI?",
        "response": "AI is artificial intelligence...",
        "ground_truth": "AI stands for artificial intelligence..."
    },
    # ... more evaluations
]

results = processor.process_batch(evaluations, output_dir="./results")
```

---

## 📁 Project Structure

```
.
├── src/
│   └── evaluator/
│       ├── __init__.py
│       ├── framework.py              # Main evaluation orchestrator
│       ├── base_evaluator.py         # Abstract base metric class
│       ├── batch_processor.py        # Batch processing and output
│       ├── report_generator.py       # Report formatting
│       └── metrics/
│           ├── __init__.py
│           ├── factual_accuracy.py   # Semantic similarity scoring
│           ├── hallucination.py      # Semantic drift detection
│           ├── reasoning.py          # Linguistic depth analysis
│           ├── compliance.py         # Instruction parsing
│           ├── coherence.py          # Sentence-level similarity
│           ├── safety.py             # Content safety checks
│           └── embeddings.py         # Shared embedding utilities
├── config/
│   └── metrics_config.yaml           # Metric weights and thresholds
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_metrics.py               # Unit tests for metrics
│   └── test_framework.py             # Integration tests
├── examples/
│   ├── basic_evaluation.py           # Simple usage example
│   └── batch_evaluation.py           # Batch processing example
├── requirements.txt
└── README.md
```

---

## 🔍 Metric Details

### Factual Accuracy (weight: 0.25)
- **Method**: Semantic similarity (cosine distance)
- **Measures**: How well the response matches ground truth
- **Score**: 0-5 (higher = more accurate)
- **Requires**: Ground truth

### Hallucination Severity (weight: 0.20)
- **Method**: Semantic drift analysis
- **Measures**: Deviation from ground truth and topic coherence
- **Score**: 0-5 (higher = less hallucination)
- **Requires**: Optional ground truth

### Reasoning Depth (weight: 0.15)
- **Method**: Linguistic marker detection
- **Measures**: Presence of logical connectors and reasoning indicators
- **Score**: 0-5 (higher = deeper reasoning)
- **Examples**: "because", "therefore", "however", "for example"

### Instruction Compliance (weight: 0.20)
- **Method**: Pattern matching and instruction parsing
- **Measures**: Adherence to explicit instructions in prompt
- **Score**: 0-5 (higher = more compliant)
- **Examples**: Format requirements, negation rules, length constraints

### Coherence (weight: 0.10)
- **Method**: Sentence-level semantic similarity
- **Measures**: Internal consistency and flow
- **Score**: 0-5 (higher = more coherent)
- **Analysis**: Measures relationship between consecutive sentences

### Safety (weight: 0.10)
- **Method**: Content pattern matching
- **Measures**: Absence of harmful content
- **Score**: 0-5 (higher = safer)
- **Flags**: Violence, illegal activity, discrimination, self-harm

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
# Test a specific metric
pytest tests/test_metrics.py::TestFactualAccuracyMetric -v

# Test framework integration
pytest tests/test_framework.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Report Output

Evaluation reports include:

```json
{
  "timestamp": "2026-06-24T12:00:00",
  "prompt_analyzed": "Explain quantum computing",
  "response_analyzed": "Quantum computing uses...",
  "metrics_scores": {
    "factual_accuracy": 4.2,
    "hallucination_severity": 4.8,
    "reasoning_depth": 3.9,
    "instruction_compliance": 5.0,
    "coherence": 4.3,
    "safety": 5.0
  },
  "average_alignment_score": 4.36,
  "status": "PASSED (Gold Standard)",
  "evaluation_time_seconds": 2.15,
  "metrics_details": {
    /* Detailed per-metric analysis */
  }
}
```

---

## ⚙️ Configuration

Customize metric weights and thresholds in `config/metrics_config.yaml`:

```yaml
metrics:
  factual_accuracy:
    weight: 0.25      # Relative importance
    threshold: 3.5    # Minimum acceptable score
    method: "semantic_similarity"
```

Load custom config:

```python
import yaml

with open('config/metrics_config.yaml') as f:
    config = yaml.safe_load(f)

evaluator = AIPromptEvaluationFramework(config=config)
```

---

## 🔧 Extending the Framework

### Add a Custom Metric

1. Create a new metric class inheriting from `BaseMetric`:

```python
from src.evaluator.base_evaluator import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self, weight=0.1, threshold=3.0):
        super().__init__("custom_metric", weight, threshold)
    
    def evaluate(self, prompt, response, ground_truth=None):
        # Your evaluation logic
        return {
            "score": 4.0,
            "explanation": "...",
            "method": "custom",
            "details": {}
        }
```

2. Add to framework initialization:

```python
self.metrics.append(CustomMetric())
```

---

## 📝 Examples

See the `examples/` directory for:

- `basic_evaluation.py` — Single response evaluation
- `batch_evaluation.py` — Batch processing with output generation

Run examples:

```bash
python examples/basic_evaluation.py
python examples/batch_evaluation.py
```

---

## 🔗 Dependencies

- **sentence-transformers** — Embedding-based semantic similarity
- **scikit-learn** — Utilities for analysis
- **pydantic** — Configuration validation
- **tqdm** — Progress bars
- **pytest** — Testing framework

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📮 Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Ready to evaluate LLM responses like a pro?** 🚀
