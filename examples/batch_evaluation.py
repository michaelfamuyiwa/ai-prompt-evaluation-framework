"""Batch processing example with output generation."""

import json
from src.evaluator import BatchProcessor


def main():
    """Run batch evaluation example."""
    # Initialize batch processor
    processor = BatchProcessor(batch_size=50)

    # Sample batch of evaluations
    evaluations = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a field of AI where systems learn from data.",
            "ground_truth": "ML is an AI discipline enabling systems to learn from data.",
        },
        {
            "prompt": "Explain neural networks without mentioning deep learning.",
            "response": "Neural networks are inspired by biological neurons and process information.",
            "ground_truth": "Neural networks mimic biological systems for information processing.",
        },
        {
            "prompt": "Why is version control important?",
            "response": "Version control is crucial because it tracks code changes, enables collaboration, and allows reverting to previous states. For example, Git helps teams work together. Therefore, it is essential.",
            "ground_truth": "VC tracks changes, enables collaboration, and provides history recovery.",
        },
    ]

    print(f"[INFO] Processing batch of {len(evaluations)} evaluations...")
    results = processor.process_batch(evaluations, output_dir="./results")

    print(f"\n[INFO] Batch processing complete!")
    print(f"[INFO] Results saved to ./results/")
    print(f"[INFO] Generated files:")
    print(f"  - evaluation_results.json")
    print(f"  - evaluation_report.md")
    print(f"  - summary.json")

    # Print quick summary
    print("\n--- BATCH SUMMARY ---")
    passed = sum(1 for r in results if r["status"] == "PASSED (Gold Standard)")
    rejected = sum(1 for r in results if r["status"] == "REJECTED")
    avg_score = sum(r["average_alignment_score"] for r in results) / len(results)

    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Rejected: {rejected}")
    print(f"Average Score: {avg_score:.2f}/5.0")


if __name__ == "__main__":
    main()
