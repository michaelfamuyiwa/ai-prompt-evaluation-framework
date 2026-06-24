"""Basic usage example of the evaluation framework."""

import json
from src.evaluator import AIPromptEvaluationFramework


def main():
    """Run a basic evaluation example."""
    # Initialize the framework
    evaluator = AIPromptEvaluationFramework()

    # Define an evaluation
    prompt = "Explain why compiled languages execute faster than interpreted ones."
    response = """Compiled languages translate source code directly into machine language 
    instructions before execution, which makes execution incredibly swift. The compilation 
    process happens once, and then the machine code runs directly on the CPU. 
    Interpreted languages, by contrast, parse and execute code line-by-line at runtime, 
    which adds overhead. For example, a compiled C program runs much faster than an 
    equivalent Python script."""

    ground_truth = """Compiled languages are translated to machine code ahead of time, 
    allowing direct CPU execution. Interpreted languages execute code line-by-line at runtime, 
    adding interpretation overhead. This makes compiled languages generally faster."""

    # Run evaluation
    print("[INFO] Starting evaluation...")
    report = evaluator.evaluate_response(prompt, response, ground_truth)

    # Display results
    print("\n--- EVALUATION REPORT ---")
    print(json.dumps(report, indent=2))

    # Print summary
    print("\n--- SUMMARY ---")
    print(f"Status: {report['status']}")
    print(f"Overall Score: {report['average_alignment_score']}/5.0")
    print(f"Evaluation Time: {report['evaluation_time_seconds']}s")

    print("\n--- METRIC SCORES ---")
    for metric, score in report["metrics_scores"].items():
        print(f"  {metric}: {score}/5.0")


if __name__ == "__main__":
    main()
