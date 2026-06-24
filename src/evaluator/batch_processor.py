"""Batch processing and output generation."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from .framework import AIPromptEvaluationFramework

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing and report generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, batch_size: int = 50):
        """
        Initialize batch processor.

        Args:
            config: Framework configuration
            batch_size: Number of items to process per batch
        """
        self.framework = AIPromptEvaluationFramework(config)
        self.batch_size = batch_size
        logger.info(f"Batch processor initialized with batch_size={batch_size}")

    def process_batch(
        self, evaluations: List[Dict[str, str]], output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of evaluations.

        Args:
            evaluations: List of evaluation items
            output_dir: Optional directory to save results

        Returns:
            List of evaluation reports
        """
        logger.info(f"Processing batch of {len(evaluations)} evaluations")
        results = self.framework.batch_evaluate(evaluations, show_progress=True)

        if output_dir:
            self._save_results(results, output_dir)

        return results

    def _save_results(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Save results to JSON and Markdown files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_file = output_path / "evaluation_results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON results to {json_file}")

        # Save as Markdown
        md_file = output_path / "evaluation_report.md"
        self._generate_markdown_report(results, md_file)

        # Save summary
        summary_file = output_path / "summary.json"
        summary = self._generate_summary(results)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    def _generate_markdown_report(self, results: List[Dict[str, Any]], output_file: Path) -> None:
        """Generate a Markdown report of results."""
        with open(output_file, "w") as f:
            f.write("# AI Prompt Evaluation Report\n\n")

            # Summary statistics
            passed = sum(1 for r in results if r["status"] == "PASSED (Gold Standard)")
            rejected = sum(1 for r in results if r["status"] == "REJECTED")
            avg_score = sum(r["average_alignment_score"] for r in results) / len(results)

            f.write("## Summary\n\n")
            f.write(f"- **Total Evaluations:** {len(results)}\n")
            f.write(f"- **Passed (Gold Standard):** {passed}\n")
            f.write(f"- **Rejected:** {rejected}\n")
            f.write(f"- **Average Score:** {avg_score:.2f}/5.0\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for idx, result in enumerate(results, 1):
                f.write(f"### Evaluation {idx}: {result['status']}\n\n")
                f.write(f"**Score:** {result['average_alignment_score']}/5.0\n\n")
                f.write(f"**Prompt:** {result['prompt_analyzed'][:100]}...\n\n")
                f.write(f"**Metrics:**\n")
                for metric, score in result["metrics_scores"].items():
                    f.write(f"- {metric}: {score:.1f}/5.0\n")
                f.write("\n---\n\n")

        logger.info(f"Generated Markdown report: {output_file}")

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of evaluation results."""
        if not results:
            return {}

        metric_names = list(results[0]["metrics_scores"].keys())
        summary = {
            "total_evaluations": len(results),
            "passed_count": sum(1 for r in results if r["status"] == "PASSED (Gold Standard)"),
            "rejected_count": sum(1 for r in results if r["status"] == "REJECTED"),
            "average_score": round(
                sum(r["average_alignment_score"] for r in results) / len(results), 2
            ),
            "metric_averages": {},
            "metric_ranges": {},
        }

        for metric in metric_names:
            scores = [r["metrics_scores"][metric] for r in results]
            summary["metric_averages"][metric] = round(sum(scores) / len(scores), 2)
            summary["metric_ranges"][metric] = {"min": round(min(scores), 2), "max": round(max(scores), 2)}

        return summary
