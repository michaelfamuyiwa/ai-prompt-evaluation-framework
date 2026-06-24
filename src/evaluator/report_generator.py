"""Report generation utilities."""

import json
from typing import Dict, Any, List
from datetime import datetime


class ReportGenerator:
    """Generate various report formats from evaluation results."""

    @staticmethod
    def to_json(results: List[Dict[str, Any]]) -> str:
        """Convert results to JSON string."""
        return json.dumps(results, indent=2, default=str)

    @staticmethod
    def to_markdown(results: List[Dict[str, Any]]) -> str:
        """Convert results to Markdown string."""
        if not results:
            return "# Empty Results\n"

        md = "# Evaluation Results Report\n\n"
        md += f"*Generated: {datetime.now().isoformat()}*\n\n"

        # Statistics
        passed = sum(1 for r in results if r["status"] == "PASSED (Gold Standard)")
        rejected = sum(1 for r in results if r["status"] == "REJECTED")
        avg_score = sum(r["average_alignment_score"] for r in results) / len(results)

        md += f"## Statistics\n\n"
        md += f"| Metric | Value |\n"
        md += f"|--------|-------|\n"
        md += f"| Total | {len(results)} |\n"
        md += f"| Passed | {passed} ({passed/len(results)*100:.1f}%) |\n"
        md += f"| Rejected | {rejected} ({rejected/len(results)*100:.1f}%) |\n"
        md += f"| Average Score | {avg_score:.2f}/5.0 |\n\n"

        return md
