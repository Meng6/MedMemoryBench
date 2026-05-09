"""Check report generator."""

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .checker import PersonaCheckResult, QueryCheckResult
from .enhancer import EnhancementSuggestion

logger = logging.getLogger(__name__)


@dataclass
class CheckReport:
    """Check report data."""

    generated_at: str
    total_personas: int
    total_queries: int
    total_correct: int
    overall_correct_rate: float
    stats_by_type: Dict[str, Dict[str, Any]]
    persona_results: List[PersonaCheckResult]
    enhancement_suggestions: List[EnhancementSuggestion]


class ReportGenerator:
    """Generates check reports in markdown and JSON formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        persona_results: List[PersonaCheckResult],
        enhancement_suggestions: Optional[List[EnhancementSuggestion]] = None,
    ) -> CheckReport:
        """Generate a check report from persona results."""
        total_queries = sum(r.total_queries for r in persona_results)
        total_correct = sum(r.correct_count for r in persona_results)
        overall_correct_rate = total_correct / total_queries if total_queries > 0 else 0.0

        stats_by_type: Dict[str, Dict[str, Any]] = {}
        for persona_result in persona_results:
            for query_result in persona_result.query_results:
                qt = query_result.query_type
                if qt not in stats_by_type:
                    stats_by_type[qt] = {"total": 0, "correct": 0}
                stats_by_type[qt]["total"] += 1
                if query_result.is_correct:
                    stats_by_type[qt]["correct"] += 1

        for qt, stats in stats_by_type.items():
            stats["correct_rate"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        return CheckReport(
            generated_at=datetime.now().isoformat(),
            total_personas=len(persona_results),
            total_queries=total_queries,
            total_correct=total_correct,
            overall_correct_rate=overall_correct_rate,
            stats_by_type=stats_by_type,
            persona_results=persona_results,
            enhancement_suggestions=enhancement_suggestions or [],
        )

    def save_markdown_report(
        self,
        report: CheckReport,
        filename: str = "check_report.md",
    ) -> str:
        """Save report in markdown format."""
        output_path = self.output_dir / filename

        lines = []
        lines.append("# Query Difficulty Check Report")
        lines.append("")
        lines.append(f"**Generated**: {report.generated_at}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Personas checked**: {report.total_personas}")
        lines.append(f"- **Queries checked**: {report.total_queries}")
        lines.append(f"- **Model correct**: {report.total_correct}")
        lines.append(f"- **Correct rate**: {report.overall_correct_rate * 100:.1f}%")
        lines.append("")
        lines.append("> Higher correct rate means questions are too easy (not memory-dependent).")
        lines.append("")

        lines.append("## Stats by Query Type")
        lines.append("")
        lines.append("| Query Type | Total | Correct | Rate |")
        lines.append("|-----------|-------|---------|------|")
        for qt, stats in sorted(report.stats_by_type.items()):
            lines.append(
                f"| {qt} | {stats['total']} | {stats['correct']} | {stats['correct_rate'] * 100:.1f}% |"
            )
        lines.append("")

        lines.append("## Correctly Answered Questions (Need Enhancement)")
        lines.append("")

        for persona_result in report.persona_results:
            if persona_result.correct_count == 0:
                continue

            lines.append(f"### Persona {persona_result.persona_id}")
            lines.append("")
            lines.append(f"- Total queries: {persona_result.total_queries}")
            lines.append(f"- Correct: {persona_result.correct_count}")
            lines.append(f"- Rate: {persona_result.correct_rate * 100:.1f}%")
            lines.append("")

            for query_result in persona_result.query_results:
                if not query_result.is_correct:
                    continue

                lines.append(f"#### {query_result.query_id}")
                lines.append("")
                lines.append(f"- **Type**: {query_result.query_type}")
                lines.append(f"- **Question**: {query_result.question}")
                lines.append(f"- **Expected**: {query_result.expected_answer}")
                lines.append(f"- **Model output**: {query_result.model_output[:300]}...")
                lines.append("")

        if report.enhancement_suggestions:
            lines.append("## Enhancement Suggestions")
            lines.append("")

            for suggestion in report.enhancement_suggestions:
                lines.append(f"### {suggestion.query_id}")
                lines.append("")
                lines.append(f"**Original question**: {suggestion.original_question}")
                lines.append("")
                lines.append(f"**Analysis**: {suggestion.analysis}")
                lines.append("")
                lines.append("**Suggestions**:")
                for i, s in enumerate(suggestion.suggestions, 1):
                    lines.append(f"{i}. {s}")
                lines.append("")
                if suggestion.enhanced_question:
                    lines.append(f"**Enhanced question example**: {suggestion.enhanced_question}")
                    lines.append("")

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"[ReportGenerator] Markdown report saved: {output_path}")
        return str(output_path)

    def save_json_report(
        self,
        report: CheckReport,
        filename: str = "check_report.json",
    ) -> str:
        """Save report in JSON format."""
        output_path = self.output_dir / filename

        data = {
            "generated_at": report.generated_at,
            "summary": {
                "total_personas": report.total_personas,
                "total_queries": report.total_queries,
                "total_correct": report.total_correct,
                "overall_correct_rate": report.overall_correct_rate,
            },
            "stats_by_type": report.stats_by_type,
            "correct_queries": [],
            "enhancement_suggestions": [],
        }

        for persona_result in report.persona_results:
            for query_result in persona_result.query_results:
                if query_result.is_correct:
                    data["correct_queries"].append({
                        "persona_id": persona_result.persona_id,
                        "query_id": query_result.query_id,
                        "query_type": query_result.query_type,
                        "question": query_result.question,
                        "expected_answer": query_result.expected_answer,
                        "model_output": query_result.model_output,
                        "score": query_result.score,
                    })

        for suggestion in report.enhancement_suggestions:
            data["enhancement_suggestions"].append({
                "query_id": suggestion.query_id,
                "query_type": suggestion.query_type,
                "original_question": suggestion.original_question,
                "expected_answer": suggestion.expected_answer,
                "model_output": suggestion.model_output,
                "analysis": suggestion.analysis,
                "suggestions": suggestion.suggestions,
                "enhanced_question": suggestion.enhanced_question,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[ReportGenerator] JSON report saved: {output_path}")
        return str(output_path)
