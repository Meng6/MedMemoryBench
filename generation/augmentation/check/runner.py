"""Check module runner.

Orchestrates the check, analysis, report generation, and optional regeneration.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import CheckRunnerConfig, CheckerConfig, EnhancerConfig
from .checker import QueryDifficultyChecker, PersonaCheckResult
from .enhancer import DifficultyEnhancer, EnhancementSuggestion
from .regenerator import QueryRegenerator, RegenerationResult
from .report import ReportGenerator, CheckReport

logger = logging.getLogger(__name__)


@dataclass
class CheckRunResult:
    """Overall check run result."""

    start_time: str
    end_time: str
    duration_seconds: float

    total_personas: int
    total_queries: int
    total_correct: int
    overall_correct_rate: float

    regeneration_attempted: int = 0
    regeneration_success: int = 0
    regeneration_failed: int = 0

    report_path: Optional[str] = None
    json_report_path: Optional[str] = None
    enhanced_queries_path: Optional[str] = None


class CheckRunner:
    """Main runner for the check module.

    Workflow: discover personas -> check queries -> generate suggestions -> report -> optional regeneration.
    """

    def __init__(self, config: Optional[CheckRunnerConfig] = None):
        self.config = config or CheckRunnerConfig()

        self.checker = QueryDifficultyChecker(
            config=self.config.checker,
            dataset_dir=self.config.dataset_dir,
        )
        self.enhancer = DifficultyEnhancer(self.config.enhancer)
        self.regenerator = QueryRegenerator()
        self.report_generator = ReportGenerator(self.config.output_dir)

        logger.info(f"[CheckRunner] Initialized")
        logger.info(f"  Dataset dir: {self.config.dataset_dir}")
        logger.info(f"  Output dir: {self.config.output_dir}")
        logger.info(f"  Regeneration enabled: {self.config.enable_regenerate}")

    def discover_personas(self) -> List[int]:
        """Discover all personas with eval queries."""
        dataset_dir = Path(self.config.dataset_dir)
        persona_ids = []

        for persona_dir in sorted(dataset_dir.iterdir()):
            if not persona_dir.is_dir():
                continue
            if not persona_dir.name.startswith("persona_"):
                continue

            queries_path = persona_dir / "eval" / "generated_queries.json"
            if queries_path.exists():
                try:
                    persona_id = int(persona_dir.name.split("_")[1])
                    persona_ids.append(persona_id)
                except ValueError:
                    continue

        if self.config.persona_ids:
            persona_ids = [pid for pid in persona_ids if pid in self.config.persona_ids]

        logger.info(f"[CheckRunner] Found {len(persona_ids)} personas")
        return persona_ids

    def load_queries(self, persona_id: int) -> List[Dict[str, Any]]:
        """Load queries for a given persona."""
        queries_path = Path(self.config.dataset_dir) / f"persona_{persona_id}" / "eval" / "generated_queries.json"

        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("queries", [])

    def run(self) -> CheckRunResult:
        """Run the full check workflow."""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("[CheckRunner] Starting query difficulty check")
        logger.info("=" * 60)

        persona_ids = self.discover_personas()

        if not persona_ids:
            logger.error("No persona data found")
            return CheckRunResult(
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0.0,
                total_personas=0,
                total_queries=0,
                total_correct=0,
                overall_correct_rate=0.0,
            )

        # Phase 1: Check difficulty
        logger.info("\n" + "=" * 60)
        logger.info("[Phase 1] Query difficulty check")
        logger.info("=" * 60)

        all_persona_results: List[PersonaCheckResult] = []
        all_queries_map: Dict[str, Dict[str, Any]] = {}

        for persona_id in persona_ids:
            queries = self.load_queries(persona_id)

            for q in queries:
                all_queries_map[q["query_id"]] = q

            persona_result = self.checker.check_persona(persona_id, queries)
            all_persona_results.append(persona_result)

        total_queries = sum(r.total_queries for r in all_persona_results)
        total_correct = sum(r.correct_count for r in all_persona_results)
        overall_correct_rate = total_correct / total_queries if total_queries > 0 else 0.0

        logger.info(f"\n[Phase 1 done]")
        logger.info(f"  Queries: {total_queries}, Correct: {total_correct}, Rate: {overall_correct_rate * 100:.1f}%")

        # Phase 2: Generate enhancement suggestions
        logger.info("\n" + "=" * 60)
        logger.info("[Phase 2] Generate enhancement suggestions")
        logger.info("=" * 60)

        correct_results = []
        for persona_result in all_persona_results:
            for query_result in persona_result.query_results:
                if query_result.is_correct:
                    correct_results.append(query_result)

        enhancement_suggestions = []
        if correct_results:
            logger.info(f"  Generating suggestions for {len(correct_results)} questions...")
            enhancement_suggestions = self.enhancer.batch_analyze(
                correct_results, all_queries_map
            )
            logger.info(f"  Generated {len(enhancement_suggestions)} suggestions")

        # Phase 3: Generate report
        logger.info("\n" + "=" * 60)
        logger.info("[Phase 3] Generate report")
        logger.info("=" * 60)

        report = self.report_generator.generate_report(
            all_persona_results, enhancement_suggestions
        )

        report_path = None
        json_report_path = None

        if self.config.generate_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.config.report_format in ("markdown", "both"):
                report_path = self.report_generator.save_markdown_report(
                    report, f"check_report_{timestamp}.md"
                )

            if self.config.report_format in ("json", "both"):
                json_report_path = self.report_generator.save_json_report(
                    report, f"check_report_{timestamp}.json"
                )

        # Phase 4: Regeneration (if enabled)
        regeneration_attempted = 0
        regeneration_success = 0
        regeneration_failed = 0
        enhanced_queries_path = None

        if self.config.enable_regenerate and correct_results and not self.config.dry_run:
            logger.info("\n" + "=" * 60)
            logger.info("[Phase 4] Regenerate questions")
            logger.info("=" * 60)

            suggestions_map = {s.query_id: s for s in enhancement_suggestions}

            queries_to_regenerate = [
                all_queries_map[r.query_id] for r in correct_results
                if r.query_id in all_queries_map
            ]

            regeneration_results = self.regenerator.batch_regenerate(
                queries_to_regenerate, suggestions_map
            )

            regeneration_attempted = len(regeneration_results)
            regeneration_success = sum(1 for r in regeneration_results if r.success)
            regeneration_failed = regeneration_attempted - regeneration_success

            logger.info(f"  Attempted: {regeneration_attempted}")
            logger.info(f"  Success: {regeneration_success}")
            logger.info(f"  Failed: {regeneration_failed}")

            if regeneration_success > 0:
                enhanced_queries_path = self._save_enhanced_queries(
                    regeneration_results, all_queries_map
                )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info("[Check complete]")
        logger.info("=" * 60)
        logger.info(f"  Duration: {duration:.2f}s")
        if report_path:
            logger.info(f"  Markdown report: {report_path}")
        if json_report_path:
            logger.info(f"  JSON report: {json_report_path}")
        if enhanced_queries_path:
            logger.info(f"  Enhanced queries: {enhanced_queries_path}")

        return CheckRunResult(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_personas=len(persona_ids),
            total_queries=total_queries,
            total_correct=total_correct,
            overall_correct_rate=overall_correct_rate,
            regeneration_attempted=regeneration_attempted,
            regeneration_success=regeneration_success,
            regeneration_failed=regeneration_failed,
            report_path=report_path,
            json_report_path=json_report_path,
            enhanced_queries_path=enhanced_queries_path,
        )

    def _save_enhanced_queries(
        self,
        regeneration_results: List[RegenerationResult],
        all_queries_map: Dict[str, Dict[str, Any]],
    ) -> str:
        """Save enhanced queries to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"enhanced_queries_{timestamp}.json"

        enhanced_queries = []
        for result in regeneration_results:
            if result.success and result.new_query:
                enhanced_queries.append({
                    "query_id": result.query_id,
                    "original_query": result.original_query,
                    "enhanced_query": result.new_query,
                })

        output_data = {
            "generated_at": datetime.now().isoformat(),
            "total_enhanced": len(enhanced_queries),
            "queries": enhanced_queries,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[CheckRunner] Enhanced queries saved: {output_path}")
        return str(output_path)
