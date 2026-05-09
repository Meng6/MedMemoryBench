"""Token usage tracking module.

Provides centralized token usage statistics across pipeline stages.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StageTokenUsage:
    """Token usage for a specific pipeline stage."""

    stage_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def add(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> None:
        """Add token usage from a single request."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.request_count += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


class TokenTracker:
    """Global token usage tracker.

    Thread-safe singleton for tracking token usage across all LLM calls.
    Supports multiple pipeline stages and provides summary reports.
    """

    _instance: Optional["TokenTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TokenTracker":
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize tracker."""
        if self._initialized:
            return
        self._initialized = True
        self._stages: dict[str, StageTokenUsage] = {}
        self._current_stage: str = "unknown"
        self._stage_lock = threading.Lock()
        self._start_time: Optional[datetime] = None
        self._model: str = "unknown"

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._stage_lock:
            self._stages.clear()
            self._current_stage = "unknown"
            self._start_time = datetime.now()
            self._model = "unknown"
            logger.info("[TokenTracker] Reset all tracking data")

    def set_model(self, model: str) -> None:
        """Set the current model being used."""
        self._model = model

    def set_stage(self, stage_name: str) -> None:
        """Set the current pipeline stage.

        Args:
            stage_name: Name of the current stage (e.g., "persona", "event", "dialogue")
        """
        with self._stage_lock:
            self._current_stage = stage_name
            if stage_name not in self._stages:
                self._stages[stage_name] = StageTokenUsage(stage_name=stage_name)
            logger.debug(f"[TokenTracker] Stage set to: {stage_name}")

    def track(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        caller: str = "unknown",
        stage: Optional[str] = None,
    ) -> None:
        """Track token usage from an LLM call.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (may differ from sum due to special tokens)
            caller: Identifier of the calling function
            stage: Optional stage override (uses current stage if not provided)
        """
        target_stage = stage or self._current_stage

        with self._stage_lock:
            if target_stage not in self._stages:
                self._stages[target_stage] = StageTokenUsage(stage_name=target_stage)

            self._stages[target_stage].add(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        logger.debug(
            f"[TokenTracker] Tracked: stage={target_stage} caller={caller} "
            f"prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}"
        )

    def get_stage_usage(self, stage_name: str) -> Optional[StageTokenUsage]:
        """Get token usage for a specific stage."""
        with self._stage_lock:
            return self._stages.get(stage_name)

    def get_all_stages(self) -> dict[str, StageTokenUsage]:
        """Get token usage for all stages."""
        with self._stage_lock:
            return dict(self._stages)

    def get_total_usage(self) -> StageTokenUsage:
        """Get total token usage across all stages."""
        total = StageTokenUsage(stage_name="total")
        with self._stage_lock:
            for stage in self._stages.values():
                total.prompt_tokens += stage.prompt_tokens
                total.completion_tokens += stage.completion_tokens
                total.total_tokens += stage.total_tokens
                total.request_count += stage.request_count
        return total

    def get_summary(self) -> dict:
        """Get a complete summary of token usage.

        Returns:
            Dictionary with all stages and total usage
        """
        stages_data = {}
        with self._stage_lock:
            for name, stage in self._stages.items():
                stages_data[name] = stage.to_dict()

        total = self.get_total_usage()

        return {
            "model": self._model,
            "start_time": (
                self._start_time.isoformat() if self._start_time else None
            ),
            "end_time": datetime.now().isoformat(),
            "stages": stages_data,
            "total": total.to_dict(),
        }

    def print_summary(self) -> None:
        """Print a formatted summary to logger."""
        summary = self.get_summary()

        logger.info("=" * 60)
        logger.info("Token Usage Summary")
        logger.info("=" * 60)
        logger.info(f"Model: {summary['model']}")

        if summary["stages"]:
            logger.info("-" * 60)
            logger.info(
                f"{'Stage':<20} {'Prompt':>12} {'Completion':>12} {'Total':>12} {'Requests':>10}"
            )
            logger.info("-" * 60)

            for stage_name, stage_data in summary["stages"].items():
                logger.info(
                    f"{stage_name:<20} {stage_data['prompt_tokens']:>12,} "
                    f"{stage_data['completion_tokens']:>12,} {stage_data['total_tokens']:>12,} "
                    f"{stage_data['request_count']:>10}"
                )

            logger.info("-" * 60)
            total = summary["total"]
            logger.info(
                f"{'TOTAL':<20} {total['prompt_tokens']:>12,} "
                f"{total['completion_tokens']:>12,} {total['total_tokens']:>12,} "
                f"{total['request_count']:>10}"
            )

        logger.info("=" * 60)

    def print_console_summary(self) -> None:
        """Print a formatted summary to console (stdout)."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("Token Usage Summary")
        print("=" * 70)
        print(f"Model: {summary['model']}")

        if summary["stages"]:
            print("-" * 70)
            print(
                f"{'Stage':<25} {'Prompt':>12} {'Completion':>12} {'Total':>12} {'Requests':>8}"
            )
            print("-" * 70)

            for stage_name, stage_data in summary["stages"].items():
                print(
                    f"{stage_name:<25} {stage_data['prompt_tokens']:>12,} "
                    f"{stage_data['completion_tokens']:>12,} {stage_data['total_tokens']:>12,} "
                    f"{stage_data['request_count']:>8}"
                )

            print("-" * 70)
            total = summary["total"]
            print(
                f"{'TOTAL':<25} {total['prompt_tokens']:>12,} "
                f"{total['completion_tokens']:>12,} {total['total_tokens']:>12,} "
                f"{total['request_count']:>8}"
            )

        print("=" * 70 + "\n")

    def save_to_file(self, filepath: Path) -> None:
        """Save token usage summary to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        summary = self.get_summary()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[TokenTracker] Saved summary to: {filepath}")


# Global tracker instance
_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance.

    Returns:
        The singleton TokenTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker
