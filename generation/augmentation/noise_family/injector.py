"""Family/friends noise data injector.

Randomly inserts generated family consultation noise sessions into the original dataset.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import FamilyNoiseConfig
from .generator import FamilyNoiseSession

logger = logging.getLogger(__name__)


class FamilyNoiseInjector:
    """Family/friends noise data injector.

    Randomly inserts family consultation noise sessions into the original dialogue dataset.
    """

    def __init__(self, config: Optional[FamilyNoiseConfig] = None):
        """Initialize the injector.

        Args:
            config: Configuration object.
        """
        self.config = config or FamilyNoiseConfig()
        logger.info("[FamilyNoiseInjector] Initialization complete")

    def _generate_weighted_positions(
        self,
        num_original: int,
        num_noise: int,
    ) -> List[int]:
        """Generate noise insertion positions with a 3:3:2:1:1 ratio.

        Allocates more noise to earlier session intervals:
        - 30% noise randomly inserted into the first 20% of sessions
        - 30% noise randomly inserted into the first 40% of sessions
        - 20% noise randomly inserted into the first 60% of sessions
        - 10% noise randomly inserted into the first 80% of sessions
        - 10% noise randomly inserted into the first 100% of sessions

        Args:
            num_original: Number of original sessions.
            num_noise: Number of noise sessions.

        Returns:
            Sorted list of insertion positions.
        """
        if num_original == 0 or num_noise == 0:
            return []

        # Ratio weights 3:3:2:1:1 (sum is 10)
        weights = [3, 3, 2, 1, 1]
        total_weight = sum(weights)

        # Compute noise count per interval
        noise_counts = []
        remaining = num_noise
        for i, w in enumerate(weights):
            if i == len(weights) - 1:
                # Last interval takes all remaining
                noise_counts.append(remaining)
            else:
                count = int(num_noise * w / total_weight)
                noise_counts.append(count)
                remaining -= count

        # Compute position range per interval
        # Interval i corresponds to original session positions [0, (i+1)*20%]
        positions = []
        for i, count in enumerate(noise_counts):
            upper_bound = int(num_original * (i + 1) * 0.2)
            # Ensure upper bound is at least 1
            upper_bound = max(1, upper_bound)
            # Randomly select `count` positions in [0, upper_bound]
            for _ in range(count):
                pos = random.randint(0, upper_bound)
                positions.append(pos)

        # Sort positions
        positions.sort()

        logger.info(f"  Noise distribution: interval counts {noise_counts}, total {len(positions)} positions")
        return positions

    def load_original_data(self) -> Dict[str, Any]:
        """Load the original dialogue data.

        Returns:
            Original data dictionary.
        """
        data_path = Path(self.config.data_dir) / self.config.input_filename
        logger.info(f"[FamilyNoiseInjector] Load original data: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"  Original session count: {len(data.get('sessions', []))}")
        return data

    def inject_noise(
        self,
        original_data: Dict[str, Any],
        noise_sessions: List[FamilyNoiseSession],
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Insert family consultation noise sessions into the original data using weighted distribution.

        Injection strategy:
        1. Preserve the original session order.
        2. Use a 3:3:2:1:1 ratio to inject more noise into earlier sessions:
           - 30% noise into the first 20% of sessions
           - 30% noise into the first 40% of sessions
           - 20% noise into the first 60% of sessions
           - 10% noise into the first 80% of sessions
           - 10% noise into the first 100% of sessions
        3. Ensure all noise sessions are inserted.

        Args:
            original_data: Original dialogue data.
            noise_sessions: List of noise sessions.
            random_seed: Random seed for reproducibility.

        Returns:
            Data with injected noise.
        """
        if random_seed is not None:
            random.seed(random_seed)

        original_sessions = original_data.get("sessions", [])
        num_original = len(original_sessions)
        num_noise = len(noise_sessions)

        logger.info(f"[FamilyNoiseInjector] Starting injection offamily consultation noise data")
        logger.info(f"  Original session count: {num_original}")
        logger.info(f"  Noise session count: {num_noise}")

        # Distribute noise counts across intervals using weighted ratio
        insert_positions = self._generate_weighted_positions(
            num_original, num_noise
        )

        # Build result list
        result_sessions = []
        noise_idx = 0

        for orig_idx in range(num_original + 1):
            # Insert all noise sessions assigned before this position
            while noise_idx < num_noise and insert_positions[noise_idx] == orig_idx:
                noise_session = noise_sessions[noise_idx]
                result_sessions.append(noise_session.to_dict())
                noise_idx += 1

            # Then insert the original session (if any remain)
            if orig_idx < num_original:
                result_sessions.append(original_sessions[orig_idx])

        logger.info(f"  Total sessions after injection: {len(result_sessions)}")

        # Build result data
        result_data = {
            "metadata": {
                **original_data.get("metadata", {}),
                "family_noise_injection_time": datetime.now().isoformat(),
                "total_sessions_with_family_noise": len(result_sessions),
                "original_sessions": num_original,
                "family_noise_sessions_injected": num_noise,
                "family_noise_type": "family_health_consultation",
            },
            "sessions": result_sessions,
        }

        return result_data

    def save_data(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Save data with injected noise.

        Args:
            data: Data with injected noise.
            output_path: Output file path (optional).

        Returns:
            Actual path where data was saved.
        """
        if output_path is None:
            output_path = str(
                Path(self.config.data_dir) / self.config.output_filename
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[FamilyNoiseInjector] Data saved to: {output_path}")
        return output_path

    def get_injection_stats(
        self,
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get injection statistics.

        Args:
            result_data: Data with injected noise.

        Returns:
            Statistics dictionary.
        """
        sessions = result_data.get("sessions", [])

        original_count = 0
        family_noise_count = 0
        other_noise_count = 0
        noise_positions = []

        for idx, session in enumerate(sessions):
            if "noise_family_id" in session:
                family_noise_count += 1
                noise_positions.append(idx)
            elif "noise_id" in session:
                other_noise_count += 1
            else:
                original_count += 1

        # Compute noise distribution gaps
        gaps = []
        prev_pos = -1
        for pos in noise_positions:
            gaps.append(pos - prev_pos - 1)
            prev_pos = pos

        return {
            "total_sessions": len(sessions),
            "original_sessions": original_count,
            "family_noise_sessions": family_noise_count,
            "other_noise_sessions": other_noise_count,
            "family_noise_ratio": family_noise_count / len(sessions) if sessions else 0,
            "family_noise_positions": noise_positions,
            "avg_gap_between_family_noise": sum(gaps) / len(gaps) if gaps else 0,
        }

    def merge_with_existing_noise(
        self,
        data_with_existing_noise: Dict[str, Any],
        family_noise_sessions: List[FamilyNoiseSession],
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Merge family consultation noise into a dataset that already contains other noise.

        Used to inject family consultation noise on top of data that already has
        type-1 noise injected.

        Args:
            data_with_existing_noise: Data already containing other noise.
            family_noise_sessions: List of family consultation noise sessions.
            random_seed: Random seed.

        Returns:
            Merged data.
        """
        return self.inject_noise(
            data_with_existing_noise,
            family_noise_sessions,
            random_seed,
        )
