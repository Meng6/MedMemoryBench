"""Query generation pipeline with per-KP deduplication."""
import json
import logging
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.services.query_generation import QueryGenerationService
from app.schemas.query import (
    Query,
    QueryGenerationRequest,
    QueryGenerationResponse,
)

logger = logging.getLogger(__name__)


class KPUsageTracker:
    """Tracks knowledge point usage across query types to prevent duplicates.

    Maintains separate usage pools for EEM, TLA, SUA, MQ/IG (shared), and MCD.
    MQ and IG share a deduplication pool. MCD has its own independent pool.
    """

    def __init__(self):
        """Initialize the tracker."""
        # Use (category, name, content) as unique kp identifier
        self.used_eem: set[tuple] = set()
        self.used_tla: set[tuple] = set()
        self.sua_category_usage_count: dict[str, int] = {}
        # MQ and IG share a single deduplication pool
        self.used_trap_kps: set[tuple] = set()
        self.generated_queries_by_type: dict[str, list[dict]] = {
            "mq": [],
            "ig": [],
            "mcd": [],
        }
        self.mcd_generated_queries: list[dict] = []

    def _kp_key(self, kp: dict) -> tuple:
        """Generate a unique key tuple for a knowledge point."""
        return (
            kp.get("category", ""),
            kp.get("name", ""),
            kp.get("content", ""),
        )

    def is_kp_used_for_eem(self, kp: dict) -> bool:
        """Check if a kp has been used for EEM."""
        return self._kp_key(kp) in self.used_eem

    def mark_kp_used_for_eem(self, kp: dict) -> None:
        """Mark a kp as used for EEM."""
        self.used_eem.add(self._kp_key(kp))

    def is_kp_used_for_tla(self, kp: dict) -> bool:
        """Check if a kp has been used for TLA."""
        return self._kp_key(kp) in self.used_tla

    def mark_kp_used_for_tla(self, kp: dict) -> None:
        """Mark a kp as used for TLA."""
        self.used_tla.add(self._kp_key(kp))

    def is_category_used_for_sua(self, category: str) -> bool:
        """Check if a category has been used for SUA (deprecated, kept for compat)."""
        return self.sua_category_usage_count.get(category, 0) > 0

    def mark_category_used_for_sua(self, category: str) -> None:
        """Mark a category as used for SUA (increments usage count)."""
        self.sua_category_usage_count[category] = (
            self.sua_category_usage_count.get(category, 0) + 1
        )

    def is_kp_used_for_type(self, kp: dict, query_type: str) -> bool:
        """Check if a kp has been used for the given query type (mq/ig)."""
        kp_key = self._kp_key(kp)
        if query_type in ("mq", "ig"):
            return kp_key in self.used_trap_kps
        return False

    def mark_kp_used_for_type(self, kp: dict, query_type: str) -> None:
        """Mark a kp as used for the given query type (mq/ig share a pool)."""
        kp_key = self._kp_key(kp)
        if query_type in ("mq", "ig"):
            self.used_trap_kps.add(kp_key)

    def get_unused_kps_for_eem(self, kps: list[dict]) -> list[dict]:
        """Return kps not yet used for EEM."""
        return [kp for kp in kps if not self.is_kp_used_for_eem(kp)]

    def get_unused_kps_for_tla(self, kps: list[dict]) -> list[dict]:
        """Return kps not yet used for TLA, preferring those with time info."""
        unused = [kp for kp in kps if not self.is_kp_used_for_tla(kp)]
        with_time = [kp for kp in unused if kp.get("time")]
        return with_time if with_time else unused

    def get_unused_categories_for_sua(self, kps: list[dict]) -> list[str]:
        """Return categories not yet used for SUA (usage count == 0)."""
        all_categories = set(kp.get("category", "") for kp in kps)
        return [cat for cat in all_categories if self.sua_category_usage_count.get(cat, 0) == 0]


    def get_category_usage_count_for_sua(self, category: str) -> int:
        """Return the usage count of a category for SUA."""
        return self.sua_category_usage_count.get(category, 0)

    def get_unused_kps_for_type(self, kps: list[dict], query_type: str) -> list[dict]:
        """Return kps not yet used for the given query type (mq/ig)."""
        return [kp for kp in kps if not self.is_kp_used_for_type(kp, query_type)]

    def random_select_kp_for_type(self, kps: list[dict], query_type: str) -> Optional[dict]:
        """Randomly select an unused kp for the given query type, or None."""
        unused = self.get_unused_kps_for_type(kps, query_type)
        if not unused:
            return None
        return random.choice(unused)

    def get_existing_queries(self, query_type: str) -> list[dict]:
        """Return previously generated queries for the given type."""
        return self.generated_queries_by_type.get(query_type, [])

    def record_generated_query(self, query_type: str, query: "Query") -> None:
        """Record a generated query (question + answers) for deduplication."""
        if query_type in self.generated_queries_by_type and query:
            query_info = {
                "question": query.question,
                "answers": [
                    {"content": ans.content, "is_correct": ans.is_correct}
                    for ans in query.answers
                ],
            }
            self.generated_queries_by_type[query_type].append(query_info)

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "eem_used_count": len(self.used_eem),
            "tla_used_count": len(self.used_tla),
            "sua_category_usage": dict(self.sua_category_usage_count),
            "trap_used_count": len(self.used_trap_kps),
            "mcd_generated_count": len(self.mcd_generated_queries),
            "generated_queries_by_type": {
                qtype: len(queries)
                for qtype, queries in self.generated_queries_by_type.items()
            },
        }

    def get_mcd_existing_queries(self) -> list[dict]:
        """Return previously generated MCD queries for deduplication."""
        return self.mcd_generated_queries

    def record_mcd_query(self, query: "Query") -> None:
        """Record a generated MCD query for deduplication."""
        if query:
            query_info = {
                "question": query.question,
                "answers": [
                    {"content": ans.content, "is_correct": ans.is_correct}
                    for ans in query.answers
                ],
                "metadata": query.metadata,
            }
            self.mcd_generated_queries.append(query_info)
            self.generated_queries_by_type["mcd"].append({
                "question": query.question,
                "answers": [
                    {"content": ans.content, "is_correct": ans.is_correct}
                    for ans in query.answers
                ],
            })


class QueryGenerator:
    """Query generator with interval-based generation and cross-session deduplication."""

    def __init__(self, request: QueryGenerationRequest):
        self.request = request
        self.service = QueryGenerationService(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        self.persona_trackers: dict[int, KPUsageTracker] = {}
        self.stats = {
            "total_sessions": 0,
            "processed_sessions": 0,
            "skipped_sessions": 0,
            "total_queries": 0,
            "queries_by_type": {},
            "errors": [],
            "start_time": None,
            "end_time": None,
        }
        self.session_event_map: dict[int, str] = {}
        self.sessions_list: list[dict] = []
        self.events_data: list[dict] = []
        self.dialogues_data: list[dict] = []

    def _get_tracker(self, persona_id: int) -> KPUsageTracker:
        """Get or create a KPUsageTracker for the given persona."""
        if persona_id not in self.persona_trackers:
            self.persona_trackers[persona_id] = KPUsageTracker()
        return self.persona_trackers[persona_id]

    async def generate(self) -> QueryGenerationResponse:
        """Execute the query generation pipeline."""
        self.stats["start_time"] = datetime.now()

        logger.info("=" * 80)
        logger.info("[QueryGen Pipeline] Starting query generation (interval mode)")
        logger.info("=" * 80)
        logger.info(f"Input: {self.request.input_file}")
        logger.info(f"Output: {self.request.output_file}")
        logger.info(
            f"Config: EEM={self.request.num_eem}, TLA={self.request.num_tla}, "
            f"SUA={self.request.num_sua}, "
            f"MQ={self.request.num_mq}, IG={self.request.num_ig}, "
            f"MCD={self.request.num_mcd}"
        )
        logger.info(f"Frequency: every {self.request.generate_every_n_sessions} sessions")
        logger.info(f"MCD frequency: every {self.request.mcd_generate_every_n_sessions} sessions")

        dialogue_data = self._load_dialogue_data()
        if not dialogue_data:
            logger.error("[QueryGen] Failed to load dialogue data")
            return self._build_response()

        sessions = dialogue_data.get("sessions", [])
        self.stats["total_sessions"] = len(sessions)
        self.sessions_list = sessions
        self.dialogues_data = sessions

        logger.info(f"[QueryGen] Found {len(sessions)} sessions")

        self._build_session_event_map(sessions)
        self.events_data = self._load_events_data()

        persona_sessions = self._group_sessions_by_persona(sessions)
        logger.info(
            f"[QueryGen] Grouped into {len(persona_sessions)} personas"
        )

        all_queries = []
        generate_interval = self.request.generate_every_n_sessions
        mcd_generate_interval = self.request.mcd_generate_every_n_sessions

        for persona_id, persona_sessions_list in persona_sessions.items():
            logger.info(f"\n[QueryGen] ===== Processing Persona {persona_id} =====")
            logger.info(f"[QueryGen] {len(persona_sessions_list)} sessions for this persona")

            tracker = self._get_tracker(persona_id)

            mcd_eval_indices = set(
                range(mcd_generate_interval - 1, len(persona_sessions_list), mcd_generate_interval)
            )

            for i in range(generate_interval - 1, len(persona_sessions_list), generate_interval):
                current_session = persona_sessions_list[i]
                session_id = current_session.get("session_id")

                logger.info(
                    f"[QueryGen] Processing Persona {persona_id} Session {session_id} "
                    f"(#{i + 1}, every {generate_interval})"
                )

                try:
                    key_points = current_session.get("knowledge_points", [])

                    if not key_points:
                        logger.warning(
                            f"[QueryGen] Session {session_id} has no knowledge points, skipping"
                        )
                        self.stats["skipped_sessions"] += 1
                        continue

                    queries = await self._generate_queries_with_dedup(
                        session_id=session_id,
                        key_points=key_points,
                        tracker=tracker,
                    )

                    if i in mcd_eval_indices and self.request.num_mcd > 0:
                        logger.info(
                            f"[QueryGen] Session {session_id} is MCD eval point "
                            f"(#{i + 1}, every {mcd_generate_interval})"
                        )
                        mcd_queries = await self._generate_mcd_queries(
                            session_id=session_id,
                            key_points=key_points,
                            tracker=tracker,
                            num_mcd=self.request.num_mcd,
                        )
                        queries.extend(mcd_queries)

                    if queries:
                        all_queries.extend(queries)
                        self.stats["processed_sessions"] += 1
                        self.stats["total_queries"] += len(queries)

                        for query in queries:
                            query_type = query.query_type.value
                            self.stats["queries_by_type"][query_type] = (
                                self.stats["queries_by_type"].get(query_type, 0) + 1
                            )

                        logger.info(
                            f"[QueryGen] Session {session_id} generated {len(queries)} queries"
                        )
                    else:
                        logger.warning(
                            f"[QueryGen] Session {session_id} produced no queries"
                        )
                        self.stats["skipped_sessions"] += 1

                except Exception as e:
                    error_msg = f"Session {session_id} processing failed: {str(e)}"
                    logger.error(f"[QueryGen] {error_msg}")
                    self.stats["errors"].append({
                        "session_id": session_id,
                        "error": str(e),
                    })
                    self.stats["skipped_sessions"] += 1

            logger.info(
                f"[QueryGen] Persona {persona_id} tracker stats: {tracker.get_stats()}"
            )

        self._export_queries(all_queries)

        self.stats["end_time"] = datetime.now()
        return self._build_response()

    async def _generate_queries_with_dedup(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
    ) -> list[Query]:
        """Generate queries with cross-session deduplication."""
        queries = []

        if self.request.num_eem > 0:
            eem_queries = await self._generate_eem_queries(
                session_id=session_id,
                key_points=key_points,
                tracker=tracker,
                num_queries=self.request.num_eem,
            )
            queries.extend(eem_queries)

        if self.request.num_tla > 0:
            tla_queries = await self._generate_tla_queries(
                session_id=session_id,
                key_points=key_points,
                tracker=tracker,
                num_queries=self.request.num_tla,
            )
            queries.extend(tla_queries)

        if self.request.num_sua > 0:
            sua_queries = await self._generate_sua_queries(
                session_id=session_id,
                key_points=key_points,
                tracker=tracker,
                num_queries=self.request.num_sua,
            )
            queries.extend(sua_queries)

        trap_queries = await self._generate_trap_queries(
            session_id=session_id,
            key_points=key_points,
            tracker=tracker,
            num_mq=self.request.num_mq,
            num_ig=self.request.num_ig,
        )
        queries.extend(trap_queries)

        return queries

    async def _generate_eem_queries(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
        num_queries: int,
    ) -> list[Query]:
        """Generate EEM queries (fill-in-the-blank).

        Selection: filter outdated kps, then sort by high trap_score + earlier time.
        """
        queries = []
        unused_kps = tracker.get_unused_kps_for_eem(key_points)

        if not unused_kps:
            logger.warning(f"[QueryGen] Session {session_id} no available kps for EEM")
            return queries

        valid_kps = self._filter_outdated_kps_for_eem(unused_kps, session_id)

        if not valid_kps:
            logger.warning(f"[QueryGen] Session {session_id} no valid kps for EEM after filtering")
            return queries

        # Sort: high trap_score first, then earlier session_id
        valid_kps.sort(
            key=lambda x: (
                -x.get("trap_score", 0.5),
                x.get("session_id", 999999),
            )
        )

        for i in range(min(num_queries, len(valid_kps))):
            kp = valid_kps[i]

            try:
                query = await self.service.generate_eem_from_single_kp(
                    session_id=session_id,
                    kp=kp,
                    query_idx=i + 1,
                )
                if query:
                    queries.append(query)
                    tracker.mark_kp_used_for_eem(kp)
                    logger.debug(
                        f"[QueryGen] EEM: used kp '{kp.get('name')}' "
                        f"(score={kp.get('trap_score', 0.5):.2f}, session={kp.get('session_id')})"
                    )
            except Exception as e:
                logger.warning(
                    f"[QueryGen] EEM generation failed: kp={kp.get('name')}, error={str(e)}"
                )

        return queries


    def _filter_outdated_kps_for_eem(
        self,
        kps: list[dict],
        current_session_id: int,
    ) -> list[dict]:
        """Filter outdated kps to avoid ambiguous EEM answers.

        For each (category, name) pair, only keep the latest record
        up to current_session_id.
        """
        latest_session: dict[tuple[str, str], int] = {}
        for kp in kps:
            key = (kp.get("category", ""), kp.get("name", ""))
            kp_session = kp.get("session_id", 0)
            if kp_session <= current_session_id:
                if key not in latest_session or kp_session > latest_session[key]:
                    latest_session[key] = kp_session

        filtered_kps = []
        for kp in kps:
            key = (kp.get("category", ""), kp.get("name", ""))
            kp_session = kp.get("session_id", 0)

            if kp_session > current_session_id:
                continue

            if kp_session == latest_session.get(key, -1):
                filtered_kps.append(kp)

        return filtered_kps

    async def _generate_tla_queries(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
        num_queries: int,
    ) -> list[Query]:
        """Generate TLA queries (temporal localization).

        Randomly selects unused kps with above-median trap_score.
        """
        queries = []
        unused_kps = tracker.get_unused_kps_for_tla(key_points)

        if not unused_kps:
            logger.warning(f"[QueryGen] Session {session_id} no available kps for TLA")
            return queries

        scores = [kp.get("trap_score", 0.5) for kp in unused_kps]
        if len(scores) > 1:
            median_score = statistics.median(scores)
        else:
            median_score = scores[0] if scores else 0.5

        above_median_kps = [
            kp for kp in unused_kps
            if kp.get("trap_score", 0.5) >= median_score and kp.get("time")
        ]

        # Fallback: all kps with time info, then all unused kps
        if len(above_median_kps) < num_queries:
            above_median_kps = [kp for kp in unused_kps if kp.get("time")]

        if not above_median_kps:
            above_median_kps = unused_kps

        logger.debug(
            f"[QueryGen] TLA: median_score={median_score:.2f}, "
            f"eligible_kps={len(above_median_kps)}"
        )

        selected_kps = []
        available_kps = above_median_kps.copy()
        for i in range(min(num_queries, len(available_kps))):
            if not available_kps:
                break
            kp = random.choice(available_kps)
            available_kps.remove(kp)
            selected_kps.append(kp)

        for i, kp in enumerate(selected_kps):
            try:
                query = await self.service.generate_tla_from_single_kp(
                    session_id=session_id,
                    kp=kp,
                    query_idx=i + 1,
                )
                if query:
                    queries.append(query)
                    tracker.mark_kp_used_for_tla(kp)
                    logger.debug(
                        f"[QueryGen] TLA: selected kp '{kp.get('name')}' "
                        f"(score={kp.get('trap_score', 0.5):.2f}, time={kp.get('time')})"
                    )
            except Exception as e:
                logger.warning(
                    f"[QueryGen] TLA generation failed: kp={kp.get('name')}, error={str(e)}"
                )

        return queries

    async def _generate_sua_queries(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
        num_queries: int,
    ) -> list[Query]:
        """Generate SUA queries (status update awareness).

        Selects categories with least usage, requiring 2+ kps from different sessions.
        """
        queries = []

        categories: dict[str, list[dict]] = {}
        for kp in key_points:
            cat = kp.get("category", "")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(kp)

        # Require 2+ kps from different sessions per category
        valid_categories: list[tuple[str, list[dict]]] = []
        for cat, kps in categories.items():
            filtered_kps = [
                kp for kp in kps
                if kp.get("session_id", 0) <= session_id
            ]
            if len(filtered_kps) < 2:
                continue
            sessions = set(kp.get("session_id", 0) for kp in filtered_kps)
            if len(sessions) >= 2:
                valid_categories.append((cat, filtered_kps))

        if not valid_categories:
            logger.warning(
                f"[QueryGen] Session {session_id} no eligible categories for SUA"
            )
            return queries

        # Sort by usage count ascending, random tiebreaker
        valid_categories.sort(
            key=lambda x: (
                tracker.get_category_usage_count_for_sua(x[0]),
                random.random(),
            )
        )

        for i in range(min(num_queries, len(valid_categories))):
            category, category_kps = valid_categories[i]

            try:
                query = await self.service.generate_sua_from_category_kps(
                    session_id=session_id,
                    category=category,
                    category_kps=category_kps,
                    query_idx=i + 1,
                )
                if query:
                    queries.append(query)
                    tracker.mark_category_used_for_sua(category)
                    logger.debug(
                        f"[QueryGen] SUA: session {session_id} used category '{category}' "
                        f"(usage_count={tracker.get_category_usage_count_for_sua(category)}, "
                        f"kps_count={len(category_kps)})"
                    )
                else:
                    logger.debug(
                        f"[QueryGen] SUA: session {session_id} category '{category}' "
                        f"no valid status change, skipping"
                    )
            except Exception as e:
                logger.warning(
                    f"[QueryGen] SUA generation failed: category={category}, error={str(e)}"
                )

        return queries

    async def _generate_trap_queries(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
        num_mq: int,
        num_ig: int,
    ) -> list[Query]:
        """Generate MQ/IG trap queries using a two-phase process.

        Phase 1: randomly pick an unused kp, retrieve its source event.
        Phase 2: LLM trap reasoning + question generation.
        """
        queries = []
        total_needed = num_mq + num_ig

        if total_needed == 0:
            return queries

        query_types = []
        if num_mq > 0:
            query_types.extend(["mq"] * num_mq)
        if num_ig > 0:
            query_types.extend(["ig"] * num_ig)

        random.shuffle(query_types)

        query_idx_counter = {"sq": 0, "mq": 0, "ig": 0}

        for query_type in query_types:
            target_kp = tracker.random_select_kp_for_type(key_points, query_type)

            if target_kp is None:
                logger.warning(
                    f"[QueryGen] Session {session_id} no available kp for {query_type.upper()}"
                )
                continue

            source_event_content = self._get_event_content_for_kp(target_kp)
            existing_queries = tracker.get_existing_queries(query_type)

            logger.debug(
                f"[QueryGen] {query_type.upper()}: selected kp '{target_kp.get('name')}' "
                f"(category={target_kp.get('category')}, session={target_kp.get('session_id')}, "
                f"has_event={bool(source_event_content)}, existing_queries={len(existing_queries)})"
            )

            try:
                query_idx_counter[query_type] += 1
                query_idx = query_idx_counter[query_type]

                query = await self.service.generate_trap_query_two_phase(
                    session_id=session_id,
                    target_kp=target_kp,
                    all_key_points=key_points,
                    query_type=query_type,
                    query_idx=query_idx,
                    source_event_content=source_event_content,
                    existing_queries=existing_queries,
                )

                if query:
                    queries.append(query)
                    tracker.mark_kp_used_for_type(target_kp, query_type)
                    tracker.record_generated_query(query_type, query)
                    logger.info(
                        f"[QueryGen] {query_type.upper()}: generated trap query "
                        f"(kp: {target_kp.get('name')})"
                    )
                else:
                    logger.warning(
                        f"[QueryGen] {query_type.upper()}: two-phase generation returned empty"
                    )

            except Exception as e:
                logger.error(
                    f"[QueryGen] {query_type.upper()} two-phase generation failed: "
                    f"kp={target_kp.get('name')}, error={type(e).__name__}: {str(e)}"
                )

        return queries

    async def _generate_mcd_queries(
        self,
        session_id: int,
        key_points: list[dict],
        tracker: KPUsageTracker,
        num_mcd: int,
    ) -> list[Query]:
        """Generate MCD (Multi-hop Clinical Deduction) queries.

        Four-phase pipeline: causal chain mining, validation, content
        refinement, and question synthesis.
        """
        queries = []

        if num_mcd <= 0:
            return queries

        existing_mcd_queries = tracker.get_mcd_existing_queries()

        logger.info(
            f"[QueryGen] MCD: generating {num_mcd} queries "
            f"({len(existing_mcd_queries)} existing for dedup)"
        )

        for idx in range(num_mcd):
            try:
                query = await self.service.generate_mcd_three_phase(
                    session_id=session_id,
                    all_key_points=key_points,
                    events_data=self.events_data,
                    query_idx=idx + 1,
                    existing_mcd_queries=existing_mcd_queries,
                    sessions_data=self.sessions_list,
                    dialogues_data=self.dialogues_data,
                )

                if query:
                    queries.append(query)
                    tracker.record_mcd_query(query)
                    existing_mcd_queries = tracker.get_mcd_existing_queries()
                    logger.info(f"[QueryGen] MCD: generated query {idx + 1}/{num_mcd}")
                else:
                    logger.warning(f"[QueryGen] MCD: query {idx + 1} generation returned empty")

            except Exception as e:
                logger.error(
                    f"[QueryGen] MCD generation failed: "
                    f"idx={idx + 1}, error={type(e).__name__}: {str(e)}"
                )

        return queries

    def _load_events_data(self) -> list[dict]:
        """Load event data for MCD generation."""
        events_file = Path(self.request.input_file).parent / "generated_events.json"
        if not events_file.exists():
            logger.warning(f"[QueryGen] Events file not found: {events_file}")
            return []

        try:
            with open(events_file, "r", encoding="utf-8") as f:
                events_data = json.load(f)

            all_events = []
            for graph in events_data.get("event_graphs", []):
                for event in graph.get("events", []):
                    all_events.append(event)

            logger.info(f"[QueryGen] Loaded {len(all_events)} events for MCD generation")
            return all_events

        except Exception as e:
            logger.error(f"[QueryGen] Failed to load events file: {e}")
            return []

    def _group_sessions_by_persona(self, sessions: list[dict]) -> dict[int, list[dict]]:
        """Group sessions by persona_id, sorted by session_id within each group."""
        persona_sessions = {}
        for session in sessions:
            persona_id = session.get("persona_id")
            if persona_id not in persona_sessions:
                persona_sessions[persona_id] = []
            persona_sessions[persona_id].append(session)

        for persona_id in persona_sessions:
            persona_sessions[persona_id].sort(key=lambda s: s.get("session_id", 0))

        return persona_sessions

    def _load_dialogue_data(self) -> Optional[dict]:
        """Load dialogue data from the input file, or None on failure."""
        input_path = Path(self.request.input_file)
        if not input_path.exists():
            logger.error(f"[QueryGen] Input file not found: {input_path}")
            return None

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[QueryGen] Loaded dialogue data: {input_path}")
            return data
        except Exception as e:
            logger.error(
                f"[QueryGen] Failed to load dialogue data: {input_path}, error={str(e)}"
            )
            return None


    def _build_session_event_map(self, sessions: list[dict]) -> None:
        """Build session_id -> event_content mapping for trap query generation."""
        events_map: dict[int, str] = {}

        events_file = Path(self.request.input_file).parent / "generated_events.json"
        if events_file.exists():
            try:
                with open(events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)

                for graph in events_data.get("event_graphs", []):
                    for event in graph.get("events", []):
                        event_id = event.get("event_id")
                        event_content = event.get("event", "")
                        if event_id and event_content:
                            events_map[event_id] = event_content

                logger.info(
                    f"[QueryGen] Loaded {len(events_map)} events for trap query generation"
                )
            except Exception as e:
                logger.warning(
                    f"[QueryGen] Failed to load events file: {e}, source event content unavailable"
                )

        for session in sessions:
            session_id = session.get("session_id")
            event_id = session.get("event_id")

            if session_id and event_id and event_id in events_map:
                self.session_event_map[session_id] = events_map[event_id]

        logger.info(
            f"[QueryGen] Built {len(self.session_event_map)} session-event mappings"
        )

    def _get_event_content_for_kp(self, target_kp: dict) -> str:
        """Return the event content for the source session of a kp, or empty string."""
        source_session_id = target_kp.get("session_id")
        if source_session_id and source_session_id in self.session_event_map:
            return self.session_event_map[source_session_id]
        return ""

    def _export_queries(self, queries: list[Query]) -> None:
        """Export generated queries to a JSON file."""
        output_path = Path(self.request.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_queries": len(queries),
                "queries_by_type": self.stats["queries_by_type"],
                "source_file": self.request.input_file,
                "generation_mode": "dedup",
            },
            "queries": [query.model_dump() for query in queries],
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[QueryGen] Queries exported to: {output_path}")
        except Exception as e:
            logger.error(
                f"[QueryGen] Failed to export queries: {output_path}, error={str(e)}"
            )

    def _build_response(self) -> QueryGenerationResponse:
        """Build the generation response with statistics."""
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]

        logger.info("=" * 80)
        logger.info("[QueryGen Pipeline] Summary")
        logger.info("=" * 80)
        logger.info(f"Total sessions: {self.stats['total_sessions']}")
        logger.info(f"Processed: {self.stats['processed_sessions']}")
        logger.info(f"Skipped: {self.stats['skipped_sessions']}")
        logger.info(f"Generated queries: {self.stats['total_queries']}")
        logger.info(f"By type: {self.stats['queries_by_type']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        if duration:
            logger.info(f"Duration: {duration}")
        logger.info(f"Output: {self.request.output_file}")
        logger.info("=" * 80)

        return QueryGenerationResponse(
            total_sessions=self.stats["total_sessions"],
            processed_sessions=self.stats["processed_sessions"],
            skipped_sessions=self.stats["skipped_sessions"],
            total_queries=self.stats["total_queries"],
            queries_by_type=self.stats["queries_by_type"],
            output_file=self.request.output_file,
            errors=self.stats["errors"],
        )
