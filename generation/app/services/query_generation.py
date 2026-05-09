"""Query generation service.

Supports five query types:
- EEM: Fill-in-the-blank from a single key point
- TLA: Time-location question from a single key point
- SUA: Status-update question from all key points in a category
- MQ/IG: Two-phase generation (trap reasoning then question synthesis)
"""
import json
import logging
from typing import Optional

from ..schemas.query import (
    QueryType,
    Query,
    Answer,
    SourceKeyPoint,
)
from ..prompts.query_generation import (
    EEM_SINGLE_KP_PROMPT,
    TLA_SINGLE_KP_PROMPT,
    SUA_CATEGORY_PROMPT,
    TRAP_REASONING_PROMPT,
    MQ_FROM_TRAP_PROMPT,
    IG_FROM_TRAP_PROMPT,
)
from .llm import get_llm_service, LLMService
from ..config import get_settings
from ..schemas.dialogue import filter_kps_for_query_generation

settings = get_settings()
logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 3000


class QueryGenerationService:
    """Query generation service."""

    def __init__(
        self,
        llm: Optional[LLMService] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize the service."""
        self.llm = llm or get_llm_service()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_eem_from_single_kp(
        self,
        session_id: int,
        kp: dict,
        query_idx: int,
    ) -> Optional[Query]:
        """Generate an EEM (fill-in-the-blank) query from a single key point."""
        kp_text = self._format_single_kp(kp)

        prompt = EEM_SINGLE_KP_PROMPT.format(
            current_session_id=session_id,
            kp_text=kp_text,
        )

        caller = "query_generation.generate_eem_single"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] EEM: LLM 未Return有效 query")
                return None

            source_kp = SourceKeyPoint(
                category=kp.get("category", ""),
                name=kp.get("name", ""),
                content=kp.get("content", ""),
                trap_score=float(kp.get("trap_score", 0.5)),
                time=kp.get("time"),
                session_id=int(kp.get("session_id", session_id)),
            )

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_eem_{query_idx}",
                session_id=session_id,
                query_type=QueryType.EEM,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=[source_kp],
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} kp_name={kp.get('name')} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} error={type(e).__name__}: {str(e)}"
            )
            return None

    async def generate_tla_from_single_kp(
        self,
        session_id: int,
        kp: dict,
        query_idx: int,
    ) -> Optional[Query]:
        """Generate a TLA (time-location) query from a single key point."""
        kp_text = self._format_single_kp(kp)

        prompt = TLA_SINGLE_KP_PROMPT.format(
            current_session_id=session_id,
            kp_text=kp_text,
        )

        caller = "query_generation.generate_tla_single"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] TLA: LLM 未Return有效 query")
                return None

            source_kp = SourceKeyPoint(
                category=kp.get("category", ""),
                name=kp.get("name", ""),
                content=kp.get("content", ""),
                trap_score=float(kp.get("trap_score", 0.5)),
                time=kp.get("time"),
                session_id=int(kp.get("session_id", session_id)),
            )

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_tla_{query_idx}",
                session_id=session_id,
                query_type=QueryType.TLA,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=[source_kp],
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} kp_name={kp.get('name')} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} error={type(e).__name__}: {str(e)}"
            )
            return None

    async def generate_sua_from_category_kps(
        self,
        session_id: int,
        category: str,
        category_kps: list[dict],
        query_idx: int,
    ) -> Optional[Query]:
        """Generate an SUA (status-update) query from all key points in a category."""
        # Sort key points by time
        sorted_kps = sorted(category_kps, key=lambda x: x.get("time", "") or "")
        kps_text = self._format_category_kps(sorted_kps)

        prompt = SUA_CATEGORY_PROMPT.format(
            current_session_id=session_id,
            category=category,
            kps_text=kps_text,
            kps_count=len(category_kps),
        )

        caller = "query_generation.generate_sua_category"

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=caller,
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] SUA: LLM 未Return有效 query")
                return None

            source_kps = []
            for kp in category_kps:
                source_kps.append(SourceKeyPoint(
                    category=kp.get("category", ""),
                    name=kp.get("name", ""),
                    content=kp.get("content", ""),
                    trap_score=float(kp.get("trap_score", 0.5)),
                    time=kp.get("time"),
                    session_id=int(kp.get("session_id", session_id)),
                ))

            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            query = Query(
                query_id=f"session_{session_id}_sua_{query_idx}",
                session_id=session_id,
                query_type=QueryType.SUA,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=query_data.get("metadata", {}),
            )

            logger.info(
                f"[QueryGen Response] caller={caller} "
                f"session_id={session_id} category={category} kps_count={len(category_kps)} success=True"
            )
            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller} "
                f"session_id={session_id} category={category} error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_single_kp(self, kp: dict) -> str:
        """Format a single key point for prompt injection."""
        return (
            f"类别: {kp.get('category', '未知')}\n"
            f"名称: {kp.get('name', '未知')}\n"
            f"内容: {kp.get('content', '')}\n"
            f"Time: {kp.get('time', '未知Time')}\n"
            f"难度评分: {kp.get('trap_score', 0.5):.2f}\n"
            f"来源Session: {kp.get('session_id', 0)}"
        )

    def _format_category_kps(self, kps: list[dict]) -> str:
        """Format all key points in a category for prompt injection."""
        lines = []
        for i, kp in enumerate(kps, 1):
            lines.append(
                f"[{i}] {kp.get('name', '未知')} | "
                f"Time: {kp.get('time', '未知')} | "
                f"内容: {kp.get('content', '')}"
            )
        return "\n".join(lines)

    # ========== Two-phase generation methods ==========

    async def generate_trap_query_two_phase(
        self,
        session_id: int,
        target_kp: dict,
        all_key_points: list[dict],
        query_type: str,
        query_idx: int,
        use_layered_memory: bool = True,
        trap_score_threshold: float = 0.5,
        source_event_content: str = "",
        existing_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """Two-phase generation of trap-type queries (MQ/IG).

        Phase 1: Trap reasoning based on target kp and background info.
        Phase 2: Generate question from trap reasoning result.
        """
        caller = f"query_generation.trap_two_phase_{query_type}"

        # Filter background key points using layered memory
        if use_layered_memory:
            filtered_kps = filter_kps_for_query_generation(all_key_points, trap_score_threshold)
        else:
            filtered_kps = all_key_points

        logger.info(
            f"[QueryGen Request] caller={caller} "
            f"session_id={session_id} target_kp={target_kp.get('name')} "
            f"all_kps_count={len(all_key_points)} filtered_kps_count={len(filtered_kps)} "
            f"has_event_content={bool(source_event_content)} "
            f"existing_queries_count={len(existing_queries or [])}"
        )

        # Phase 1: Trap reasoning
        trap_reasoning = await self._phase1_trap_reasoning(
            target_kp=target_kp,
            all_key_points=filtered_kps,
            caller=caller,
            source_event_content=source_event_content,
        )

        if not trap_reasoning:
            logger.warning(
                f"[QueryGen] {query_type.upper()}: TrapReasoning阶段Failed"
            )
            return None

        # Phase 2: Generate question
        query = await self._phase2_generate_query(
            session_id=session_id,
            target_kp=target_kp,
            all_key_points=filtered_kps,
            trap_reasoning=trap_reasoning,
            query_type=query_type,
            query_idx=query_idx,
            caller=caller,
            source_event_content=source_event_content,
            existing_queries=existing_queries,
        )

        return query

    async def _phase1_trap_reasoning(
        self,
        target_kp: dict,
        all_key_points: list[dict],
        caller: str,
        source_event_content: str = "",
    ) -> Optional[dict]:
        """Phase 1: Trap reasoning based on target kp, source event, and background key points."""
        # Format background key points
        background_kps = self._format_background_kps(all_key_points, target_kp)

        # Format source event content
        if source_event_content:
            formatted_event = f"**Event内容**：{source_event_content}"
        else:
            formatted_event = "（无来源EventInfo）"

        prompt = TRAP_REASONING_PROMPT.format(
            target_category=target_kp.get("category", "未知"),
            target_name=target_kp.get("name", "未知"),
            target_content=target_kp.get("content", ""),
            target_time=target_kp.get("time", "未知Time"),
            target_session_id=target_kp.get("session_id", 0),
            source_event_content=formatted_event,
            background_kps=background_kps,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1",
            )

            if not result.get("trap_points"):
                logger.warning(
                    f"[QueryGen] Phase1: 未Generate有效的Trap要点"
                )
                return None

            logger.info(
                f"[QueryGen Response] caller={caller}_phase1 "
                f"trap_points_count={len(result.get('trap_points', []))}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _phase2_generate_query(
        self,
        session_id: int,
        target_kp: dict,
        all_key_points: list[dict],
        trap_reasoning: dict,
        query_type: str,
        query_idx: int,
        caller: str,
        source_event_content: str = "",
        existing_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """Phase 2: Generate question from trap reasoning result."""
        prompt_map = {
            "mq": MQ_FROM_TRAP_PROMPT,
            "ig": IG_FROM_TRAP_PROMPT,
        }
        query_type_map = {
            "mq": QueryType.MQ,
            "ig": QueryType.IG,
        }

        prompt_template = prompt_map.get(query_type)
        if not prompt_template:
            logger.error(f"[QueryGen] 未知的 query_type: {query_type}")
            return None

        trap_reasoning_text = json.dumps(trap_reasoning, ensure_ascii=False, indent=2)

        background_summary = self._format_background_summary(all_key_points)

        existing_queries_hint = self._format_existing_queries_hint(existing_queries)

        prompt = prompt_template.format(
            target_category=target_kp.get("category", "未知"),
            target_name=target_kp.get("name", "未知"),
            target_content=target_kp.get("content", ""),
            source_event_content=source_event_content if source_event_content else "（无来源EventInfo）",
            trap_reasoning=trap_reasoning_text,
            background_summary=background_summary,
            existing_queries_hint=existing_queries_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2",
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(
                    f"[QueryGen] Phase2: LLM 未Return有效 query"
                )
                return None

            # Parse source_key_points
            source_kps = []

            # Add target kp first
            source_kps.append(SourceKeyPoint(
                category=target_kp.get("category", ""),
                name=target_kp.get("name", ""),
                content=target_kp.get("content", ""),
                trap_score=float(target_kp.get("trap_score", 0.5)),
                time=target_kp.get("time"),
                session_id=int(target_kp.get("session_id", session_id)),
            ))

            # Add other related kps from LLM response
            for skp in query_data.get("source_key_points", []):
                try:
                    # Skip duplicate of target kp
                    if (skp.get("name") == target_kp.get("name") and
                        skp.get("content") == target_kp.get("content")):
                        continue

                    source_kps.append(SourceKeyPoint(
                        category=skp.get("category", ""),
                        name=skp.get("name", ""),
                        content=skp.get("content", ""),
                        trap_score=float(skp.get("trap_score", 0.5)),
                        time=skp.get("time"),
                        session_id=int(skp.get("session_id", session_id)),
                    ))
                except Exception as e:
                    logger.warning(
                        f"[QueryGen] source_key_point ParseFailed: {e}"
                    )

            # Parse answers
            answers = []
            for ans_data in query_data.get("answers", []):
                answers.append(Answer(
                    content=ans_data.get("content", ""),
                    is_correct=ans_data.get("is_correct", True),
                    explanation=ans_data.get("explanation"),
                ))

            metadata = query_data.get("metadata", {})

            # Add trap design info to metadata
            trap_design = query_data.get("trap_design", {})
            if trap_design:
                metadata["trap_design"] = trap_design

            # For IG type, add common wrong answer
            common_wrong = query_data.get("common_wrong_answer", {})
            if common_wrong:
                metadata["common_wrong_answer"] = common_wrong

            query = Query(
                query_id=f"session_{session_id}_{query_type}_{query_idx}",
                session_id=session_id,
                query_type=query_type_map[query_type],
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=metadata,
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase2 "
                f"session_id={session_id} query_type={query_type} success=True"
            )

            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_background_kps(self, all_key_points: list[dict], target_kp: dict) -> str:
        """Format background key points for trap reasoning, grouped by category."""
        if not all_key_points:
            return "（暂无背景Info）"

        grouped: dict[str, list[dict]] = {}
        for kp in all_key_points:
            category = kp.get("category", "未知")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(kp)

        lines = []
        for category, kps in grouped.items():
            lines.append(f"### {category}")
            for kp in kps:
                is_target = (
                    kp.get("name") == target_kp.get("name") and
                    kp.get("content") == target_kp.get("content")
                )
                marker = " ★【目标Knowledge points】" if is_target else ""

                lines.append(
                    f"- [{kp.get('name', '未知')}] {kp.get('content', '')}"
                    f" (Time: {kp.get('time', '未知')}, session: {kp.get('session_id', 0)})"
                    f"{marker}"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_background_summary(self, all_key_points: list[dict]) -> str:
        """Format a concise background info summary for question generation."""
        if not all_key_points:
            return "（暂无背景Info）"

        category_counts: dict[str, int] = {}
        key_info: list[str] = []

        for kp in all_key_points:
            category = kp.get("category", "未知")
            category_counts[category] = category_counts.get(category, 0) + 1

            content = kp.get("content", "")
            name = kp.get("name", "")

            # Extract critical info (allergies, contraindications, preferences)
            important_keywords = ["过敏", "禁忌", "不喜欢", "不能", "禁用", "偏好", "喜欢"]
            for keyword in important_keywords:
                if keyword in content or keyword in name:
                    key_info.append(f"- {name}: {content}")
                    break

        lines = ["**患者Info概览：**"]
        lines.append(f"共有 {len(all_key_points)} 条Knowledge points记录")
        lines.append(f"覆盖类别: {', '.join(category_counts.keys())}")

        if key_info:
            lines.append("\n**重要Info提示（请特别注意）：**")
            for info in list(set(key_info))[:10]:  # Max 10 items
                lines.append(info)

        return "\n".join(lines)


    def _format_existing_queries_hint(self, existing_queries: Optional[list[dict]]) -> str:
        """Format existing same-type queries as a deduplication hint for the LLM."""
        if not existing_queries:
            return ""

        lines = [
            "",
            "## ⚠️ Generated的同类题目（请避免重复！）",
            "",
            "以下是该题型Generated的题目，**请务必避免与这些题目的考点、Question内容或Answer选项重复**，寻找新的考察角度：",
            "",
        ]

        for i, q in enumerate(existing_queries, 1):
            question = q.get("question", "")
            answers = q.get("answers", [])

            lines.append(f"### 已有题目 {i}")
            lines.append(f"**Question**：{question}")

            if answers:
                lines.append("**Answer**：")
                for ans in answers:
                    content = ans.get("content", "")
                    is_correct = ans.get("is_correct", False)
                    marker = "✓" if is_correct else "✗"
                    lines.append(f"  - [{marker}] {content}")

            lines.append("")

        lines.extend([
            "**要求**：",
            "1. 不要问与上述题目相似的Question",
            "2. 不要使用相同的考察角度或TrapType",
            "3. 寻找新的医学要点来设计题目",
        ])

        return "\n".join(lines)

    # ========== MCD (Multi-hop Clinical Deduction) three-phase generation ==========

    async def generate_mcd_three_phase(
        self,
        session_id: int,
        all_key_points: list[dict],
        events_data: list[dict],
        query_idx: int,
        existing_mcd_queries: Optional[list[dict]] = None,
        sessions_data: Optional[list[dict]] = None,
        dialogues_data: Optional[list[dict]] = None,
        max_retry: int = 2,
    ) -> Optional[Query]:
        """Multi-phase generation of MCD (Multi-hop Clinical Deduction) query.

        Phase 1: Causal chain mining from events and key points across sessions.
        Phase 2: Chain validation for medical correctness.
        Phase 1.5: Improve chain based on validation feedback (up to max_retry times).
        Phase 2.5: Content enrichment using real dialogue/event data.
        Phase 3: Question synthesis from the validated chain.
        """
        caller = "query_generation.mcd_three_phase"

        # Filter data to only include session_id <= current_session_id
        filtered_key_points = [
            kp for kp in all_key_points
            if kp.get("session_id", 0) <= session_id
        ]
        filtered_sessions = [
            s for s in (sessions_data or [])
            if s.get("session_id", 0) <= session_id
        ]
        # Filter events by valid session event_ids
        valid_event_ids = set()
        for s in filtered_sessions:
            eid = s.get("event_id")
            if eid:
                valid_event_ids.add(eid)
        filtered_events = [
            e for e in events_data
            if e.get("event_id") in valid_event_ids
        ]
        filtered_dialogues = [
            d for d in (dialogues_data or [])
            if d.get("session_id", 0) <= session_id
        ]

        logger.info(
            f"[QueryGen Request] caller={caller} "
            f"session_id={session_id} kps_count={len(filtered_key_points)} "
            f"events_count={len(filtered_events)} existing_mcd_count={len(existing_mcd_queries or [])}"
        )

        # Phase 1: Causal chain mining
        candidate_chains = await self._mcd_phase1_mine_causal_chains(
            session_id=session_id,
            all_key_points=filtered_key_points,
            events_data=filtered_events,
            existing_mcd_queries=existing_mcd_queries,
            sessions_data=filtered_sessions,
            caller=caller,
        )

        if not candidate_chains:
            logger.warning(f"[QueryGen] MCD Phase1: 未挖掘到有效的Causal chain")
            return None

        # Phase 2: Chain validation with retry
        validated_chain = None
        current_chain = candidate_chains[0]  # Start with best candidate
        retry_count = 0

        while retry_count <= max_retry:
            validation_result = await self._mcd_phase2_validate_chain(
                session_id=session_id,
                candidate_chain=current_chain,
                all_key_points=filtered_key_points,
                existing_mcd_queries=existing_mcd_queries,
                caller=caller,
            )

            if validation_result.get("is_valid"):
                validated_chain = validation_result.get("refined_chain")
                break

            # Validation failed, attempt improvement
            rejection_reason = validation_result.get("rejection_reason", "未知原因")
            improvement_suggestions = validation_result.get("improvement_suggestions", [])

            logger.warning(
                f"[QueryGen] MCD Phase2: 验证不通过 (第 {retry_count + 1} 次) - {rejection_reason}"
            )

            if retry_count >= max_retry:
                logger.warning(
                    f"[QueryGen] MCD: 达到最大Retry次数 ({max_retry})，放弃Generate"
                )
                break

            # Try improving the chain
            improved_chain = await self._mcd_phase1_5_improve_chain(
                session_id=session_id,
                rejected_chain=current_chain,
                rejection_reason=rejection_reason,
                improvement_suggestions=improvement_suggestions,
                all_key_points=filtered_key_points,
                events_data=filtered_events,
                existing_mcd_queries=existing_mcd_queries,
                sessions_data=filtered_sessions,
                caller=caller,
            )

            if not improved_chain:
                logger.warning(
                    f"[QueryGen] MCD Phase1.5: 无法改进Reasoning链，尝试下一个候选"
                )
                # Try next candidate chain if available
                chain_idx = retry_count + 1
                if chain_idx < len(candidate_chains):
                    current_chain = candidate_chains[chain_idx]
                else:
                    break
            else:
                current_chain = improved_chain

            retry_count += 1

        if not validated_chain:
            logger.warning(f"[QueryGen] MCD: 所有Retry均Failed，无法Generate有效的Reasoning链")
            return None

        # Phase 2.5: Content enrichment
        enriched_chain = await self._mcd_phase2_5_enrich_chain(
            session_id=session_id,
            validated_chain=validated_chain,
            all_key_points=filtered_key_points,
            events_data=filtered_events,
            sessions_data=filtered_sessions,
            dialogues_data=filtered_dialogues,
            caller=caller,
        )

        # Phase 3: Question synthesis
        query = await self._mcd_phase3_synthesize_question(
            session_id=session_id,
            validated_chain=enriched_chain,
            all_key_points=filtered_key_points,
            query_idx=query_idx,
            caller=caller,
            existing_mcd_queries=existing_mcd_queries,
        )

        return query

    async def _mcd_phase1_mine_causal_chains(
        self,
        session_id: int,
        all_key_points: list[dict],
        events_data: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        sessions_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[list[dict]]:
        """MCD Phase 1: Mine causal chains from events and key points."""
        from ..prompts.mcd_generation import MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT

        events_timeline = self._format_events_timeline(events_data, session_id)

        kps_by_session = self._format_kps_by_session(
            all_key_points, sessions_data, current_session_id=session_id
        )

        existing_chains_hint = self._format_existing_mcd_chains(existing_mcd_queries)

        prompt = MCD_PHASE1_CAUSAL_CHAIN_MINING_PROMPT.format(
            current_session_id=session_id,
            events_timeline=events_timeline,
            knowledge_points_by_session=kps_by_session,
            existing_chains_hint=existing_chains_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1",
            )

            candidate_chains = result.get("candidate_chains", [])
            if not candidate_chains:
                logger.warning(f"[QueryGen] MCD Phase1: LLM 未Return候选链")
                return None

            # Filter out chains containing future information
            valid_chains = []
            for chain in candidate_chains:
                is_valid = True
                for node in chain.get("nodes", []):
                    node_session_id = node.get("session_id", 0)
                    # session_id=0 means medical knowledge node, allowed
                    if node_session_id > session_id and node_session_id != 0:
                        logger.warning(
                            f"[QueryGen] MCD Phase1: 候选链包含未来Info "
                            f"(node session_id={node_session_id} > current={session_id})"
                        )
                        is_valid = False
                        break
                if is_valid:
                    valid_chains.append(chain)

            if not valid_chains:
                logger.warning(f"[QueryGen] MCD Phase1: 所有候选链都包含未来Info")
                return None

            # Sort by quality_score
            valid_chains.sort(
                key=lambda x: x.get("quality_score", 0),
                reverse=True
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase1 "
                f"candidates_count={len(valid_chains)} "
                f"best_score={valid_chains[0].get('quality_score', 0):.2f}"
            )

            return valid_chains

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _mcd_phase2_validate_chain(
        self,
        session_id: int,
        candidate_chain: dict,
        all_key_points: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        caller: str,
    ) -> dict:
        """MCD Phase 2: Validate the reasoning chain for medical correctness.

        Returns a dict with:
        - is_valid: whether validation passed
        - refined_chain: refined chain (if valid)
        - rejection_reason: reason for rejection (if invalid)
        - improvement_suggestions: suggestions for improvement (if invalid)
        """
        from ..prompts.mcd_generation import MCD_PHASE2_CHAIN_VALIDATION_PROMPT

        candidate_chain_json = json.dumps(candidate_chain, ensure_ascii=False, indent=2)

        all_kps_text = self._format_all_kps_for_validation(all_key_points)

        existing_queries_hint = self._format_existing_mcd_queries_hint(existing_mcd_queries)

        prompt = MCD_PHASE2_CHAIN_VALIDATION_PROMPT.format(
            current_session_id=session_id,
            candidate_chain_json=candidate_chain_json,
            all_knowledge_points=all_kps_text,
            existing_queries_hint=existing_queries_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2",
            )

            validation_result = result.get("validation_result", {})
            is_valid = validation_result.get("is_valid", False)

            if not is_valid:
                rejection_reason = result.get("rejection_reason", "未知原因")
                improvement_suggestions = result.get("improvement_suggestions", [])
                logger.info(
                    f"[QueryGen Response] caller={caller}_phase2 "
                    f"is_valid=False reason={rejection_reason}"
                )
                return {
                    "is_valid": False,
                    "rejection_reason": rejection_reason,
                    "improvement_suggestions": improvement_suggestions,
                    "validation_details": validation_result.get("validation_details", {}),
                }

            refined_chain = result.get("refined_chain", {})
            if not refined_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase2: 未Return精化后的链")
                return {
                    "is_valid": False,
                    "rejection_reason": "验证通过但未Return精化后的链",
                    "improvement_suggestions": [],
                }

            logger.info(
                f"[QueryGen Response] caller={caller}_phase2 "
                f"is_valid=True overall_score={validation_result.get('overall_score', 0):.2f}"
            )

            return {
                "is_valid": True,
                "refined_chain": refined_chain,
                "validation_details": validation_result.get("validation_details", {}),
            }

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return {
                "is_valid": False,
                "rejection_reason": f"验证过程出错: {str(e)}",
                "improvement_suggestions": [],
            }


    async def _mcd_phase1_5_improve_chain(
        self,
        session_id: int,
        rejected_chain: dict,
        rejection_reason: str,
        improvement_suggestions: list[str],
        all_key_points: list[dict],
        events_data: list[dict],
        existing_mcd_queries: Optional[list[dict]],
        sessions_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[dict]:
        """MCD Phase 1.5: Improve chain based on validation feedback."""
        from ..prompts.mcd_generation import MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT

        rejected_chain_json = json.dumps(rejected_chain, ensure_ascii=False, indent=2)

        suggestions_text = "\n".join(
            f"- {s}" for s in improvement_suggestions
        ) if improvement_suggestions else "（无具体建议）"

        events_timeline = self._format_events_timeline(events_data, session_id)

        kps_by_session = self._format_kps_by_session(all_key_points, sessions_data)

        existing_chains_hint = self._format_existing_mcd_chains(existing_mcd_queries)

        prompt = MCD_PHASE1_5_CHAIN_IMPROVEMENT_PROMPT.format(
            current_session_id=session_id,
            rejected_chain_json=rejected_chain_json,
            rejection_reason=rejection_reason,
            improvement_suggestions=suggestions_text,
            events_timeline=events_timeline,
            knowledge_points_by_session=kps_by_session,
            existing_chains_hint=existing_chains_hint,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase1_5",
            )

            improved_chain = result.get("improved_chain", {})
            if not improved_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase1.5: LLM 未Return有效的改进链")
                return None

            # Verify improved chain has no future information
            for node in improved_chain.get("nodes", []):
                node_session_id = node.get("session_id", 0)
                if node_session_id > session_id and node_session_id != 0:
                    logger.warning(
                        f"[QueryGen] MCD Phase1.5: 改进链仍包含未来Info "
                        f"(node session_id={node_session_id} > current={session_id})"
                    )
                    return None

            improvement_made = improved_chain.get("improvement_made", "")
            logger.info(
                f"[QueryGen Response] caller={caller}_phase1_5 "
                f"improvement_made={improvement_made[:50]}..."
            )

            return improved_chain

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase1_5 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    async def _mcd_phase2_5_enrich_chain(
        self,
        session_id: int,
        validated_chain: dict,
        all_key_points: list[dict],
        events_data: list[dict],
        sessions_data: Optional[list[dict]],
        dialogues_data: Optional[list[dict]],
        caller: str,
    ) -> Optional[dict]:
        """MCD Phase 2.5: Enrich chain content using real dialogue/event data.

        Returns the enriched chain, or the original chain on failure.
        """
        from ..prompts.mcd_generation import MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT

        # Collect session IDs involved in the chain
        involved_sessions = set()
        for node in validated_chain.get("nodes", []):
            sid = node.get("session_id", 0)
            if sid > 0:
                involved_sessions.add(sid)

        if not involved_sessions:
            logger.warning(f"[QueryGen] MCD Phase2.5: not found涉及的 session")
            return validated_chain

        validated_chain_json = json.dumps(validated_chain, ensure_ascii=False, indent=2)

        dialogues_content = self._format_dialogues_for_sessions(
            dialogues_data, involved_sessions, sessions_data
        )

        events_content = self._format_events_for_sessions(
            events_data, involved_sessions, sessions_data
        )

        kps_content = self._format_kps_for_sessions(
            all_key_points, involved_sessions
        )

        prompt = MCD_PHASE2_5_CONTENT_ENRICHMENT_PROMPT.format(
            current_session_id=session_id,
            validated_chain_json=validated_chain_json,
            dialogues_content=dialogues_content,
            events_content=events_content,
            kps_content=kps_content,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase2_5",
            )

            enriched_chain = result.get("enriched_chain", {})
            if not enriched_chain.get("nodes"):
                logger.warning(f"[QueryGen] MCD Phase2.5: LLM 未Return有效的精化链，使用原链")
                return validated_chain

            enrichment_summary = enriched_chain.get("enrichment_summary", "")
            logger.info(
                f"[QueryGen Response] caller={caller}_phase2_5 "
                f"enrichment_summary={enrichment_summary[:80]}..."
            )

            return enriched_chain

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase2_5 "
                f"error={type(e).__name__}: {str(e)}"
            )
            # Fall back to original chain on error
            return validated_chain

    def _format_dialogues_for_sessions(
        self,
        dialogues_data: Optional[list[dict]],
        involved_sessions: set[int],
        sessions_data: Optional[list[dict]],
    ) -> str:
        """Format dialogue content for involved sessions."""
        if not dialogues_data:
            return "（无DialogueData）"

        # Build session_id -> event_id mapping
        session_to_event = {}
        if sessions_data:
            for s in sessions_data:
                sid = s.get("session_id", 0)
                eid = s.get("event_id")
                if sid and eid:
                    session_to_event[sid] = eid

        lines = []
        for sid in sorted(involved_sessions):
            dialogue = None
            for d in dialogues_data:
                if d.get("session_id") == sid:
                    dialogue = d
                    break

            if not dialogue:
                lines.append(f"\n### Session {sid}")
                lines.append("（not foundDialogueData）")
                continue

            lines.append(f"\n### Session {sid}")
            event_id = session_to_event.get(sid)
            if event_id:
                lines.append(f"关联EventID: {event_id}")

            turns = dialogue.get("turns", [])
            if turns:
                lines.append("Dialogue内容：")
                for turn in turns[-6:]:  # Last 6 turns max
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    role_label = "患者" if role == "user" else "医生"
                    # Truncate long content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    lines.append(f"- {role_label}: {content}")
            else:
                lines.append("（无Dialogue轮次）")

        return "\n".join(lines)

    def _format_events_for_sessions(
        self,
        events_data: list[dict],
        involved_sessions: set[int],
        sessions_data: Optional[list[dict]],
    ) -> str:
        """Format event details for involved sessions."""
        if not events_data:
            return "（无EventData）"

        # Build session_id -> event_id mapping
        session_to_event = {}
        if sessions_data:
            for s in sessions_data:
                sid = s.get("session_id", 0)
                eid = s.get("event_id")
                if sid and eid:
                    session_to_event[sid] = eid

        relevant_event_ids = {
            session_to_event.get(sid)
            for sid in involved_sessions
            if session_to_event.get(sid)
        }

        lines = []
        for event in events_data:
            eid = event.get("event_id")
            if eid not in relevant_event_ids:
                continue

            lines.append(f"\n### Event {eid}")
            lines.append(f"Date: {event.get('event_date', '未知')}")
            lines.append(f"Type: {event.get('type', '未知')}")
            lines.append(f"内容: {event.get('event', '')}")
            triggered_by = event.get("triggered_by", [])
            if triggered_by:
                lines.append(f"触发关系: 由Event {triggered_by} 触发")

        return "\n".join(lines) if lines else "（无相关Event）"

    def _format_kps_for_sessions(
        self,
        all_key_points: list[dict],
        involved_sessions: set[int],
    ) -> str:
        """Format key points for involved sessions."""
        if not all_key_points:
            return "（无Knowledge pointsData）"

        grouped: dict[int, list[dict]] = {}
        for kp in all_key_points:
            sid = kp.get("session_id", 0)
            if sid in involved_sessions:
                if sid not in grouped:
                    grouped[sid] = []
                grouped[sid].append(kp)

        lines = []
        for sid in sorted(grouped.keys()):
            kps = grouped[sid]
            lines.append(f"\n### Session {sid} Knowledge points")
            for kp in kps:
                category = kp.get("category", "未知")
                name = kp.get("name", "未知")
                content = kp.get("content", "")
                trap_score = kp.get("trap_score", 0.5)
                lines.append(f"- [{category}] {name}: {content} (trap_score={trap_score:.2f})")

        return "\n".join(lines) if lines else "（无相关Knowledge points）"

    async def _mcd_phase3_synthesize_question(
        self,
        session_id: int,
        validated_chain: dict,
        all_key_points: list[dict],
        query_idx: int,
        caller: str,
        existing_mcd_queries: Optional[list[dict]] = None,
    ) -> Optional[Query]:
        """MCD Phase 3: Synthesize question from validated reasoning chain."""
        from ..prompts.mcd_generation import MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT

        validated_chain_json = json.dumps(validated_chain, ensure_ascii=False, indent=2)

        background_summary = self._format_background_summary(all_key_points)

        # Use v2 dedup format with specificity info
        existing_questions_list = self._format_existing_questions_for_dedup_v2(
            existing_mcd_queries
        )

        prompt = MCD_PHASE3_QUESTION_SYNTHESIS_PROMPT.format(
            current_session_id=session_id,
            query_idx=query_idx,
            validated_chain_json=validated_chain_json,
            background_summary=background_summary,
            existing_questions_list=existing_questions_list,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"{caller}_phase3",
            )

            query_data = result.get("query", {})
            if not query_data:
                logger.warning(f"[QueryGen] MCD Phase3: LLM 未Return有效 query")
                return None

            diversity_check = result.get("diversity_check", {})

            # Parse answers with defensive type checks
            answers = []
            answers_raw = query_data.get("answers", [])
            if isinstance(answers_raw, list):
                for ans_data in answers_raw:
                    if isinstance(ans_data, dict):
                        answers.append(Answer(
                            content=ans_data.get("content", ""),
                            is_correct=ans_data.get("is_correct", True),
                            explanation=ans_data.get("explanation"),
                        ))
                    elif isinstance(ans_data, str):
                        # Handle string-form answers from LLM
                        answers.append(Answer(
                            content=ans_data,
                            is_correct=True,
                            explanation=None,
                        ))

            # Parse source_key_points with defensive type checks
            source_kps = []
            source_kps_raw = query_data.get("source_key_points", [])
            if isinstance(source_kps_raw, list):
                for skp in source_kps_raw:
                    if isinstance(skp, dict):
                        source_kps.append(SourceKeyPoint(
                            category=skp.get("category", ""),
                            name=skp.get("name", ""),
                            content=skp.get("content", ""),
                            trap_score=float(skp.get("trap_score", 0.5)),
                            time=skp.get("time"),
                            session_id=int(skp.get("session_id", session_id)),
                        ))

            # Build metadata with defensive type checks
            metadata_raw = query_data.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            reasoning_chain_raw = query_data.get("reasoning_chain", [])
            reasoning_chain = reasoning_chain_raw if isinstance(reasoning_chain_raw, list) else []
            metadata["reasoning_chain"] = reasoning_chain

            required_memory_nodes_raw = query_data.get("required_memory_nodes", [])
            required_memory_nodes = required_memory_nodes_raw if isinstance(required_memory_nodes_raw, list) else []
            metadata["required_memory_nodes"] = required_memory_nodes

            question_style = query_data.get("question_style", "")
            if question_style:
                metadata["question_style"] = question_style

            if diversity_check:
                metadata["diversity_check"] = diversity_check

            query = Query(
                query_id=f"session_{session_id}_mcd_{query_idx}",
                session_id=session_id,
                query_type=QueryType.MCD,
                question=query_data.get("question", ""),
                answers=answers,
                source_key_points=source_kps,
                metadata=metadata,
            )

            logger.info(
                f"[QueryGen Response] caller={caller}_phase3 "
                f"session_id={session_id} hop_count={metadata.get('hop_count', 0)} "
                f"question_style={question_style} "
                f"reasoning_pattern={metadata.get('reasoning_pattern', 'unknown')} success=True"
            )

            return query

        except Exception as e:
            logger.error(
                f"[QueryGen Error] caller={caller}_phase3 "
                f"error={type(e).__name__}: {str(e)}"
            )
            return None

    def _format_existing_questions_for_dedup(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing questions for deduplication check."""
        if not existing_mcd_queries:
            return "（暂无Generated的Question）"

        lines = []
        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})
            pattern = metadata.get("question_pattern", "未知")
            lines.append(f"{i}. [{pattern}模式] {question}")

        return "\n".join(lines)

    def _format_existing_questions_for_dedup_v2(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing questions for deduplication check (v2 with specificity info)."""
        if not existing_mcd_queries:
            return "（暂无Generated的Question）"

        lines = [
            "以下是Generated的 MCD Question，请确保新Question的**特异性入口Type**与这些不同：",
            "",
        ]

        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})

            specificity_type = metadata.get("specificity_type", "未知")
            question_pattern = metadata.get("question_pattern", "未知")

            question_design = metadata.get("question_design", {})
            professional_info = question_design.get("professional_info_used", [])
            hidden_reasoning = question_design.get("hidden_reasoning", "")

            lines.append(f"### 已有Question {i}")
            lines.append(f"**Question**：{question}")
            lines.append(f"**特异性Type**：{specificity_type}")
            lines.append(f"**Question模式**：{question_pattern}")

            if professional_info:
                lines.append(f"**涉及Info**：{', '.join(professional_info[:5])}")

            if hidden_reasoning:
                lines.append(f"**核心Reasoning**：{hidden_reasoning[:150]}...")

            lines.append("")

        lines.extend([
            "---",
            "**⚠️ 去重要求**：",
            "1. 新Question的**特异性入口Type**必须与上述Question不同",
            "2. 不能只换血糖数值，必须换**完全不同的角度**",
            "3. 如果上面已经有药物相关Question，请考虑生活Event/检查Result/症状组合等其他Type",
        ])

        return "\n".join(lines)

    def _format_events_timeline(
        self,
        events_data: list[dict],
        current_session_id: int,
    ) -> str:
        """Format events timeline for MCD generation.

        Note: events_data should already be filtered to session_id <= current_session_id.
        """
        if not events_data:
            return f"（截止 Session {current_session_id} 暂无EventData）"

        # Sort by event date
        sorted_events = sorted(events_data, key=lambda x: x.get("event_date", ""))

        lines = [f"⚠️ 注意：以下Event均为 Session {current_session_id} 及之前的Data，禁止使用任何超出此范围的Info。", ""]
        for event in sorted_events:
            event_id = event.get("event_id", 0)
            event_date = event.get("event_date", "未知Date")
            event_type = event.get("type", "未知Type")
            event_content = event.get("event", "")
            triggered_by = event.get("triggered_by", [])

            lines.append(f"[Event {event_id}] {event_date} ({event_type})")
            lines.append(f"内容: {event_content}")
            if triggered_by:
                lines.append(f"触发关系: 由Event {triggered_by} 触发")
            lines.append("---")

        return "\n".join(lines)

    def _format_kps_by_session(
        self,
        all_key_points: list[dict],
        sessions_data: Optional[list[dict]],
        current_session_id: Optional[int] = None,
    ) -> str:
        """Format key points grouped by session.

        Note: all_key_points should already be filtered to session_id <= current_session_id.
        """
        if not all_key_points:
            return "（暂无Knowledge points）"

        # Build session_id -> date mapping
        session_dates = {}
        if sessions_data:
            for session in sessions_data:
                sid = session.get("session_id")
                event_info = session.get("event_info", {})
                session_dates[sid] = event_info.get("date", "未知Date")

        grouped: dict[int, list[dict]] = {}
        for kp in all_key_points:
            sid = kp.get("session_id", 0)
            if sid not in grouped:
                grouped[sid] = []
            grouped[sid].append(kp)

        lines = []
        if current_session_id:
            lines.append(f"⚠️ 注意：以下Knowledge points均来自 Session {current_session_id} 及之前，禁止假设或使用任何超出此范围的Info。")
            lines.append("")

        max_session_in_data = max(grouped.keys()) if grouped else 0
        for sid in sorted(grouped.keys()):
            kps = grouped[sid]
            date = session_dates.get(sid, kps[0].get("time", "未知Date") if kps else "未知Date")
            lines.append(f"\n### Session {sid} ({date})")

            for kp in kps:
                lines.append(
                    f"- [{kp.get('category', '未知')}] {kp.get('name', '未知')}: "
                    f"{kp.get('content', '')}"
                )

        if current_session_id:
            lines.append(f"\n⚠️ 最大可用 Session ID: {max_session_in_data}")

        return "\n".join(lines)

    def _format_existing_mcd_chains(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing MCD reasoning chains as a deduplication hint."""
        if not existing_mcd_queries:
            return "（暂无Generated的Reasoning链）"

        lines = [
            "## ⚠️ Generated的Reasoning链（请避免重复！）",
            "",
            "以下是Generated的 MCD Question及其Reasoning链，**请务必挖掘新的、不重复的ReasoningPath**：",
            "",
        ]

        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            metadata = q.get("metadata", {})
            reasoning_chain = metadata.get("reasoning_chain", [])
            pattern = metadata.get("reasoning_pattern", "未知")

            lines.append(f"### 已有Question {i}")
            lines.append(f"**Question**: {question}")
            lines.append(f"**Reasoning模式**: {pattern}")
            lines.append(f"**Reasoning链**:")

            for node in reasoning_chain:
                node_id = node.get("node_id", 0)
                content = node.get("content", "")
                role = node.get("role", "")
                lines.append(f"  - 节点{node_id} ({role}): {content}")

            lines.append("")

        return "\n".join(lines)

    def _format_all_kps_for_validation(
        self,
        all_key_points: list[dict],
    ) -> str:
        """Format all key points for the validation phase, grouped by category."""
        if not all_key_points:
            return "（暂无Knowledge points）"

        grouped: dict[str, list[dict]] = {}
        for kp in all_key_points:
            cat = kp.get("category", "未知")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(kp)

        lines = []
        for cat, kps in grouped.items():
            lines.append(f"\n### {cat}")
            for kp in kps:
                lines.append(
                    f"- [{kp.get('name', '未知')}] {kp.get('content', '')} "
                    f"(Session {kp.get('session_id', 0)}, {kp.get('time', '未知Time')})"
                )

        return "\n".join(lines)

    def _format_existing_mcd_queries_hint(
        self,
        existing_mcd_queries: Optional[list[dict]],
    ) -> str:
        """Format existing MCD queries hint for validation phase."""
        if not existing_mcd_queries:
            return "（暂无Generated的Question）"

        lines = []
        for i, q in enumerate(existing_mcd_queries, 1):
            question = q.get("question", "")
            answers = q.get("answers", [])
            correct_answer = next(
                (a.get("content", "") for a in answers if a.get("is_correct")),
                ""
            )

            lines.append(f"{i}. Question: {question}")
            lines.append(f"   Answer: {correct_answer[:200]}...")
            lines.append("")

        return "\n".join(lines)
