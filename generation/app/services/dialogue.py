"""Dialogue service for handling dialogue simulation."""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Dialogue, Message, ExpandedPersona, EventGraph, EventNode, AsyncTask
from ..schemas.dialogue import DialogueGenerateRequest, BatchDialogueGenerateRequest
from ..prompts import (
    build_user_prompt_with_memory,
    build_doctor_prompt_with_memory,
    DIALOGUE_END_CHECK_PROMPT,
    KNOWLEDGE_EXTRACT_PROMPT_INITIAL,
    EVENT_SELECTION_PROMPT,
    EVENT_SELECTION_TRAP_PRIORITY_PROMPT,
    TRAP_EVENT_TYPES_SET,
)
from ..prompts.knowledge_extract import KNOWLEDGE_EXTRACT_PROMPT_ACCUMULATED, EXISTING_KEY_POINTS_FORMAT
from ..schemas.dialogue import AccumulatedKeyPoints
from ..config import get_settings
from .llm import get_llm_service, LLMService

settings = get_settings()
logger = logging.getLogger(__name__)

# Default LLM hyperparameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1500


def calculate_event_phase(
    event_date: str,
    start_date: str,
    time_span_days: int = 365,
    num_phases: int = 5,
) -> int:
    """Calculate phase number from event date.

    Args:
        event_date: Event date (YYYY-MM-DD format).
        start_date: Event graph start date (YYYY-MM-DD format).
        time_span_days: Total time span in days.
        num_phases: Total number of phases (default 5).

    Returns:
        Phase number (1 through num_phases).
    """
    try:
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 1

    days_since_start = (event_dt - start_dt).days
    if days_since_start < 0:
        return 1

    days_per_phase = time_span_days // num_phases
    if days_per_phase <= 0:
        return 1

    phase = (days_since_start // days_per_phase) + 1
    return min(max(phase, 1), num_phases)


class DialogueService:
    """Service for dialogue simulation."""

    def __init__(
        self,
        db: AsyncSession,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.db = db
        self.llm = get_llm_service()
        self.llm_small = LLMService(model=settings.llm_model_small)
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def get_all_dialogues(
        self, persona_id: Optional[int] = None
    ) -> list[Dialogue]:
        """Get all dialogues, optionally filtered by persona."""
        query = select(Dialogue).options(selectinload(Dialogue.messages))
        if persona_id:
            query = query.where(Dialogue.expanded_persona_id == persona_id)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_dialogue(self, dialogue_id: int) -> Optional[Dialogue]:
        """Get a single dialogue by ID."""
        result = await self.db.execute(
            select(Dialogue)
            .where(Dialogue.id == dialogue_id)
            .options(selectinload(Dialogue.messages))
        )
        return result.scalar_one_or_none()

    async def delete_dialogue(self, dialogue_id: int) -> bool:
        """Delete a dialogue."""
        dialogue = await self.get_dialogue(dialogue_id)
        if not dialogue:
            return False
        await self.db.delete(dialogue)
        return True

    async def generate_dialogue(
        self,
        request: DialogueGenerateRequest,
        skip_knowledge_extraction: bool = False,
        accumulated_knowledge_points: list[dict] = None,
        session_id: int = 0,
    ) -> Dialogue:
        """Generate a new dialogue using dual-agent simulation.

        Args:
            request: Dialogue generation request.
            skip_knowledge_extraction: Whether to skip knowledge extraction (controlled by pipeline).
            accumulated_knowledge_points: Accumulated knowledge points for doctor agent memory.
            session_id: Current session ID for determining disease phase (0 = no phase awareness).
        """
        # Get persona with base info
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == request.expanded_persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            raise ValueError(f"Persona {request.expanded_persona_id} not found")

        # Get event graph
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.expanded_persona_id == request.expanded_persona_id)
            .options(selectinload(EventGraph.event_nodes))
        )
        graph = result.scalar_one_or_none()
        if not graph or not graph.event_nodes:
            raise ValueError(f"No events found for persona {request.expanded_persona_id}")

        # Determine current event
        if request.event_node_id:
            current_event = next(
                (n for n in graph.event_nodes if n.id == request.event_node_id), None
            )
            if not current_event:
                raise ValueError(f"Event {request.event_node_id} not found")
        else:
            # Use latest event by event_date
            current_event = max(graph.event_nodes, key=lambda n: n.event_date)

        # Get context events (latest 3 + source events)
        context_events = self._get_context_events(graph, current_event)

        # Create dialogue
        dialogue = Dialogue(
            expanded_persona_id=request.expanded_persona_id,
            current_event_node_id=current_event.id,
            context_events=[e.id for e in context_events],
            status="in_progress",
        )
        self.db.add(dialogue)
        await self.db.flush()

        # Build agent prompts
        persona_context = self._build_persona_context(persona)
        event_context = self._build_event_context(context_events, current_event)

        # Extract disease progression from enriched persona data
        disease_progression = None
        if persona.enriched_data:
            health_details = persona.enriched_data.get("health_details", {})
            disease_progression = health_details.get("disease_progression")

        user_system = build_user_prompt_with_memory(
            persona_context=persona_context,
            event_context=event_context,
            knowledge_points=accumulated_knowledge_points or [],
            session_id=session_id,
            disease_progression=disease_progression,
        )
        doctor_system = build_doctor_prompt_with_memory(accumulated_knowledge_points or [])

        # Run dialogue simulation
        messages = []
        turn = 0

        while turn < request.max_turns:
            turn += 1

            # User agent turn
            user_messages = [{"role": "system", "content": user_system}]
            for msg in messages:
                role = "assistant" if msg.agent_type == "user_agent" else "user"
                user_messages.append({"role": role, "content": msg.content})

            if turn == 1:
                # First turn - user initiates based on current event type
                first_turn_prompt = self._build_first_turn_prompt(current_event)
                user_messages.append({
                    "role": "user",
                    "content": first_turn_prompt
                })

            user_response = await self.llm.complete(
                messages=user_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_user_turn",
            )

            user_msg = Message(
                dialogue_id=dialogue.id,
                role="user",
                content=user_response,
                agent_type="user_agent",
                turn_number=turn,
            )
            self.db.add(user_msg)
            messages.append(user_msg)

            # Doctor agent turn
            doctor_messages = [{"role": "system", "content": doctor_system}]
            for msg in messages:
                role = "user" if msg.agent_type == "user_agent" else "assistant"
                doctor_messages.append({"role": role, "content": msg.content})

            doctor_response = await self.llm.complete(
                messages=doctor_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_doctor_turn",
            )

            doctor_msg = Message(
                dialogue_id=dialogue.id,
                role="assistant",
                content=doctor_response,
                agent_type="doctor_agent",
                turn_number=turn,
            )
            self.db.add(doctor_msg)
            messages.append(doctor_msg)

            # Check if dialogue should end (every 6 turns to reduce LLM calls)
            if request.allow_natural_end and turn >= 6 and turn % 6 == 0:
                should_end = await self._check_dialogue_end(messages)
                if should_end:
                    break

        # Update dialogue status
        dialogue.status = "completed"

        # Extract knowledge points (unless skipped for pipeline control)
        if not skip_knowledge_extraction:
            knowledge_points = await self._extract_knowledge_points(messages)
            dialogue.knowledge_points = knowledge_points

        await self.db.flush()

        # Reload dialogue with messages to avoid lazy loading issues
        result = await self.db.execute(
            select(Dialogue)
            .where(Dialogue.id == dialogue.id)
            .options(selectinload(Dialogue.messages))
        )
        return result.scalar_one()

    async def continue_dialogue(
        self, dialogue_id: int, max_additional_turns: int = 5
    ) -> Optional[Dialogue]:
        """Continue an existing dialogue."""
        dialogue = await self.get_dialogue(dialogue_id)
        if not dialogue or dialogue.status != "in_progress":
            return None

        # Get persona
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == dialogue.expanded_persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            return None

        # Get event graph
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.expanded_persona_id == dialogue.expanded_persona_id)
            .options(selectinload(EventGraph.event_nodes))
        )
        graph = result.scalar_one_or_none()

        current_event = None
        if dialogue.current_event_node_id and graph:
            current_event = next(
                (n for n in graph.event_nodes if n.id == dialogue.current_event_node_id), None
            )

        context_events = []
        if graph and current_event:
            context_events = self._get_context_events(graph, current_event)

        # Build agent prompts
        persona_context = self._build_persona_context(persona)
        event_context = self._build_event_context(context_events, current_event) if context_events else ""

        # Extract disease progression from enriched persona data
        disease_progression = None
        if persona.enriched_data:
            health_details = persona.enriched_data.get("health_details", {})
            disease_progression = health_details.get("disease_progression")

        # Use existing knowledge_points as memory when continuing a dialogue
        existing_knowledge_points = dialogue.knowledge_points or []
        user_system = build_user_prompt_with_memory(
            persona_context=persona_context,
            event_context=event_context,
            knowledge_points=existing_knowledge_points,
            session_id=0,  # continue scenario does not use phase awareness
            disease_progression=disease_progression,
        )
        doctor_system = build_doctor_prompt_with_memory(existing_knowledge_points)

        messages = list(dialogue.messages)
        current_turn = max((m.turn_number for m in messages), default=0)

        for _ in range(max_additional_turns):
            current_turn += 1

            # User agent turn
            user_messages = [{"role": "system", "content": user_system}]
            for msg in messages:
                role = "assistant" if msg.agent_type == "user_agent" else "user"
                user_messages.append({"role": role, "content": msg.content})

            user_response = await self.llm.complete(
                messages=user_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_user_turn",
            )

            user_msg = Message(
                dialogue_id=dialogue.id,
                role="user",
                content=user_response,
                agent_type="user_agent",
                turn_number=current_turn,
            )
            self.db.add(user_msg)
            messages.append(user_msg)

            # Doctor agent turn
            doctor_messages = [{"role": "system", "content": doctor_system}]
            for msg in messages:
                role = "user" if msg.agent_type == "user_agent" else "assistant"
                doctor_messages.append({"role": role, "content": msg.content})

            doctor_response = await self.llm.complete(
                messages=doctor_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_doctor_turn",
            )

            doctor_msg = Message(
                dialogue_id=dialogue.id,
                role="assistant",
                content=doctor_response,
                agent_type="doctor_agent",
                turn_number=current_turn,
            )
            self.db.add(doctor_msg)
            messages.append(doctor_msg)

            # Check if dialogue should end (every 6 turns to reduce LLM calls)
            if current_turn >= 6 and current_turn % 6 == 0:
                should_end = await self._check_dialogue_end(messages)
                if should_end:
                    dialogue.status = "completed"
                    break

        # Extract knowledge points
        knowledge_points = await self._extract_knowledge_points(messages)
        dialogue.knowledge_points = knowledge_points

        await self.db.flush()

        # Reload dialogue with messages to avoid lazy loading issues
        result = await self.db.execute(
            select(Dialogue)
            .where(Dialogue.id == dialogue.id)
            .options(selectinload(Dialogue.messages))
        )
        return result.scalar_one()

    def _get_context_events(
        self, graph: EventGraph, current_event: EventNode
    ) -> list[EventNode]:
        """Get context events for dialogue: latest 3 + source events (via triggered_by)."""
        # Sort events by event_date
        sorted_events = sorted(graph.event_nodes, key=lambda n: n.event_date)

        # Get events up to current event
        past_events = [e for e in sorted_events if e.event_date <= current_event.event_date]

        # Get latest 3
        latest = past_events[-settings.event_context_limit:] if len(past_events) > settings.event_context_limit else past_events

        # Find source events through triggered_by
        source_events = set()
        if current_event.triggered_by:
            for trigger_id in current_event.triggered_by:
                source_event = next((e for e in graph.event_nodes if e.id == trigger_id), None)
                if source_event and source_event not in latest:
                    source_events.add(source_event)

        # Combine and deduplicate
        result = list(latest)
        for se in source_events:
            if se not in result:
                result.append(se)

        return sorted(result, key=lambda n: n.event_date)

    def _build_persona_context(self, persona: ExpandedPersona) -> str:
        """Build persona context for user agent."""
        base = persona.base_persona
        enriched = persona.enriched_data or {}

        parts = [
            f"性别: {base.gender if base else '未知'}",
            f"年龄段: {enriched.get('age_range', '未知')}",
            f"职业: {enriched.get('occupation_detail', '未知')}",
            f"用户类型: {base.type_name if base else '未知'}",
        ]

        health_details = enriched.get("health_details", {})
        if health_details.get("current_symptoms"):
            parts.append(f"当前症状: {', '.join(health_details['current_symptoms'])}")
        if health_details.get("medical_history"):
            parts.append(f"既往病史: {', '.join(health_details['medical_history'])}")
        if health_details.get("medications"):
            parts.append(f"用药情况: {', '.join(health_details['medications'])}")

        lifestyle = enriched.get("lifestyle", {})
        if lifestyle.get("stress_level"):
            parts.append(f"压力水平: {lifestyle['stress_level']}")

        return "\n".join(parts)

    def _build_event_context(
        self, context_events: list[EventNode], current_event: EventNode
    ) -> str:
        """Build event context for user agent, marking the current consultation topic."""
        trap_type_names = {
            "allergy": "过敏史",
            "medication_history": "用药史",
            "disease_history": "疾病史",
            "medication_preference": "给药偏好",
            "diet_preference": "饮食偏好",
            "lifestyle_economic": "生活&经济情况",
        }

        regular_type_names = {
            "health": "健康",
            "life": "生活",
            "work": "工作",
        }

        parts = []

        # Current consultation topic (primary)
        current_type = current_event.type
        if current_type in trap_type_names:
            type_label = trap_type_names[current_type]
        else:
            type_label = regular_type_names.get(current_type, current_type)

        parts.append("=" * 50)
        parts.append(f"【本次咨询主题】类型: {type_label} ({current_type})")
        parts.append(f"事件内容: {current_event.event}")
        parts.append("")
        parts.append("⚠️ 你这次对话必须围绕上述事件展开！")
        parts.append("⚠️ 事件中提到的所有具体信息（数值、药物名、症状等）都要在对话中表达出来！")
        parts.append("=" * 50)
        parts.append("")

        # Background events (for reference)
        background_events = [e for e in context_events if e.id != current_event.id]
        if background_events:
            parts.append("【背景信息】以下是你近期的其他健康相关事件，可在适当时候提及：")
            for event in background_events:
                if event.type in trap_type_names:
                    type_label = trap_type_names[event.type]
                else:
                    type_label = regular_type_names.get(event.type, event.type)
                parts.append(f"- [{event.event_date}] [{type_label}]: {event.event[:100]}{'...' if len(event.event) > 100 else ''}")

        return "\n".join(parts)

    def _build_first_turn_prompt(self, current_event: EventNode) -> str:
        """Build first-turn prompt based on current event type."""
        event_type = current_event.type

        trap_prompts = {
            "allergy": (
                "你现在需要咨询医生，在对话中自然地提及你的过敏史。"
                "请根据【本次咨询主题】中的过敏信息，告诉医生你对什么过敏、"
                "之前有过什么反应，以便医生在开药时能够避开。"
            ),
            "medication_history": (
                "你现在需要咨询医生，在对话中告诉医生你正在长期服用的药物。"
                "请根据【本次咨询主题】中的用药信息，说明你在吃什么药、"
                "为什么吃、医生有什么特别叮嘱。"
            ),
            "disease_history": (
                "你现在需要咨询医生，在对话中提及你的既往病史。"
                "请根据【本次咨询主题】中的疾病史信息，告诉医生你以前得过什么病、"
                "当时的情况、医生对你有什么提醒。"
            ),
            "medication_preference": (
                "你现在需要咨询医生，在对话中表达你对药物剂型的特殊需求。"
                "请根据【本次咨询主题】中的信息，告诉医生你在服药方面有什么困难或偏好。"
            ),
            "diet_preference": (
                "你现在需要咨询医生，在对话中自然地提及你的饮食习惯。"
                "请根据【本次咨询主题】中的饮食偏好信息，告诉医生你的饮食习惯，"
                "尤其是那些可能与医嘱冲突的地方。"
            ),
            "lifestyle_economic": (
                "你现在需要咨询医生，在对话中表达你的经济或生活方面的顾虑。"
                "请根据【本次咨询主题】中的信息，告诉医生你的医保情况、经济压力、"
                "或者工作生活中的困难。"
            ),
        }

        regular_prompts = {
            "health": (
                "你现在联系医生进行健康咨询。"
                "请根据【本次咨询主题】中描述的具体健康问题开始对话，"
                "完整描述事件中提到的症状、检查结果、数值等信息。"
            ),
            "life": (
                "你现在联系医生咨询一些生活方面的健康问题。"
                "请根据【本次咨询主题】中描述的生活变化或调整来开始对话，"
                "说明这些变化与你的健康有什么关系。"
            ),
            "work": (
                "你现在联系医生咨询工作对健康的影响。"
                "请根据【本次咨询主题】中描述的工作相关问题开始对话，"
                "说明工作对你健康造成了什么影响。"
            ),
        }

        if event_type in trap_prompts:
            prompt = trap_prompts[event_type]
        else:
            prompt = regular_prompts.get(event_type, regular_prompts["health"])

        return prompt

    async def _check_dialogue_end(self, messages: list[Message]) -> bool:
        """Check if dialogue should naturally end using small model."""
        dialogue_history = "\n".join([
            f"{'患者' if m.agent_type == 'user_agent' else '医生'}: {m.content}"
            for m in messages
        ])

        prompt = DIALOGUE_END_CHECK_PROMPT.format(dialogue_history=dialogue_history)

        try:
            # Use small model for cost efficiency
            result = await self.llm_small.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=200,
                caller="dialogue.check_dialogue_end",
            )
            return result.get("should_end", False)
        except Exception:
            return False

    async def _extract_knowledge_points(
        self,
        messages: list[Message],
        event_time: str = None,
        accumulated_key_points: AccumulatedKeyPoints = None,
        session_id: int = None,
        event_context: str = None,
    ) -> list[dict]:
        """Extract knowledge points from dialogue (deduplicate-and-accumulate only).

        Args:
            messages: List of dialogue messages.
            event_time: Event time (from event_info.date).
            accumulated_key_points: Accumulated key points (AccumulatedKeyPoints instance).
            session_id: Current session ID.
            event_context: Event background content for the current session.

        Returns:
            Newly extracted knowledge points with time and session_id fields.
        """
        dialogue_history = "\n".join([
            f"第{m.turn_number}轮 {'患者' if m.agent_type == 'user_agent' else '医生'}: {m.content}"
            for m in messages
        ])

        # Choose prompt based on whether accumulated key points exist
        if accumulated_key_points and accumulated_key_points.total_entries > 0:
            existing_text = self._format_existing_key_points(accumulated_key_points)
            prompt = KNOWLEDGE_EXTRACT_PROMPT_ACCUMULATED.format(
                current_session_id=session_id or 0,
                event_time=event_time or "未知",
                dialogue_history=dialogue_history,
                existing_key_points=existing_text,
                event_context=event_context or "（无事件背景）",
            )
        else:
            prompt = KNOWLEDGE_EXTRACT_PROMPT_INITIAL.format(
                dialogue_history=dialogue_history,
                event_context=event_context or "（无事件背景）",
            )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
                caller="dialogue.extract_knowledge_points",
            )
            raw_knowledge_points = result.get("knowledge_points", [])

            # Process extraction results and add to accumulated structure (with dedup)
            processed_kps = []
            for kp in raw_knowledge_points:
                category = kp.get("category", "")
                name = kp.get("name", "")
                content = kp.get("content", "")
                trap_score = kp.get("trap_score", 0.5)

                # Use accumulated structure for dedup
                if accumulated_key_points:
                    added = accumulated_key_points.add_if_not_exists(
                        category=category,
                        name=name,
                        content=content,
                        trap_score=trap_score,
                        time=event_time,
                        session_id=session_id or 0,
                    )
                    # Only return successfully added entries
                    if added:
                        processed_kps.append({
                            "category": category,
                            "name": name,
                            "content": content,
                            "trap_score": trap_score,
                            "time": event_time,
                            "session_id": session_id,
                        })
                else:
                    processed_kps.append({
                        "category": category,
                        "name": name,
                        "content": content,
                        "trap_score": trap_score,
                        "time": event_time,
                        "session_id": session_id,
                    })

            # Fallback: generate a key point from event content if extraction yielded nothing
            if not processed_kps and event_context:
                fallback_kp = self._generate_fallback_key_point(
                    event_context=event_context,
                    event_time=event_time,
                    session_id=session_id,
                    accumulated_key_points=accumulated_key_points,
                )
                if fallback_kp:
                    processed_kps.append(fallback_kp)

            return processed_kps
        except Exception:
            if event_context:
                fallback_kp = self._generate_fallback_key_point(
                    event_context=event_context,
                    event_time=event_time,
                    session_id=session_id,
                    accumulated_key_points=accumulated_key_points,
                )
                if fallback_kp:
                    return [fallback_kp]
            return []

    def _generate_fallback_key_point(
        self,
        event_context: str,
        event_time: str,
        session_id: int,
        accumulated_key_points: AccumulatedKeyPoints = None,
    ) -> dict | None:
        """Generate a fallback key point from event content when LLM extraction yields nothing.

        Args:
            event_context: Event background content.
            event_time: Event time.
            session_id: Current session ID.
            accumulated_key_points: Accumulated key points structure.

        Returns:
            A fallback key point dict, or None if not applicable.
        """
        if not event_context or event_context == "（无事件背景）" or event_context == "（未知事件）":
            return None

        fallback_kp = {
            "category": "疾病状况",
            "name": "就诊事件",
            "content": event_context[:100] if len(event_context) > 100 else event_context,
            "trap_score": 0.5,
            "time": event_time,
            "session_id": session_id,
        }

        # Attempt to add via accumulated structure (auto-dedup)
        if accumulated_key_points:
            added = accumulated_key_points.add_if_not_exists(
                category=fallback_kp["category"],
                name=fallback_kp["name"],
                content=fallback_kp["content"],
                trap_score=fallback_kp["trap_score"],
                time=event_time,
                session_id=session_id or 0,
            )
            if not added:
                return None

        return fallback_kp

    def _format_existing_key_points(self, accumulated: AccumulatedKeyPoints) -> str:
        """Format accumulated key points for inclusion in prompts.

        Args:
            accumulated: AccumulatedKeyPoints instance.

        Returns:
            Formatted key points string.
        """
        if not accumulated or accumulated.total_entries == 0:
            return "（暂无历史知识点）"

        lines = []
        for entry in accumulated.to_flat_list():
            time_str = entry.get("time") or "未知时间"
            line = EXISTING_KEY_POINTS_FORMAT.format(
                category=entry.get("category", ""),
                name=entry.get("name", ""),
                content=entry.get("content", ""),
                session_id=entry.get("session_id", 0),
                time=time_str,
            )
            lines.append(line)

        return "\n".join(lines)

    # ========== Batch Dialogue Generation Methods ==========

    async def select_starting_event(
        self,
        persona: ExpandedPersona,
        events: list[EventNode],
        selected_event_ids: list[int],
        ensure_trap_coverage: bool = False,
    ) -> dict:
        """Select a starting event for dialogue using LLM.

        Args:
            persona: The user persona
            events: List of available events
            selected_event_ids: List of already selected event IDs to avoid
            ensure_trap_coverage: If True, prioritize selecting trap events
                                  to ensure all 6 types are covered

        Returns:
            Dict with selected_event_id, event_summary, selection_reason, dialogue_angle
        """
        persona_context = self._build_persona_context(persona)

        # Trap event type names (Chinese labels)
        trap_type_names = {
            "allergy": "过敏史",
            "medication_history": "用药史",
            "disease_history": "疾病史",
            "medication_preference": "给药偏好",
            "diet_preference": "饮食偏好",
            "lifestyle_economic": "生活&经济情况",
        }

        # Build events list with priority indicator
        events_list_parts = []
        for e in events:
            type_label = {"health": "健康", "life": "生活", "work": "工作"}.get(e.type, e.type)
            # Show Chinese label for trap event types
            if e.type in trap_type_names:
                type_label = trap_type_names[e.type]
            priority = ""
            if e.type == "health":
                priority = "【推荐】"
            elif e.type in TRAP_EVENT_TYPES_SET:
                priority = "【陷阱事件】"
            events_list_parts.append(f"- ID: {e.id}, 类型: {type_label} {priority}, 日期: {e.event_date}, 描述: {e.event}")
        events_list = "\n".join(events_list_parts)

        # Build selected events info
        if selected_event_ids:
            selected_info = ", ".join([str(eid) for eid in selected_event_ids])
            selected_events = f"已选择的事件ID: {selected_info}"
        else:
            selected_events = "（尚未选择任何事件）"

        if ensure_trap_coverage:
            selected_trap_types = set()
            for event_id in selected_event_ids:
                event = next((e for e in events if e.id == event_id), None)
                if event and event.type in TRAP_EVENT_TYPES_SET:
                    selected_trap_types.add(event.type)

            missing_trap_types = TRAP_EVENT_TYPES_SET - selected_trap_types

            if missing_trap_types:
                missing_types_str = ", ".join([
                    f"{t} ({trap_type_names.get(t, t)})"
                    for t in missing_trap_types
                ])
                prompt = EVENT_SELECTION_TRAP_PRIORITY_PROMPT.format(
                    persona_context=persona_context,
                    events_list=events_list,
                    selected_events=selected_events,
                    missing_trap_types=missing_types_str,
                )
            else:
                prompt = EVENT_SELECTION_PROMPT.format(
                    persona_context=persona_context,
                    events_list=events_list,
                    selected_events=selected_events,
                )
        else:
            prompt = EVENT_SELECTION_PROMPT.format(
                persona_context=persona_context,
                events_list=events_list,
                selected_events=selected_events,
            )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.select_starting_event",
            )
            return result
        except Exception as e:
            # Fallback: prioritize uncovered trap events
            if ensure_trap_coverage:
                selected_trap_types = set()
                for event_id in selected_event_ids:
                    event = next((ev for ev in events if ev.id == event_id), None)
                    if event and event.type in TRAP_EVENT_TYPES_SET:
                        selected_trap_types.add(event.type)

                missing_trap_types = TRAP_EVENT_TYPES_SET - selected_trap_types
                for trap_type in missing_trap_types:
                    trap_event = next(
                        (ev for ev in events if ev.type == trap_type and ev.id not in selected_event_ids),
                        None
                    )
                    if trap_event:
                        return {
                            "selected_event_id": trap_event.id,
                            "event_summary": trap_event.event[:50],
                            "selection_reason": f"自动选择陷阱事件（LLM失败: {str(e)}）",
                            "dialogue_angle": f"透露{trap_type_names.get(trap_type, trap_type)}",
                        }

            # Final fallback: pick a health event or the first available event
            health_events = [ev for ev in events if ev.type == "health"]
            fallback_event = health_events[0] if health_events else events[0]
            return {
                "selected_event_id": fallback_event.id,
                "event_summary": fallback_event.event[:50],
                "selection_reason": f"默认选择（LLM调用失败: {str(e)}）",
                "dialogue_angle": "首诊",
            }

    def select_events_by_time_order(
        self,
        events: list[EventNode],
        count: int,
        ensure_trap_coverage: bool = True,
    ) -> list[dict]:
        """Select events in chronological order for dialogue generation.

        Args:
            events: Available event list.
            count: Number of events to select.
            ensure_trap_coverage: Whether to ensure all trap event types are covered.

        Returns:
            List of selected events in chronological order, each containing
            selected_event_id, event_summary, etc.
        """
        if not events:
            return []

        # Trap event type mapping
        trap_type_names = {
            "allergy": "过敏史",
            "medication_history": "用药史",
            "disease_history": "疾病史",
            "medication_preference": "给药偏好",
            "diet_preference": "饮食偏好",
            "lifestyle_economic": "生活&经济情况",
        }

        # Sort all events by date
        sorted_events = sorted(events, key=lambda e: e.event_date or "9999-12-31")

        selections = []
        selected_ids = set()

        # Collect trap events for coverage if needed
        trap_events_map: dict[str, EventNode] = {}
        if ensure_trap_coverage:
            for event in sorted_events:
                if event.type in TRAP_EVENT_TYPES_SET and event.type not in trap_events_map:
                    trap_events_map[event.type] = event

        for event in sorted_events:
            if len(selections) >= count:
                break

            if event.id in selected_ids:
                continue

            is_trap = event.type in TRAP_EVENT_TYPES_SET
            type_name = trap_type_names.get(event.type, event.type)

            selection = {
                "selected_event_id": event.id,
                "event_summary": event.event[:100] if event.event else "",
                "selection_reason": f"按时间顺序选取（{event.event_date}）",
                "dialogue_angle": f"透露{type_name}" if is_trap else "常规问诊",
                "event_date": event.event_date,
                "event_type": event.type,
            }

            selections.append(selection)
            selected_ids.add(event.id)

        return selections

    async def create_batch_task(self, request: BatchDialogueGenerateRequest) -> AsyncTask:
        """Create a batch dialogue generation task."""
        task = AsyncTask(
            task_type="generate_batch_dialogues",
            status="pending",
            total_items=request.count,
            completed_items=0,
            params={
                "expanded_persona_id": request.expanded_persona_id,
                "count": request.count,
                "max_turns": request.max_turns,
                "allow_natural_end": request.allow_natural_end,
            },
            result={
                "successful_dialogues": [],
                "failures": [],
            },
        )
        self.db.add(task)
        await self.db.flush()
        return task

    async def get_batch_task(self, task_id: int) -> Optional[AsyncTask]:
        """Get a batch task by ID."""
        result = await self.db.execute(
            select(AsyncTask).where(AsyncTask.id == task_id)
        )
        return result.scalar_one_or_none()

    async def _generate_single_dialogue_for_batch(
        self,
        persona: ExpandedPersona,
        graph: EventGraph,
        event_selection: dict,
        max_turns: int,
        allow_natural_end: bool,
        accumulated_knowledge_points: list[dict] = None,
        session_id: int = 0,
    ) -> dict:
        """Generate a single dialogue for batch generation.

        Args:
            persona: User persona.
            graph: Event graph.
            event_selection: Event selection result.
            max_turns: Maximum dialogue turns.
            allow_natural_end: Whether to allow natural ending.
            accumulated_knowledge_points: Accumulated knowledge points for doctor agent memory.
            session_id: Current session ID for disease phase.

        Returns:
            Dict with dialogue info or error.
        """
        # Find the selected event
        event_id = event_selection.get("selected_event_id")
        current_event = next(
            (e for e in graph.event_nodes if e.id == event_id), None
        )
        if not current_event:
            raise ValueError(f"Event {event_id} not found")

        # Get context events
        context_events = self._get_context_events(graph, current_event)

        # Create dialogue
        dialogue = Dialogue(
            expanded_persona_id=persona.id,
            current_event_node_id=current_event.id,
            context_events=[e.id for e in context_events],
            status="in_progress",
        )
        self.db.add(dialogue)
        await self.db.flush()

        # Build agent prompts
        persona_context = self._build_persona_context(persona)
        event_context = self._build_event_context(context_events, current_event)

        # Extract disease progression from enriched persona data
        disease_progression = None
        if persona.enriched_data:
            health_details = persona.enriched_data.get("health_details", {})
            disease_progression = health_details.get("disease_progression")

        user_system = build_user_prompt_with_memory(
            persona_context=persona_context,
            event_context=event_context,
            knowledge_points=accumulated_knowledge_points or [],
            session_id=session_id,
            disease_progression=disease_progression,
        )
        doctor_system = build_doctor_prompt_with_memory(accumulated_knowledge_points or [])

        # Run dialogue simulation
        messages = []
        turn = 0
        end_reason = "max_turns_reached"

        while turn < max_turns:
            turn += 1

            # User agent turn
            user_messages = [{"role": "system", "content": user_system}]
            for msg in messages:
                role = "assistant" if msg.agent_type == "user_agent" else "user"
                user_messages.append({"role": role, "content": msg.content})

            if turn == 1:
                # First turn - user initiates based on current event type
                first_turn_prompt = self._build_first_turn_prompt(current_event)
                user_messages.append({
                    "role": "user",
                    "content": first_turn_prompt
                })

            user_response = await self.llm.complete(
                messages=user_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_user_turn",
            )

            user_msg = Message(
                dialogue_id=dialogue.id,
                role="user",
                content=user_response,
                agent_type="user_agent",
                turn_number=turn,
            )
            self.db.add(user_msg)
            messages.append(user_msg)

            # Doctor agent turn
            doctor_messages = [{"role": "system", "content": doctor_system}]
            for msg in messages:
                role = "user" if msg.agent_type == "user_agent" else "assistant"
                doctor_messages.append({"role": role, "content": msg.content})

            doctor_response = await self.llm.complete(
                messages=doctor_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller="dialogue.generate_doctor_turn",
            )

            doctor_msg = Message(
                dialogue_id=dialogue.id,
                role="assistant",
                content=doctor_response,
                agent_type="doctor_agent",
                turn_number=turn,
            )
            self.db.add(doctor_msg)
            messages.append(doctor_msg)

            # Check if dialogue should end (every 6 turns to reduce LLM calls)
            if allow_natural_end and turn >= 6 and turn % 6 == 0:
                should_end = await self._check_dialogue_end(messages)
                if should_end:
                    end_reason = "natural_end"
                    break

        # Update dialogue status
        dialogue.status = "completed"

        # Extract knowledge points
        knowledge_points = await self._extract_knowledge_points(messages)
        dialogue.knowledge_points = knowledge_points

        await self.db.flush()

        return {
            "dialogue_id": dialogue.id,
            "event_id": current_event.id,
            "event_summary": event_selection.get("event_summary", current_event.event[:50]),
            "selection_reason": event_selection.get("selection_reason", ""),
            "turn_count": turn,
            "end_reason": end_reason,
        }

    async def run_batch_generation(self, task_id: int) -> None:
        """Run batch dialogue generation task in chronological event order."""
        task = await self.get_batch_task(task_id)
        if not task:
            return

        # Update task status
        task.status = "running"
        await self.db.flush()

        params = task.params
        persona_id = params["expanded_persona_id"]
        count = params["count"]
        max_turns = params["max_turns"]
        allow_natural_end = params["allow_natural_end"]

        # Get persona
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            task.status = "failed"
            task.error = f"Persona {persona_id} not found"
            await self.db.flush()
            return

        # Get event graph
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.expanded_persona_id == persona_id)
            .options(selectinload(EventGraph.event_nodes))
        )
        graph = result.scalar_one_or_none()
        if not graph or not graph.event_nodes:
            task.status = "failed"
            task.error = f"No events found for persona {persona_id}"
            await self.db.flush()
            return

        # Select events in chronological order
        event_selections = self.select_events_by_time_order(
            events=graph.event_nodes,
            count=count,
            ensure_trap_coverage=True,
        )

        if len(event_selections) < count:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[Dialogue] 事件数量不足: 需要 {count} 个，仅有 {len(event_selections)} 个可用"
            )

        # Accumulated knowledge points for doctor agent memory
        accumulated_knowledge_points: list[dict] = []

        # Generate dialogues sequentially to maintain knowledge accumulation
        successful_dialogues = []
        failures = []

        for i, selection in enumerate(event_selections):
            try:
                result_data = await self._generate_single_dialogue_for_batch(
                    persona=persona,
                    graph=graph,
                    event_selection=selection,
                    max_turns=max_turns,
                    allow_natural_end=allow_natural_end,
                    accumulated_knowledge_points=accumulated_knowledge_points,
                )
                successful_dialogues.append(result_data)

                # Accumulate knowledge points from this dialogue
                dialogue_id = result_data.get("dialogue_id")
                if dialogue_id:
                    dialogue_result = await self.db.execute(
                        select(Dialogue).where(Dialogue.id == dialogue_id)
                    )
                    dialogue = dialogue_result.scalar_one_or_none()
                    if dialogue and dialogue.knowledge_points:
                        accumulated_knowledge_points.extend(dialogue.knowledge_points)

            except Exception as e:
                failures.append({"index": i, "error": str(e)})

        # Update task
        task.status = "completed"
        task.completed_items = len(successful_dialogues)
        task.progress = 1.0
        task.result = {
            "successful_dialogues": successful_dialogues,
            "failures": failures,
        }

        await self.db.flush()
