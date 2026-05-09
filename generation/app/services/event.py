"""Event service for handling event graph operations."""
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import EventGraph, EventNode, ExpandedPersona
from ..schemas.event import EventNodeCreate, EventNodeUpdate, EventGenerateRequest
from ..prompts.trap_events import (
    TRAP_EVENT_PROMPTS,
    TRAP_EVENT_TYPE_NAMES,
    TRAP_EVENT_TYPES,
    EXTRACT_HEALTH_CONDITION_PROMPT,
)
from ..prompts.event_phased import build_phased_event_prompt, PHASE_NAMES
from .llm import get_llm_service

# Medical report file paths
USER_REPORT_PATH = Path(__file__).parent.parent.parent / "data" / "user_report.json"

# Default LLM hyperparameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 3000

logger = logging.getLogger(__name__)


class EventService:
    """Service for event graph operations."""

    def __init__(
        self,
        db: AsyncSession,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.db = db
        self.llm = get_llm_service()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def get_event_graph(self, persona_id: int) -> Optional[EventGraph]:
        """Get event graph for a persona."""
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.expanded_persona_id == persona_id)
            .options(selectinload(EventGraph.event_nodes))
        )
        return result.scalar_one_or_none()

    async def get_event_node(self, node_id: int) -> Optional[EventNode]:
        """Get a single event node by ID."""
        result = await self.db.execute(
            select(EventNode).where(EventNode.id == node_id)
        )
        return result.scalar_one_or_none()

    async def create_event_node(
        self, graph_id: int, data: EventNodeCreate
    ) -> EventNode:
        """Create a new event node."""
        node = EventNode(
            event_graph_id=graph_id,
            event=data.event,
            type=data.type,
            event_date=data.event_date,
            triggered_by=data.triggered_by,
        )
        self.db.add(node)
        await self.db.flush()
        await self.db.refresh(node)
        return node

    async def update_event_node(
        self, node_id: int, data: EventNodeUpdate
    ) -> Optional[EventNode]:
        """Update an event node."""
        node = await self.get_event_node(node_id)
        if not node:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(node, field, value)

        await self.db.flush()
        await self.db.refresh(node)
        return node

    async def delete_event_node(self, node_id: int) -> bool:
        """Delete an event node and update triggered_by references."""
        node = await self.get_event_node(node_id)
        if not node:
            return False

        # Update other nodes that reference this node in triggered_by
        result = await self.db.execute(
            select(EventNode).where(EventNode.event_graph_id == node.event_graph_id)
        )
        all_nodes = result.scalars().all()
        for other_node in all_nodes:
            if other_node.triggered_by and node_id in other_node.triggered_by:
                other_node.triggered_by = [
                    tid for tid in other_node.triggered_by if tid != node_id
                ]

        await self.db.delete(node)
        return True

    def _build_persona_context(self, persona: ExpandedPersona) -> str:
        """Build persona context string for prompt."""
        base = persona.base_persona
        enriched = persona.enriched_data or {}

        parts = [
            f"Gender: {base.gender if base else 'Unknown'}",
            f"Age range: {enriched.get('age_range', 'Unknown')}",
            f"Occupation: {enriched.get('occupation_detail', 'Unknown')}",
            f"User type: {base.type_name if base else 'Unknown'}",
            f"Core feature: {base.core_feature if base else 'Unknown'}",
            f"Health goals: {', '.join(base.health_goals) if base and base.health_goals else 'Unknown'}",
        ]

        health_details = enriched.get("health_details", {})
        if health_details.get("medical_history"):
            parts.append(f"Medical history: {', '.join(health_details['medical_history'])}")

        lifestyle = enriched.get("lifestyle", {})
        if lifestyle.get("diet_habits"):
            parts.append(f"Diet habits: {lifestyle['diet_habits']}")
        if lifestyle.get("exercise_frequency"):
            parts.append(f"Exercise frequency: {lifestyle['exercise_frequency']}")
        if lifestyle.get("stress_level"):
            parts.append(f"Stress level: {lifestyle['stress_level']}")
        if lifestyle.get("sleep_pattern"):
            parts.append(f"Sleep pattern: {lifestyle['sleep_pattern']}")

        if enriched.get("background_story"):
            parts.append(f"\nDetailed background: {enriched['background_story']}")

        return "\n".join(parts)

    def _load_treatment_report(self, persona_id: int) -> dict | None:
        """Load user treatment process report.

        Returns:
            Dict with 5-phase guidance, or None if not found.
        """
        if not USER_REPORT_PATH.exists():
            logger.warning(f"[Event] Medical report file not found: {USER_REPORT_PATH}")
            return None

        try:
            with open(USER_REPORT_PATH, "r", encoding="utf-8") as f:
                reports = json.load(f)
        except Exception as e:
            logger.error(f"[Event] Failed to load medical report: {e}")
            return None

        # Find matching persona_id
        for report_key, report_data in reports.items():
            if report_data.get("persona_id") == persona_id:
                logger.info(f"[Event] Found medical report for persona {persona_id}: {report_key}")
                return {
                    "phase_1": report_data.get("phase_1", ""),
                    "phase_2": report_data.get("phase_2", ""),
                    "phase_3": report_data.get("phase_3", ""),
                    "phase_4": report_data.get("phase_4", ""),
                    "phase_5": report_data.get("phase_5", ""),
                }

        logger.warning(f"[Event] Medical report not found for persona {persona_id}")
        return None

    def _normalize_events_data(self, events_data: list, default_date: str) -> list:
        """Normalize event data and fix common type errors.

        Ensures correct types for: temp_id (int), event (str), type (valid event type),
        event_date (date string), triggered_by (int array).
        """
        valid_types = {
            "health", "life", "work",
            "allergy", "medication_history", "disease_history",
            "medication_preference", "diet_preference", "lifestyle_economic",
        }
        normalized = []

        for i, event in enumerate(events_data):
            if not isinstance(event, dict):
                continue

            normalized_event = {}

            # temp_id: ensure integer
            temp_id = event.get("temp_id", i + 1)
            if isinstance(temp_id, str):
                try:
                    temp_id = int(temp_id)
                except ValueError:
                    temp_id = i + 1
            normalized_event["temp_id"] = temp_id

            # event: ensure string
            event_desc = event.get("event", "")
            if not isinstance(event_desc, str):
                event_desc = str(event_desc) if event_desc else ""
            normalized_event["event"] = event_desc

            # type: ensure valid event type
            event_type = event.get("type", "health")
            if not isinstance(event_type, str) or event_type not in valid_types:
                event_type = "health"
            normalized_event["type"] = event_type

            # event_date: ensure valid date string
            event_date = event.get("event_date", default_date)
            if not isinstance(event_date, str):
                event_date = str(event_date) if event_date else default_date
            # Basic date format validation
            if len(event_date) != 10 or event_date.count("-") != 2:
                event_date = default_date
            normalized_event["event_date"] = event_date

            # triggered_by: ensure integer array
            triggered_by = event.get("triggered_by", [])
            if not isinstance(triggered_by, list):
                if isinstance(triggered_by, int):
                    triggered_by = [triggered_by]
                else:
                    triggered_by = []
            # Ensure each element is an integer
            normalized_triggered_by = []
            for tid in triggered_by:
                if isinstance(tid, int):
                    normalized_triggered_by.append(tid)
                elif isinstance(tid, str):
                    try:
                        normalized_triggered_by.append(int(tid))
                    except ValueError:
                        pass
            normalized_event["triggered_by"] = normalized_triggered_by

            normalized.append(normalized_event)

        return normalized

    # ========== Trap Event Generation ==========

    async def _extract_health_condition(
        self, persona_context: str
    ) -> str:
        """Extract user health condition for trap event generation."""
        prompt = EXTRACT_HEALTH_CONDITION_PROMPT.format(
            persona_context=persona_context
        )

        try:
            health_condition = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500,
                caller="event.extract_health_condition",
            )
            return health_condition.strip()
        except Exception as e:
            logger.warning(f"[Event] Failed to extract health condition: {e}, using default")
            return "User has health-related questions requiring consultation"

    async def _generate_single_trap_event(
        self,
        trap_type: str,
        persona_context: str,
        health_condition: str,
        event_date: str,
    ) -> Optional[dict]:
        """Generate a single trap event of the specified type."""
        prompt_template = TRAP_EVENT_PROMPTS.get(trap_type)
        if not prompt_template:
            logger.error(f"[Event] Unknown trap event type: {trap_type}")
            return None

        prompt = prompt_template.format(
            persona_context=persona_context,
            health_condition=health_condition,
            event_date=event_date,
        )

        try:
            result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000,
                caller=f"event.generate_trap_{trap_type}",
            )

            # Validate returned event type
            if result.get("type") != trap_type:
                logger.warning(
                    f"[Event] Trap event type mismatch: expected {trap_type}, got {result.get('type')}"
                )
                result["type"] = trap_type

            # Ensure date is set
            if not result.get("event_date"):
                result["event_date"] = event_date

            # Trap events are independent (no triggers)
            result["triggered_by"] = []

            logger.info(
                f"[Event] ✓ GenerateTrapEvent [{TRAP_EVENT_TYPE_NAMES.get(trap_type, trap_type)}]"
            )
            return result

        except Exception as e:
            logger.error(f"[Event] ✗ GenerateTrapEvent {trap_type} Failed: {e}")
            return None

    # ========== Incremental Generation (Pipeline) ==========

    async def generate_events_incremental(
        self,
        persona_id: int,
        events_per_phase: int = 20,
        max_total_events: int = 100,
        start_date: str = "2024-01-01",
        time_span_days: int = 365,
    ) -> EventGraph:
        """Incrementally generate event graph in phases guided by treatment reports.

        Flow:
        1. Load persona data and treatment report
        2. Generate 6 fixed trap events
        3. Generate regular events for each of 5 phases guided by treatment report
        4. Randomly insert trap events into phase 1
        5. Write all events to database at once
        """
        # ========== Step 1: Load required data ==========
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            raise ValueError(f"Expanded persona {persona_id} not found")

        # Extract pure data to avoid lazy-loading ORM objects later
        persona_context = self._build_persona_context(persona)

        # Load treatment report
        treatment_report = self._load_treatment_report(persona_id)
        if treatment_report:
            logger.info(f"[Event] Loaded treatment report for persona {persona_id}")
        else:
            logger.warning(f"[Event] Persona {persona_id} has no medical report, using default phase descriptions")

        # Delete existing graph if present
        existing_graph = await self.get_event_graph(persona_id)
        if existing_graph:
            await self.db.delete(existing_graph)
            await self.db.flush()

        logger.info(
            f"[Event] Starting phased event graph generation for persona {persona_id} "
            f"(events_per_phase={events_per_phase}, max_total_events={max_total_events})"
        )

        # ========== Step 2: Generate 6 fixed trap events ==========
        logger.info("[Event] === Step 1: Generate 6 fixed trap events ===")
        trap_events = await self._generate_trap_events_pure(
            persona_context=persona_context,
            start_date=start_date,
        )
        trap_count = len(trap_events)
        logger.info(f"[Event] Generated {trap_count} trap events (IDs to be assigned later when inserted into phase 1)")

        # ========== Step 3: Generate regular events per phase ==========
        logger.info("[Event] === Step 2: Generate regular events by treatment phase ===")
        all_events: list[dict] = []
        num_phases = 5
        days_per_phase = time_span_days // num_phases

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Pre-assign temp_ids for trap events (1 to trap_count)
        for i, trap_event in enumerate(trap_events):
            trap_event["temp_id"] = i + 1

        for phase_num in range(1, num_phases + 1):
            # Calculate time range for this phase
            phase_start_dt = start_dt + timedelta(days=(phase_num - 1) * days_per_phase)
            if phase_num < num_phases:
                phase_end_dt = start_dt + timedelta(days=phase_num * days_per_phase - 1)
            else:
                # Last phase extends to end of time span
                phase_end_dt = start_dt + timedelta(days=time_span_days)

            phase_start_str = phase_start_dt.strftime("%Y-%m-%d")
            phase_end_str = phase_end_dt.strftime("%Y-%m-%d")

            # Get treatment guidance for this phase
            phase_guidance = ""
            if treatment_report:
                phase_guidance = treatment_report.get(f"phase_{phase_num}", "")

            phase_name = PHASE_NAMES.get(phase_num, f"Phase {phase_num}")
            logger.info(
                f"[Event] --- Phase {phase_num}/5: {phase_name} "
                f"({phase_start_str} ~ {phase_end_str}) ---"
            )

            # Calculate number of events to generate for this phase
            remaining = max_total_events - len(all_events) - trap_count
            num_to_generate = min(events_per_phase, remaining)

            if num_to_generate <= 0:
                logger.info(f"[Event] Max event count reached, skipping phase {phase_num}")
                break

            # Generate regular events for this phase
            # existing_events includes trap events + previously generated regular events
            existing_for_llm = trap_events + all_events
            phase_events = await self._generate_phased_events_pure(
                persona_context=persona_context,
                existing_events=existing_for_llm,
                num_events=num_to_generate,
                phase_number=phase_num,
                phase_guidance=phase_guidance,
                phase_start_date=phase_start_str,
                phase_end_date=phase_end_str,
            )

            if not phase_events:
                logger.warning(f"[Event] Phase {phase_num} failed to generate events")
                continue

            # Assign temp_ids continuing from existing events
            next_temp_id = trap_count + len(all_events) + 1
            for i, event in enumerate(phase_events):
                event["temp_id"] = next_temp_id + i

            all_events.extend(phase_events)
            logger.info(
                f"[Event] Phase {phase_num} complete, this phase {len(phase_events)} events, "
                f"total {len(all_events)} regular events (+ {trap_count} trap events)"
            )

        # ========== Step 4: Merge trap and regular events, sort by date ==========
        logger.info("[Event] === Step 3: Merge events and sort by time ===")

        # Assign random dates within phase 1 to trap events
        phase1_start_str = start_dt.strftime("%Y-%m-%d")
        phase1_end_dt = start_dt + timedelta(days=days_per_phase - 1)
        phase1_end_str = phase1_end_dt.strftime("%Y-%m-%d")

        for trap_event in trap_events:
            random_days = random.randint(0, max(0, days_per_phase - 1))
            random_date = start_dt + timedelta(days=random_days)
            trap_event["event_date"] = random_date.strftime("%Y-%m-%d")
            if "triggered_by" not in trap_event:
                trap_event["triggered_by"] = []

        # Merge all events
        final_events = trap_events + all_events

        # Sort by date
        final_events.sort(key=lambda e: (e.get("event_date", ""), e.get("temp_id", 0)))

        # Remap old temp_ids to new sequential temp_ids
        old_to_new_id: dict[int, int] = {}
        for i, event in enumerate(final_events):
            old_id = event["temp_id"]
            new_id = i + 1
            old_to_new_id[old_id] = new_id
            event["temp_id"] = new_id

        # Update triggered_by references
        for event in final_events:
            if event.get("triggered_by"):
                event["triggered_by"] = [
                    old_to_new_id.get(tid, tid) for tid in event["triggered_by"]
                    if old_to_new_id.get(tid, tid) < event["temp_id"]  # Can only reference earlier events
                ]

        logger.info(
            f"[Event] Merge complete: {len(final_events)} events "
            f"({trap_count} Trap + {len(all_events)} regular)"
        )

        # ========== Step 5: Write to database ==========
        logger.info(f"[Event] === Step 4: Write to database ({len(final_events)} events) ===")

        # Create event graph
        graph = EventGraph(
            expanded_persona_id=persona_id,
            start_date=start_date,
            time_span_days=time_span_days,
        )
        self.db.add(graph)
        await self.db.flush()

        # Create all event nodes
        temp_to_real_id: dict[int, int] = {}
        for event_data in final_events:
            node = EventNode(
                event_graph_id=graph.id,
                event=event_data.get("event", ""),
                type=event_data.get("type", "health"),
                event_date=event_data.get("event_date", start_date),
                triggered_by=[],  # Set empty first, update later
            )
            self.db.add(node)
            await self.db.flush()
            temp_to_real_id[event_data["temp_id"]] = node.id

        # Update triggered_by relationships
        for event_data in final_events:
            triggered_by_temp = event_data.get("triggered_by", [])
            if triggered_by_temp:
                real_id = temp_to_real_id.get(event_data["temp_id"])
                triggered_by_real = [
                    temp_to_real_id[tid]
                    for tid in triggered_by_temp
                    if tid in temp_to_real_id
                ]
                if real_id and triggered_by_real:
                    node_result = await self.db.execute(
                        select(EventNode).where(EventNode.id == real_id)
                    )
                    node = node_result.scalar_one()
                    node.triggered_by = triggered_by_real

        await self.db.flush()

        # Reload graph with nodes
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.id == graph.id)
            .options(selectinload(EventGraph.event_nodes))
        )
        graph = result.scalar_one()

        # Final statistics
        final_count = len(graph.event_nodes) if graph.event_nodes else 0
        regular_count = final_count - trap_count
        logger.info(
            f"[Event] Event graph generation complete: {final_count} events "
            f"({trap_count} Trap + {regular_count} regular)"
        )

        return graph

    async def generate_trap_events_only(
        self,
        persona_id: int,
        start_date: str = "2024-01-01",
    ) -> list[dict]:
        """Generate 6 fixed trap events without writing to database.

        Used as phase 1 of the two-stage generation pipeline.

        Returns:
            List of trap event dicts with temp_id, event, type, event_date, triggered_by.
        """
        # Load persona data
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            raise ValueError(f"Expanded persona {persona_id} not found")

        persona_context = self._build_persona_context(persona)
        base_persona_id = persona.base_persona_id

        logger.info(f"[Event] Starting trap event generation for persona {persona_id} (base_id={base_persona_id})")

        # Generate trap events
        trap_events = await self._generate_trap_events_pure(
            persona_context=persona_context,
            start_date=start_date,
        )

        # Assign temp_ids
        for i, event in enumerate(trap_events):
            event["temp_id"] = i + 1
            event["persona_id"] = persona_id

        logger.info(f"[Event] Persona {persona_id} (base_id={base_persona_id}) generated {len(trap_events)} trap events")
        return trap_events

    async def generate_regular_events_only(
        self,
        persona_id: int,
        trap_events: list[dict],
        events_per_phase: int = 20,
        max_total_events: int = 100,
        start_date: str = "2024-01-01",
        time_span_days: int = 365,
    ) -> EventGraph:
        """Generate regular events based on existing trap events.

        Used as phase 2 of the two-stage generation pipeline.
        Inserts trap events randomly into phase 1, then generates regular events per phase.
        """
        # Load persona data
        result = await self.db.execute(
            select(ExpandedPersona)
            .where(ExpandedPersona.id == persona_id)
            .options(selectinload(ExpandedPersona.base_persona))
        )
        persona = result.scalar_one_or_none()
        if not persona:
            raise ValueError(f"Expanded persona {persona_id} not found")

        persona_context = self._build_persona_context(persona)

        # Get base_persona_id for loading treatment report
        base_persona_id = persona.base_persona_id

        # Load treatment report using base_persona_id (not expanded_persona_id)
        treatment_report = self._load_treatment_report(base_persona_id)
        if treatment_report:
            logger.info(f"[Event] Loaded treatment report for persona {persona_id} (base_id={base_persona_id})")
        else:
            logger.warning(f"[Event] Persona {persona_id} (base_id={base_persona_id}) has no medical report, using default phase descriptions")

        # Delete existing graph if present
        existing_graph = await self.get_event_graph(persona_id)
        if existing_graph:
            await self.db.delete(existing_graph)
            await self.db.flush()

        logger.info(
            f"[Event] Starting regular event generation for persona {persona_id} "
            f"(events_per_phase={events_per_phase}, max={max_total_events})"
        )

        # Clean trap events (remove persona_id field for processing)
        trap_events_clean = []
        for e in trap_events:
            clean_event = {k: v for k, v in e.items() if k != "persona_id"}
            trap_events_clean.append(clean_event)
        trap_count = len(trap_events_clean)

        # Pre-assign temp_ids for trap events (1 to trap_count)
        for i, trap_event in enumerate(trap_events_clean):
            trap_event["temp_id"] = i + 1

        # Generate regular events for each of 5 phases
        all_events: list[dict] = []
        num_phases = 5
        days_per_phase = time_span_days // num_phases

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        for phase_num in range(1, num_phases + 1):
            # Calculate time range for this phase
            phase_start_dt = start_dt + timedelta(days=(phase_num - 1) * days_per_phase)
            if phase_num < num_phases:
                phase_end_dt = start_dt + timedelta(days=phase_num * days_per_phase - 1)
            else:
                phase_end_dt = start_dt + timedelta(days=time_span_days)

            phase_start_str = phase_start_dt.strftime("%Y-%m-%d")
            phase_end_str = phase_end_dt.strftime("%Y-%m-%d")

            # Get treatment guidance for this phase
            phase_guidance = ""
            if treatment_report:
                phase_guidance = treatment_report.get(f"phase_{phase_num}", "")

            phase_name = PHASE_NAMES.get(phase_num, f"Phase {phase_num}")
            logger.info(
                f"[Event] --- Phase {phase_num}/5: {phase_name} "
                f"({phase_start_str} ~ {phase_end_str}) ---"
            )

            # Calculate number of events to generate for this phase
            remaining = max_total_events - len(all_events)
            num_to_generate = min(events_per_phase, remaining)

            if num_to_generate <= 0:
                logger.info(f"[Event] Max event count reached, skipping phase {phase_num}")
                break

            # Generate regular events for this phase
            # existing_events includes trap events + previously generated regular events
            existing_for_llm = trap_events_clean + all_events
            phase_events = await self._generate_phased_events_pure(
                persona_context=persona_context,
                existing_events=existing_for_llm,
                num_events=num_to_generate,
                phase_number=phase_num,
                phase_guidance=phase_guidance,
                phase_start_date=phase_start_str,
                phase_end_date=phase_end_str,
            )

            if not phase_events:
                logger.warning(f"[Event] Phase {phase_num} failed to generate events")
                continue

            # Assign temp_ids continuing from existing events
            next_temp_id = trap_count + len(all_events) + 1
            for i, event in enumerate(phase_events):
                event["temp_id"] = next_temp_id + i

            all_events.extend(phase_events)
            logger.info(
                f"[Event] Phase {phase_num} complete, this phase {len(phase_events)} events, "
                f"total {len(all_events)} regular events (+ {trap_count} trap events)"
            )

        # Merge trap and regular events, sort by date
        logger.info("[Event] Merging events and sorting by time...")

        # Assign random dates within phase 1 to trap events
        for trap_event in trap_events_clean:
            random_days = random.randint(0, max(0, days_per_phase - 1))
            random_date = start_dt + timedelta(days=random_days)
            trap_event["event_date"] = random_date.strftime("%Y-%m-%d")
            if "triggered_by" not in trap_event:
                trap_event["triggered_by"] = []

        # Merge all events
        final_events = trap_events_clean + all_events

        # Sort by date
        final_events.sort(key=lambda e: (e.get("event_date", ""), e.get("temp_id", 0)))

        # Remap old temp_ids to new sequential temp_ids
        old_to_new_id: dict[int, int] = {}
        for i, event in enumerate(final_events):
            old_id = event["temp_id"]
            new_id = i + 1
            old_to_new_id[old_id] = new_id
            event["temp_id"] = new_id

        # Update triggered_by references
        for event in final_events:
            if event.get("triggered_by"):
                event["triggered_by"] = [
                    old_to_new_id.get(tid, tid) for tid in event["triggered_by"]
                    if old_to_new_id.get(tid, tid) < event["temp_id"]  # Can only reference earlier events
                ]

        logger.info(
            f"[Event] Merge complete: {len(final_events)} events "
            f"({trap_count} Trap + {len(all_events)} regular)"
        )

        # Write to database
        logger.info(f"[Event] Writing to database ({len(final_events)} events)")

        graph = EventGraph(
            expanded_persona_id=persona_id,
            start_date=start_date,
            time_span_days=time_span_days,
        )
        self.db.add(graph)
        await self.db.flush()

        # Create all event nodes
        temp_to_real_id: dict[int, int] = {}
        for event_data in final_events:
            node = EventNode(
                event_graph_id=graph.id,
                event=event_data.get("event", ""),
                type=event_data.get("type", "health"),
                event_date=event_data.get("event_date", start_date),
                triggered_by=[],
            )
            self.db.add(node)
            await self.db.flush()
            temp_to_real_id[event_data["temp_id"]] = node.id

        # Update triggered_by relationships
        for event_data in final_events:
            triggered_by_temp = event_data.get("triggered_by", [])
            if triggered_by_temp:
                real_id = temp_to_real_id.get(event_data["temp_id"])
                triggered_by_real = [
                    temp_to_real_id[tid]
                    for tid in triggered_by_temp
                    if tid in temp_to_real_id
                ]
                if real_id and triggered_by_real:
                    node_result = await self.db.execute(
                        select(EventNode).where(EventNode.id == real_id)
                    )
                    node = node_result.scalar_one()
                    node.triggered_by = triggered_by_real

        await self.db.flush()

        # Reload graph with nodes
        result = await self.db.execute(
            select(EventGraph)
            .where(EventGraph.id == graph.id)
            .options(selectinload(EventGraph.event_nodes))
        )
        graph = result.scalar_one()

        final_count = len(graph.event_nodes) if graph.event_nodes else 0
        regular_count = final_count - trap_count
        logger.info(
            f"[Event] Event graph generation complete: {final_count} events "
            f"({trap_count} Trap + {regular_count} regular)"
        )

        return graph

    async def _generate_trap_events_pure(
        self,
        persona_context: str,
        start_date: str,
    ) -> list[dict]:
        """Generate trap events (pure in-memory operation, no database).

        Returns:
            List of trap event dicts.
        """
        # Extract health condition
        health_condition = await self._extract_health_condition(persona_context)
        logger.info(f"[Event] User health condition: {health_condition[:100]}...")

        # Generate content for each trap event type
        trap_events = []
        for trap_type in TRAP_EVENT_TYPES:
            logger.info(
                f"[Event] GenerateTrapEvent: {TRAP_EVENT_TYPE_NAMES.get(trap_type, trap_type)}"
            )
            event_data = await self._generate_single_trap_event(
                trap_type=trap_type,
                persona_context=persona_context,
                health_condition=health_condition,
                event_date=start_date,
            )
            if event_data:
                trap_events.append(event_data)

        logger.info(f"[Event] Generated {len(trap_events)}/6 trap events")
        return trap_events

    def _insert_trap_events_randomly(
        self,
        regular_events: list[dict],
        trap_events: list[dict],
        phase_start_date: str,
        phase_end_date: str,
    ) -> list[dict]:
        """Randomly insert trap events into the regular event list.

        Trap events are inserted at random positions (not date-sorted) to test
        model's ability to handle timeline disorder.
        """
        if not trap_events:
            return regular_events

        start_dt = datetime.strptime(phase_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(phase_end_date, "%Y-%m-%d")
        date_range_days = (end_dt - start_dt).days

        # Assign random dates to each trap event
        for trap_event in trap_events:
            random_days = random.randint(0, max(0, date_range_days))
            random_date = start_dt + timedelta(days=random_days)
            trap_event["event_date"] = random_date.strftime("%Y-%m-%d")
            # Trap events are independent (no triggers)
            if "triggered_by" not in trap_event:
                trap_event["triggered_by"] = []

        # Insert trap events at random positions
        result = regular_events.copy()
        for trap_event in trap_events:
            insert_pos = random.randint(0, len(result))
            result.insert(insert_pos, trap_event)

        logger.info(
            f"[Event] Trap event random insertion complete: {len(trap_events)} trap events + "
            f"{len(regular_events)} regular events = {len(result)} events"
        )

        return result

    async def _generate_phased_events_pure(
        self,
        persona_context: str,
        existing_events: list[dict],
        num_events: int,
        phase_number: int,
        phase_guidance: str,
        phase_start_date: str,
        phase_end_date: str,
        max_retries: int = 3,
    ) -> list[dict]:
        """Generate events for a specific phase guided by treatment report (pure in-memory).

        Returns:
            List of new event dicts.
        """
        # Format existing events for prompt
        existing_events_str = self._format_existing_events_pure(existing_events)

        # Calculate starting temp_id
        max_existing_id = max((e.get("temp_id", 0) for e in existing_events), default=0)
        start_temp_id = max_existing_id + 1

        # Build phased prompt
        prompt = build_phased_event_prompt(
            persona_context=persona_context,
            phase_number=phase_number,
            phase_guidance=phase_guidance,
            existing_events=existing_events_str,
            start_date=phase_start_date,
            end_date=phase_end_date,
            num_events=num_events,
            start_temp_id=start_temp_id,
            max_existing_id=max_existing_id,
        )

        phase_name = PHASE_NAMES.get(phase_number, f"Phase {phase_number}")
        logger.info(f"[Event] Generating {num_events} events for {phase_name}...")

        # LLM call with retries
        events_data = []
        for attempt in range(1, max_retries + 1):
            llm_result = await self.llm.complete_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caller=f"event.generate_phased_events_phase_{phase_number}",
            )

            # Normalize new event data
            events_data = llm_result.get("events", [])
            logger.info(f"[Event] LLM returned {len(events_data)} events (before normalization, attempt {attempt}/{max_retries})")

            events_data = self._normalize_events_data(events_data, phase_start_date)
            logger.info(f"[Event] After normalization: {len(events_data)} events")

            # Break if events generated successfully
            if events_data:
                break

            # Retry if empty result
            if attempt < max_retries:
                logger.warning(
                    f"[Event] {phase_name} returned empty result, retry {attempt + 1}..."
                )

        return events_data

    async def _generate_phased_events_layered(
        self,
        persona_context: str,
        existing_events: list[dict],
        num_events: int,
        phase_number: int,
        phase_guidance: str,
        phase_start_date: str,
        phase_end_date: str,
        core_event_ratio: float = 0.3,
    ) -> list[dict]:
        """Layered event generation: extract core events first, then generate derived events.

        NOTE: Currently unused as dependent prompt functions are not yet implemented.
        To enable, implement in event_phased.py:
        - build_core_events_extract_prompt
        - build_derived_events_prompt
        """
        # Not yet implemented, fall back to standard generation
        logger.warning(
            "[Event] _generate_phased_events_layered not fully implemented, falling back to standard generation"
        )
        return await self._generate_phased_events_pure(
            persona_context=persona_context,
            existing_events=existing_events,
            num_events=num_events,
            phase_number=phase_number,
            phase_guidance=phase_guidance,
            phase_start_date=phase_start_date,
            phase_end_date=phase_end_date,
        )

    def _format_existing_events_pure(self, events: list[dict]) -> str:
        """Format existing event list for use in prompts (pure dict version)."""
        if not events:
            return "(No existing events)"

        # Sort by date
        sorted_events = sorted(events, key=lambda e: e.get("event_date", ""))

        lines = []
        for event in sorted_events:
            triggered_by = event.get("triggered_by", [])
            triggered_by_str = (
                f"(Triggered by event {triggered_by})"
                if triggered_by
                else "(Independent event)"
            )
            event_text = event.get("event", "")
            lines.append(
                f"[ID:{event.get('temp_id', '?')}] [{event.get('event_date', '?')}] "
                f"[{event.get('type', '?')}] "
                f"{event_text[:100]}{'...' if len(event_text) > 100 else ''} "
                f"{triggered_by_str}"
            )

        return "\n".join(lines)
