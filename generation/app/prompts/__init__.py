"""LLM prompt templates."""
from .persona_enrich import PERSONA_ENRICH_PROMPT, build_enrich_prompt
from .event_generate import (
    EVENT_TYPES,
    TRAP_EVENT_TYPES,
)
from .trap_events import (
    TRAP_EVENT_PROMPTS,
    TRAP_EVENT_TYPE_NAMES,
    TRAP_EVENT_TYPES as TRAP_EVENT_TYPES_LIST,
    EXTRACT_HEALTH_CONDITION_PROMPT,
)
from .user_agent import (
    USER_AGENT_SYSTEM_PROMPT,
    USER_AGENT_SYSTEM_PROMPT_WITH_MEMORY,
    build_user_prompt_with_memory,
)
from .doctor_agent import (
    DOCTOR_AGENT_SYSTEM_PROMPT,
    DOCTOR_AGENT_SYSTEM_PROMPT_WITH_MEMORY,
    build_doctor_prompt_with_memory,
)
from .dialogue_end import DIALOGUE_END_CHECK_PROMPT
from .knowledge_extract import KNOWLEDGE_EXTRACT_PROMPT, KNOWLEDGE_EXTRACT_PROMPT_INITIAL
from .event_selection import EVENT_SELECTION_PROMPT, EVENT_SELECTION_TRAP_PRIORITY_PROMPT, TRAP_EVENT_TYPES_SET
from .event_phased import (
    PHASE_NAMES,
    PHASE_DESCRIPTIONS,
    build_phased_event_prompt,
)

__all__ = [
    "PERSONA_ENRICH_PROMPT",
    "build_enrich_prompt",
    "EVENT_TYPES",
    "TRAP_EVENT_TYPES",
    "TRAP_EVENT_PROMPTS",
    "TRAP_EVENT_TYPE_NAMES",
    "TRAP_EVENT_TYPES_LIST",
    "EXTRACT_HEALTH_CONDITION_PROMPT",
    "USER_AGENT_SYSTEM_PROMPT",
    "USER_AGENT_SYSTEM_PROMPT_WITH_MEMORY",
    "build_user_prompt_with_memory",
    "DOCTOR_AGENT_SYSTEM_PROMPT",
    "DOCTOR_AGENT_SYSTEM_PROMPT_WITH_MEMORY",
    "build_doctor_prompt_with_memory",
    "DIALOGUE_END_CHECK_PROMPT",
    "KNOWLEDGE_EXTRACT_PROMPT",
    "KNOWLEDGE_EXTRACT_PROMPT_INITIAL",
    "EVENT_SELECTION_PROMPT",
    "EVENT_SELECTION_TRAP_PRIORITY_PROMPT",
    "TRAP_EVENT_TYPES_SET",
    # Phased event generation
    "PHASE_NAMES",
    "PHASE_DESCRIPTIONS",
    "build_phased_event_prompt",
]
