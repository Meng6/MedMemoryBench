"""Pipeline configuration module."""
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 16000


@dataclass
class LLMConfig:
    """LLM hyperparameters."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class PersonaConfig:
    """Persona generation config."""

    count: Optional[int] = None
    persona_ids: Optional[list[int]] = None
    skip_existing: bool = True
    concurrency: int = 5
    max_retries: int = 3
    export_path: str = "data/generated_personas.json"
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class EventConfig:
    """Event graph generation config.

    Two-phase generation:
    1. Generate 6 types of trap events (generate-trap-events)
    2. Generate regular events by clinical phases (generate-regular-events)
    """

    persona_ids: Optional[list[int]] = None
    skip_existing: bool = True
    start_date: str = "2024-01-01"
    time_span_days: int = 365
    concurrency: int = 3
    export_path: str = "data/generated_events.json"
    trap_events_export_path: str = "data/generated_trap_events.json"
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Phased generation
    events_per_phase: int = 20
    max_total_events: int = 100

    @property
    def batch_size(self) -> int:
        return self.events_per_phase


@dataclass
class DialogueConfig:
    """Dialogue generation config."""

    persona_ids: Optional[list[int]] = None
    sessions_per_persona: int = 5
    max_turns: int = 10
    allow_natural_end: bool = True
    skip_existing: bool = False
    concurrency: int = 2
    export_path: str = "data/generated_dialogues.json"
    export_verbose: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Trap event coverage
    ensure_trap_coverage: bool = True


@dataclass
class QueryConfig:
    """Query generation config."""

    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class GenerationConfig:
    """Pipeline master config."""

    persona: PersonaConfig = field(default_factory=PersonaConfig)
    event: EventConfig = field(default_factory=EventConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)

    continue_on_error: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Post-init validation."""
        if not isinstance(self.persona, PersonaConfig):
            self.persona = PersonaConfig()
        if not isinstance(self.event, EventConfig):
            self.event = EventConfig()
        if not isinstance(self.dialogue, DialogueConfig):
            self.dialogue = DialogueConfig()


QUICK_TEST_CONFIG = GenerationConfig(
    persona=PersonaConfig(
        persona_ids=[1, 2, 3],
        concurrency=2,
        export_path="data/test_personas.json",
    ),
    event=EventConfig(
        persona_ids=[1, 2, 3],
        time_span_days=30,
        events_per_phase=5,
        max_total_events=20,
        concurrency=2,
        export_path="data/test_events.json",
    ),
    dialogue=DialogueConfig(
        persona_ids=[1, 2, 3],
        sessions_per_persona=2,
        max_turns=5,
        concurrency=1,
        export_path="data/test_dialogues.json",
    ),
)

PRODUCTION_CONFIG = GenerationConfig(
    persona=PersonaConfig(
        count=40,
        concurrency=10,
        export_path="data/production_personas.json",
    ),
    event=EventConfig(
        persona_ids=None,
        time_span_days=365,
        events_per_phase=30,
        max_total_events=150,
        concurrency=5,
        export_path="data/production_events.json",
    ),
    dialogue=DialogueConfig(
        persona_ids=None,
        sessions_per_persona=10,
        max_turns=12,
        concurrency=3,
        export_path="data/production_dialogues.json",
    ),
)
