"""Data generation pipeline system.

Provides three independent generation modes:
1. Generate personas (generate_personas)
2. Generate event graphs (generate_events)
3. Generate dialogue interactions (generate_dialogues)
"""

from .config import (
    GenerationConfig,
    PersonaConfig,
    EventConfig,
    DialogueConfig,
    QUICK_TEST_CONFIG,
    PRODUCTION_CONFIG,
)
from .generator import DataGenerator
from .exporters import (
    PersonaExporter,
    EventExporter,
    DialogueExporter,
)

__version__ = "1.0.0"

__all__ = [
    "DataGenerator",
    "GenerationConfig",
    "PersonaConfig",
    "EventConfig",
    "DialogueConfig",
    "QUICK_TEST_CONFIG",
    "PRODUCTION_CONFIG",
    "PersonaExporter",
    "EventExporter",
    "DialogueExporter",
]
