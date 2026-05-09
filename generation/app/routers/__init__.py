"""API routers."""
from .personas import router as personas_router
from .events import router as events_router
from .dialogues import router as dialogues_router
from .tasks import router as tasks_router

__all__ = [
    "personas_router",
    "events_router",
    "dialogues_router",
    "tasks_router",
]
