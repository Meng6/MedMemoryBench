import importlib.metadata

__version__ = importlib.metadata.version("mem0ai")

from methods.mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from methods.mem0.memory.main import Memory  # noqa
