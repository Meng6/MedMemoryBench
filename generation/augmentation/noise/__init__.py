"""Noise data augmentation module.

Provides health knowledge chat noise session generation and injection
for testing agent memory robustness.
"""

from .generator import NoiseDialogueGenerator
from .injector import NoiseDataInjector
from .config import NoiseConfig

__all__ = [
    "NoiseDialogueGenerator",
    "NoiseDataInjector",
    "NoiseConfig",
]
