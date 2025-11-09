"""Terminal-Bench Agent with Memory Integration.

This agent implements the Terminal-Bench BaseAgent interface and uses the
memory system for code intelligence.
"""

__version__ = "0.1.0"

from .core import MemoryGuidedAgent

__all__ = ["MemoryGuidedAgent"]
