"""Deciding machinery. Swappable, so the physics never knows what is thinking."""

from btcs.brains.base import Brain, TurnBridge
from btcs.brains.mock import MockBrain

__all__ = ["Brain", "MockBrain", "TurnBridge"]

# LLMBrain and LocalBrain are deliberately not imported here: one needs the
# anthropic SDK, the other a running Ollama server, and neither should be a
# condition of importing this package.
