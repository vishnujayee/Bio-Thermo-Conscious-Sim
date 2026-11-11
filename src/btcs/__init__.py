"""Bio-Thermo-Conscious-Sim.

Digital organisms driven by LLM agents, where survival is bound to
non-equilibrium thermodynamics rather than to game rules.

See construction.md for the build plan and the physics specification.
"""

from btcs.config import Config, ConfigError
from btcs.events import EventLog, EventType, read_events
from btcs.rng import RngService

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "EventLog",
    "EventType",
    "RngService",
    "read_events",
    "__version__",
]
