"""
ICARUS: A Python package for the analysis, modelling and design of aircraft.
isort:skip_file
"""

__version__ = "0.4.0"

from .logging import (
    ICARUS_CONSOLE,
    setup_logging,
    setup_mp_logging,
)
from .config import (
    PLATFORM,
    CPU_COUNT,
    CPU_TO_USE,
    HAS_GPU,
    HAS_JAX,
    INSTALL_DIR,
    # Executables
    GenuVP3_exe,
    GenuVP7_exe,
    Foil_exe,
    AVL_exe,
    MAX_FLOAT,
    MAX_INT,
)
from . import core
from . import airfoils
from . import control
from . import vehicle
from . import environment
from . import flight_dynamics
from . import conceptual
from . import database
from . import visualization
from . import aero
from . import computation
from . import mission

__all__ = [
    # Logging
    "ICARUS_CONSOLE",
    "setup_logging",
    "setup_mp_logging",
    # Configuration
    "PLATFORM",
    "CPU_COUNT",
    "CPU_TO_USE",
    "HAS_GPU",
    "HAS_JAX",
    "INSTALL_DIR",
    "MAX_FLOAT",
    "MAX_INT",
    # Executables
    "GenuVP3_exe",
    "GenuVP7_exe",
    "Foil_exe",
    "AVL_exe",
    # Module imports
    "core",
    "control",
    "aero",
    "airfoils",
    "computation",
    "conceptual",
    "database",
    "environment",
    "flight_dynamics",
    "mission",
    "vehicle",
    "visualization",
]
