"""
ICARUS: A Python package for the analysis, modelling and design of aircraft.
isort:skip_file
"""

__version__ = "0.3.0"


from .settings import (
    PLATFORM,
    CPU_COUNT,
    CPU_TO_USE,
    HAS_GPU,
    HAS_JAX,
    INSTALL_DIR,
    # Executables
    GenuVP3_exe,
    GenuVP7_exe,
    F2W_exe,
    Foil_Section_exe,
    AVL_exe,
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
    # Settings
    "PLATFORM",
    "CPU_COUNT",
    "CPU_TO_USE",
    "HAS_GPU",
    "HAS_JAX",
    "INSTALL_DIR",
    # Executables
    "GenuVP3_exe",
    "GenuVP7_exe",
    "F2W_exe",
    "Foil_Section_exe",
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
