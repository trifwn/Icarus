"""
ICARUS: A Python package for the analysis, modelling and design of aircraft.
"""
import multiprocessing
import os
import platform

platform_os = platform.system()
CPU_COUNT: int = multiprocessing.cpu_count()

if CPU_COUNT > 2:
    if platform_os == "Windows":
        CPU_TO_USE: int = CPU_COUNT // 2
    else:
        CPU_TO_USE = CPU_COUNT - 2
else:
    CPU_TO_USE = 1

APPHOME: str = os.path.dirname(os.path.realpath(__file__))
APPHOME = os.path.abspath(os.path.join(APPHOME, os.pardir))


from . import airfoils
from . import aerodynamics
from . import conceptual
from . import core
from . import database
from . import environment
from . import flight_dynamics
from . import mission
from . import vehicle
from . import visualization
from . import computation

__all__ = [
    "aerodynamics",
    "airfoils",
    "conceptual",
    "core",
    "database",
    "environment",
    "flight_dynamics",
    "mission",
    "vehicle",
    "visualization",
    "computation",
]

__version__ = "0.3.0"
