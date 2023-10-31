"""
ICARUS: A Python package for the analysis, modelling and design of aircraft.
"""
from . import Aerodynamics
from . import Airfoils
from . import Conceptual
from . import Control
from . import Core
from . import Database
from . import Environment
from . import Flight_Dynamics
from . import Input_Output
from . import Mission
from . import Solvers
from . import Vehicle
from . import Visualization
from . import Workers

__all__ = [
    "Aerodynamics",
    "Airfoils",
    "Conceptual",
    "Control",
    "Core",
    "Database",
    "Environment",
    "Flight_Dynamics",
    "Input_Output",
    "Mission",
    "Solvers",
    "Vehicle",
    "Visualization",
    "Workers",
]

__version__ = "0.3.0"
