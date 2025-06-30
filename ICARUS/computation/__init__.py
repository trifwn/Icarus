"""===============================================
ICARUS Computation Module
===============================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.Computation.options
    ICARUS.Computation.analysis
    ICARUS.Computation.Solvers.solver

.. module:: ICARUS.Computation
    :platform: Unix, Windows
    :synopsis: This package contains an abstraction layer for solvers and analyses.

.. currentmodule:: ICARUS.Computation

This package contains an abstraction layer for solvers and analyses. The package is divided in three modules:

OOP Computation Framework
A comprehensive simulation framework featuring:
- Type safety and dependency injection
- Plugin architecture with advanced monitoring
- Resource management and fault tolerance
- Comprehensive lifecycle management
- Visual progress bars with tqdm integration

This package provides a complete solution for running complex simulations
with professional-grade monitoring, error handling, and progress tracking.

.. autosummary::
    :toctree: generated/

    ICARUS.Computation.analyses - Analyses module
    ICARUS.Computation.solvers - Solvers module

isort:skip_file


"""

from . import core
from .engines import (
    create_execution_engine,
    AbstractEngine,
    AsyncEngine,
    MultiprocessingEngine,
    SequentialExecutionEngine,
    ThreadingEngine,
    AdaptiveEngine,
)

from .resources.manager import SimpleResourceManager
from .monitors.rich_progress import RichProgressMonitor
from .observers import ConsoleProgressObserver
from .reporters import Reporter
from .runners import SimulationRunner

from .solver_parameters import (
    SolverParameters,
    NoSolverParameters,
    # Specific parameter types
    Parameter,
    BoolParameter,
    FloatParameter,
    IntParameter,
    IntOrNoneParameter,
    StrParameter,
)

from . import analyses

from .base_solver import Solver


__version__ = "2.0.0"
__author__ = "Enhanced OOP Framework Team"

__all__ = [
    # Core types
    "core",
    # Execution engines
    "AbstractEngine",
    "AsyncEngine",
    "MultiprocessingEngine",
    "SequentialExecutionEngine",
    "ThreadingEngine",
    "AdaptiveEngine",
    "create_execution_engine",
    # Resource management
    "SimpleResourceManager",
    # Main runner
    "SimulationRunner",
    # Reporters
    "Reporter",
    # Observers
    "ConsoleProgressObserver",
    # Monitors
    "RichProgressMonitor",
    # Solver Parameters
    "SolverParameters",
    "NoSolverParameters",
    # Specific solver parameter types
    "Parameter",
    "BoolParameter",
    "FloatParameter",
    "IntParameter",
    "IntOrNoneParameter",
    "StrParameter",
    # Analyses
    "analyses",
    # Base Solver
    "Solver",
]
