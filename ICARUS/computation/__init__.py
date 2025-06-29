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

from .monitors.progress import RichProgressMonitor

from .execution import (
    create_execution_engine,
    AbstractExecutionEngine,
    AsyncExecutionEngine,
    MultiprocessingExecutionEngine,
    SequentialExecutionEngine,
    ThreadingExecutionEngine,
    AdaptiveExecutionEngine,
)

from .resources.manager import SimpleResourceManager

from .runners import SimulationRunner
from .observers import ConsoleProgressObserver


from . import analyses
from . import solvers

__version__ = "2.0.0"
__author__ = "Enhanced OOP Framework Team"

__all__ = [
    # Core types
    "core",
    # Monitoring
    "RichProgressMonitor",
    # Execution engines
    "AbstractExecutionEngine",
    "AsyncExecutionEngine",
    "MultiprocessingExecutionEngine",
    "SequentialExecutionEngine",
    "ThreadingExecutionEngine",
    "AdaptiveExecutionEngine",
    "create_execution_engine",
    # Resource management
    "SimpleResourceManager",
    # Main runner
    "SimulationRunner",
    # Examples
    "ConsoleProgressObserver",
    # Submodules
    "analyses",
    "solvers",
]
