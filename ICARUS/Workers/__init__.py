"""
===============================================
ICARUS Workers Module
===============================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.Workers.options
    ICARUS.Workers.analysis
    ICARUS.Workers.solver

.. module:: ICARUS.Workers
    :platform: Unix, Windows
    :synopsis: This package contains an abstraction layer for solvers and analyses.

.. currentmodule:: ICARUS.Workers

This package contains an abstraction layer for solvers and analyses. The package is divided in three modules:

.. autosummary::
    :toctree: generated/

    ICARUS.Workers.analysis - Analysis class definition
    ICARUS.Workers.solver - Solver class definition
    ICARUS.Workers.options - Option class definition


"""
from .analysis import Analysis
from .options import Option
from .solver import Solver

__all__ = ["Analysis", "Solver", "Option"]
