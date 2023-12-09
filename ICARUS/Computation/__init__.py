"""
===============================================
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

.. autosummary::
    :toctree: generated/

    ICARUS.Computation.analysis - Analysis class definition
    ICARUS.Computation.Solvers.solver - Solver class definition
    ICARUS.Computation.options - Option class definition


"""
from . import Analyses
from . import Results
from . import Solvers
from . import Workflow

__all__ = ['Solvers', 'Analyses', 'Workflow', 'Results']
