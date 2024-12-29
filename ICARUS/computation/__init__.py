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

.. autosummary::
    :toctree: generated/

    ICARUS.Computation.analysis - Analysis class definition
    ICARUS.Computation.Solvers.solver - Solver class definition
    ICARUS.Computation.options - Option class definition


"""

from . import analyses
from . import results
from . import solvers
from . import workflow

__all__ = ["analyses", "results", "solvers", "workflow"]
