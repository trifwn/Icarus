"""
=====================================
ICARUS Solvers Package
=====================================

.. toctree: generated/
    :hidden:

    ICARUS.Solvers.Airplane
    ICARUS.Solvers.Airfoil

.. module:: ICARUS.Solvers
    :platform: Unix, Windows
    :synopsis: This package contains object definitions for different solvers that can be used in ICARUS.

.. currentmodule:: ICARUS.Solvers

This package contains object definitions for different solvers that can be used in ICARUS. The package is divided in the following files:
The solvers are divided in two categories: Airplane and Airfoil solvers.

Airplane Solvers
------------------

.. autosummary::
    :toctree: generated/

    ICARUS.Solvers.Airplane

Airfoil Solvers
------------------

.. autosummary::
    :toctree: generated/

    ICARUS.Solvers.Airfoil


"""
from . import Airfoil
from . import Airplane

__all__ = ["Airplane", "Airfoil"]
