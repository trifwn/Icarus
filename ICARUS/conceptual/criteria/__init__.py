"""
============================================================
ICARUS Module for adding Conceptual Design Criteria
============================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.conceptual.criteria.FAR
    ICARUS.conceptual.criteria.cruise

.. module:: ICARUS.conceptual.criteria
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining criteria for the conceptual design of flying vehicles.

.. currentmodule:: ICARUS.conceptual.criteria

This package contains class and routines for defining criteria for the conceptual design of flying vehicles. As of now the package
is not complete! The following criteria are implemented:

FAR
--------------------------

.. autosummary::
    :toctree:

    ICARUS.conceptual.criteria.FAR

Flight Segments
--------------------------

.. autosummary::
    :toctree:

    ICARUS.conceptual.criteria.cruise

"""

from . import FAR
from . import cruise

__all__ = ["FAR", "cruise"]
