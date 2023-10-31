"""
============================================================
ICARUS Module for adding Conceptual Design Criteria
============================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Conceptual.Criteria.FAR
    ICARUS.Conceptual.Criteria.cruise

.. module:: ICARUS.Conceptual.Criteria
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining criteria for the conceptual design of flying vehicles.

.. currentmodule:: ICARUS.Conceptual.Criteria

This package contains class and routines for defining criteria for the conceptual design of flying vehicles. As of now the package
is not complete! The following criteria are implemented:

FAR
--------------------------

.. autosummary::
    :toctree:

    ICARUS.Conceptual.Criteria.FAR

Flight Segments
--------------------------

.. autosummary::
    :toctree:

    ICARUS.Conceptual.Criteria.cruise

"""
from . import cruise
from . import FAR

__all__ = ["FAR", "cruise"]
