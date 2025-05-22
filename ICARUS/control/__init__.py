"""===============================
ICARUS Control Module
===============================

.. toctree: generated/
    :hidden:

.. module:: ICARUS.Control
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining and analyzing control of different vehicles.

.. currentmodule:: ICARUS.Control

This package contains class and routines for defining and analyzing control of different vehicles.
As of now the package is empty! Work in progress!


"""

from .variable import ControllerVariable, ControlVariableType
from .controller import Controlller

__all__ = [
    "ControllerVariable",
    "ControlVariableType",
    "Controlller",
]
