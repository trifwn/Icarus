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

from .controller import Controlller
from .variable import ControllerVariable
from .variable import ControlVariableType

__all__ = [
    "ControllerVariable",
    "ControlVariableType",
    "Controlller",
]
