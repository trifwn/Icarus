"""
=============================================
ICARUS Database package
=============================================

.. toctree: generated/
    :hidden:

    ICARUS.Database.db
    ICARUS.Database.Database_2D
    ICARUS.Database.Database_3D
    ICARUS.Database.AnalysesDB
    ICARUS.Database.utils

.. module:: ICARUS.Database
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining the ICARUS Database.

.. currentmodule:: ICARUS.Database

This package contains class and routines for defining the ICARUS Database.
The ICARUS Database is used in order to have data and workflow persistency.
The package is divided in the following files:

.. autosummary::
    :toctree: generated/

    ICARUS.Database.db
    ICARUS.Database.Database_2D
    ICARUS.Database.Database_3D
    ICARUS.Database.AnalysesDB
    ICARUS.Database.utils

"""
import os

# MOCK CASES ###
# 2D
APPHOME: str = os.path.dirname(os.path.realpath(__file__))
APPHOME = os.path.abspath(os.path.join(APPHOME, os.pardir))
APPHOME = os.path.abspath(os.path.join(APPHOME, os.pardir))

BASEOPENFOAM: str = os.path.join(APPHOME, "Data", "Mock", "BaseOF")
BASEFOIL2W: str = os.path.join(APPHOME, "Data", "Mock", "BaseF2W")
# 3D
BASEGNVP3: str = os.path.join(APPHOME, "Data", "Mock", "BaseGNVP3")

# DATABASES ###
DB2D: str = os.path.join(APPHOME, "Data", "2D")
DB3D: str = os.path.join(APPHOME, "Data", "3D")
ANALYSESDB: str = os.path.join(APPHOME, "Data", "Analyses")
XFLRDB: str = os.path.join(APPHOME, "Data", "XFLR5")

from . import Database_2D
from . import Database_3D
from . import AnalysesDB
from . import db

__all__ = ["Database_2D", "Database_3D", "AnalysesDB", "db"]
