"""=============================================
ICARUS Database package
=============================================

.. toctree: generated/
    :hidden:

    ICARUS.database.db
    ICARUS.database.Database_2D
    ICARUS.database.Database_3D
    ICARUS.database.AnalysesDB
    ICARUS.database.utils

.. module:: ICARUS.database
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining the ICARUS Database.

.. currentmodule:: ICARUS.database

This package contains class and routines for defining the ICARUS Database.
The ICARUS Database is used in order to have data and workflow persistency.
The package is divided in the following files:

.. autosummary::
    :toctree: generated/

    ICARUS.database.db
    ICARUS.database.Database_2D
    ICARUS.database.Database_3D
    ICARUS.database.AnalysesDB
    ICARUS.database.utils

"""
# DATABASES ###

from .analysesDB import AnalysesDB
from .database2D import AirfoilNotFoundError
from .database2D import Database_2D
from .database2D import PolarsNotFoundError
from .database3D import Database_3D
from .db import Database
from .utils import angle_to_case
from .utils import case_to_angle
from .utils import case_to_disturbance
from .utils import disturbance_to_case

__all__ = [
    "Database",
    "Database_2D",
    "Database_3D",
    "AnalysesDB",
    # Utils
    "angle_to_case",
    "case_to_angle",
    "disturbance_to_case",
    "case_to_disturbance",
    # Exceptions
    "PolarsNotFoundError",
    "AirfoilNotFoundError",
]
