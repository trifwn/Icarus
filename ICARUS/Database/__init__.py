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

from ICARUS import APPHOME
from ICARUS import platform_os

### SOLVER EXECUTABLES ###

if platform_os == "Windows":
    GenuVP3_exe: str = os.path.join(APPHOME, "bin", "gnvp3.exe")
    GenuVP7_exe: str = os.path.join(APPHOME, "bin", "gnvp7.exe")
    F2W_exe: str = os.path.join(APPHOME, "bin", "f2w.exe")
    Foil_Section_exe: str = os.path.join(APPHOME, "bin", "foil_section.exe")
    AVL_exe: str = os.path.join(APPHOME, "bin", "avl.exe")
elif platform_os == "Linux":
    GenuVP3_exe = os.path.join(APPHOME, "bin", "gnvp3")
    GenuVP7_exe = os.path.join(APPHOME, "bin", "gnvp7")
    F2W_exe = os.path.join(APPHOME, "bin", "f2w")
    Foil_Section_exe = os.path.join(APPHOME, "bin", "foil_section")
    AVL_exe = os.path.join(APPHOME, "bin", "avl.exe")

# DATABASES ###
DB2D: str = os.path.join(APPHOME, "Data", "2D")
DB3D: str = os.path.join(APPHOME, "Data", "3D")
ANALYSESDB: str = os.path.join(APPHOME, "Data", "Analyses")
EXTERNAL_DB: str = os.path.join(APPHOME, "Data", "3d_Party")

from . import Database_2D
from . import Database_3D
from . import AnalysesDB
from . import db

__all__ = ["Database_2D", "Database_3D", "AnalysesDB", "db"]

from .db import Database

cwd = os.getcwd()
DB = Database()
DB.load_data()
# DB.inspect()
os.chdir(cwd)
