"""
=============================================
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
    # F2W_exe = os.path.join(APPHOME, "bin", "f2w")
    Foil_Section_exe = os.path.join(APPHOME, "bin", "foil_section")
    AVL_exe = os.path.join(APPHOME, "bin", "avl")

    # Check if the files have execution permission
    if not os.access(GenuVP3_exe, os.X_OK):
        os.chmod(GenuVP3_exe, 0o755)

    if not os.access(GenuVP7_exe, os.X_OK):
        os.chmod(GenuVP7_exe, 0o755)

    # if not os.access(F2W_exe, os.X_OK):
    # os.chmod(F2W_exe, 0o755)

    if not os.access(Foil_Section_exe, os.X_OK):
        os.chmod(Foil_Section_exe, 0o755)

    if not os.access(AVL_exe, os.X_OK):
        os.chmod(AVL_exe, 0o755)

# DATABASES ###
DB2D: str = os.path.join(APPHOME, "Data", "2D")
DB3D: str = os.path.join(APPHOME, "Data", "3D")
ANALYSESDB: str = os.path.join(APPHOME, "Data", "Analyses")
EXTERNAL_DB: str = os.path.join(APPHOME, "Data", "3d_Party")

from . import analysesDB
from . import database2D
from . import database3D
from . import db

__all__ = ["database2D", "database3D", "analysesDB", "db"]

from .db import Database

cwd = os.getcwd()
DB = Database()
# DB.load_data()
# DB.inspect()
os.chdir(cwd)
