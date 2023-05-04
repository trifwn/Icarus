import os

from . import APPHOME
from .AnalysesDB import AnalysesDB
from .Database_2D import Database_2D
from .Database_3D import Database_3D
from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Flight_Dynamics.dyn_plane import dyn_Airplane
from ICARUS.Vehicle.plane import Airplane


class DB:
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.foilsDB = Database_2D()
        self.vehiclesDB = Database_3D()
        self.analysesDB = AnalysesDB()

    def loadData(self):
        self.foilsDB.loadData()
        self.vehiclesDB.loadData()

    def __str__(self):
        return f"Database"

    def __enter__(self, obj):
        if isinstance(obj, Airplane):
            self.vehiclesDB.__enter__(obj)
        elif isinstance(obj, dyn_Airplane):
            self.vehiclesDB.__enter__(obj)
        elif isinstance(obj, AirfoilD):
            self.foilsDB.__enter__(obj)
        else:
            print(f"Object {obj} not supported")

    def __exit__(self):
        os.chdir(self.HOMEDIR)
