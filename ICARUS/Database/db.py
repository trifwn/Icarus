from .Database_2D import Database_2D
from .Database_3D import Database_3D
from .AnalysesDB import AnalysesDB
from . import APPHOME

class DB():
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