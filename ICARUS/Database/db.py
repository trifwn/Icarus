from .Database_2D import Database_2D
from .Database_3D import Database_3D

class DB():
    def __init__(self):
        self.foilsDB = Database_2D()
        self.vehiclesDB = Database_3D()
    
    def loadData(self):
        self.foilsDB.loadData()
        self.vehiclesDB.loadData()