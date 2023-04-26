from ICARUS.Core.struct import Struct
from . import ANALYSESDB, APPHOME
import os

class AnalysesDB():
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.Data = Struct()

    def loadData(self):
        self.scan()
        
    def scan(self):
        os.chdir(ANALYSESDB)
        ## LOAD ANALYSES
        os.chdir(self.HOMEDIR)
