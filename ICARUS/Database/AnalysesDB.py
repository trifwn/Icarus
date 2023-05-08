import os

from . import ANALYSESDB
from . import APPHOME
from ICARUS.Core.struct import Struct


class AnalysesDB:
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.DATADIR = ANALYSESDB
        self.Data = Struct()

    def loadData(self):
        self.scan()

    def scan(self):
        os.chdir(ANALYSESDB)
        # LOAD ANALYSES
        os.chdir(self.HOMEDIR)
