import os

from . import ANALYSESDB
from ICARUS import APPHOME
from ICARUS.Core.struct import Struct


class AnalysesDB:
    def __init__(self) -> None:
        self.HOMEDIR: str = APPHOME
        self.DATADIR: str = ANALYSESDB
        self.data = Struct()

    def load_data(self) -> None:
        self.scan()

    def scan(self) -> None:
        os.chdir(ANALYSESDB)
        # LOAD ANALYSES
        os.chdir(self.HOMEDIR)
