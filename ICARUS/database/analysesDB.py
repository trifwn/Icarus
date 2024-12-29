import os

from ICARUS.core.struct import Struct


class AnalysesDB:
    def __init__(self, APPHOME: str, location: str) -> None:
        self.HOMEDIR: str = APPHOME
        self.DBANALYSES: str = location
        self.data = Struct()

    def load_data(self) -> None:
        self.scan()

    def scan(self) -> None:
        os.chdir(self.DBANALYSES)
        # LOAD ANALYSES
        os.chdir(self.HOMEDIR)
