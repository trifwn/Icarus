from ICARUS.core.base_types import Struct


class AnalysesDB:
    def __init__(self, APPHOME: str, location: str) -> None:
        self.HOMEDIR: str = APPHOME
        self.ANALYSESDB: str = location
        self.data = Struct()

    def load_data(self) -> None:
        self.scan()

    def scan(self) -> None:
        # LOAD ANALYSES
        pass
