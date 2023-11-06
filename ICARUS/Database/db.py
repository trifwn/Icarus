from . import APPHOME
from .AnalysesDB import AnalysesDB
from .Database_2D import Database_2D
from .Database_3D import Database_3D


class Database:
    """Master Database Class Containing other Databases and managing them."""

    def __init__(self) -> None:
        self.HOMEDIR: str = APPHOME
        self.foils_db: Database_2D = Database_2D()
        self.vehicles_db: Database_3D = Database_3D()
        self.analyses_db: AnalysesDB = AnalysesDB()

    def load_data(self) -> None:
        """Loads all the data from the databases"""
        self.foils_db.load_data()
        self.vehicles_db.load_data()

    def __str__(self) -> str:
        return "Master Database"

    # def __enter__(self, obj):
    #     if isinstance(obj, Airplane):
    #         self.vehiclesDB.__enter__(obj)
    #     elif isinstance(obj, dyn_Airplane):
    #         self.vehiclesDB.__enter__(obj)
    #     elif isinstance(obj, Airfoil):
    #         self.foilsDB.__enter__(obj)
    #     else:
    #         print(f"Object {obj} not supported")

    # def __exit__(self):
    #     os.chdir(self.HOMEDIR)
