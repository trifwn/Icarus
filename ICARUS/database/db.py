from __future__ import annotations

from typing import TYPE_CHECKING

from ICARUS import APPHOME

from .analysesDB import AnalysesDB
from .database2D import Database_2D
from .database3D import Database_3D

if TYPE_CHECKING:
    from ICARUS.airfoils.airfoil import Airfoil
    from ICARUS.vehicle.plane import Airplane


class Database:
    """Master Database Class Containing other Databases and managing them."""

    # Create only one instance of the database
    _instance = None

    def __new__(cls) -> Database:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializes the Database"""
        self.HOMEDIR: str = APPHOME
        self.foils_db: Database_2D = Database_2D()
        self.vehicles_db: Database_3D = Database_3D()
        self.analyses_db: AnalysesDB = AnalysesDB()

    def load_data(self) -> None:
        """Loads all the data from the databases"""
        self.foils_db.load_data()
        self.vehicles_db.load_data()

    def get_airfoil(self, name: str) -> Airfoil:
        return self.foils_db.get_airfoil(name)

    def get_vehicle(self, name: str) -> Airplane:
        return self.vehicles_db.get_vehicle(name)

    def __str__(self) -> str:
        return "Master Database"

    def inspect(self) -> None:
        """Prints the content of the database"""
        print("Master Database Contents:")
        print("")
        print("------------------------------------------------")
        print(f"|        {self.foils_db}                          |")
        print("------------------------------------------------")
        for foil in self.foils_db._raw_data.keys():
            string = f"|{foil}\t\t\t\t\t|\n"
            for solver in self.foils_db._raw_data[foil].keys():
                string += f"|  - {solver}:"
                reyns = list(self.foils_db._raw_data[foil][solver].keys())
                reyns_num = [float(reyn) for reyn in reyns]
                string += f"\t Re: {min(reyns_num)} - {max(reyns_num)} "
                string += "\t|\n"
            string += "|\t\t\t\t\t\t|\n|\t\t\t\t\t\t|"
            print(string)
        print("-----------------------------------------")
        print("")

        print("------------------------------------------------")
        print(f"|        {self.vehicles_db}             |")
        print("------------------------------------------------")

        for vehicle in self.vehicles_db.polars.keys():
            string = f"|{vehicle}\n"
            for solver in self.vehicles_db.polars[vehicle].keys():
                string += f"|\t - {solver}:"
                string += "\n"
            string += "|\n|"
            print(string)
        print("-----------------------------------------")
        print("")

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
