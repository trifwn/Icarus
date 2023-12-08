from .AnalysesDB import AnalysesDB
from .Database_2D import Database_2D
from .Database_3D import Database_3D
from ICARUS import APPHOME


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

    def inspect(self) -> None:
        """Prints the content of the database"""
        print("Master Database Contents:")
        print("")
        print("------------------------------------------------")
        print(f"|        {self.foils_db}                          |")
        print("------------------------------------------------")
        for foil in self.foils_db.data.keys():
            string = f"|{foil}\t\t\t\t\t|\n"
            for solver in self.foils_db.data[foil].keys():
                string += f"|  - {solver}:"
                reyns = list(self.foils_db.data[foil][solver].keys())
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

        for vehicle in self.vehicles_db.data.keys():
            string = f"|{vehicle}\n"
            for solver in self.vehicles_db.data[vehicle].keys():
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
