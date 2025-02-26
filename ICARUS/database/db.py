from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pandas import DataFrame

from ICARUS.airfoils.airfoil_polars import AirfoilData
from ICARUS.airfoils.airfoil_polars import AirfoilPolars
from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.flight_dynamics.state import State

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

    def __new__(cls, *args: Any, **kwargs: Any) -> Database:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Database:
        if cls._instance is None:
            raise ValueError("Database not initialized")
        return cls._instance

    def __init__(self, APPHOME: str) -> None:
        """Initializes the Database
        Args:
            APPHOME (str): The path to the database directory
        """
        APPHOME = os.path.abspath(APPHOME)
        self.HOMEDIR: str = APPHOME
        self.EXTERNAL_DB: str = os.path.join(APPHOME, "3d_Party")
        DB2D: str = os.path.join(APPHOME, "2D")
        DB3D: str = os.path.join(APPHOME, "3D")
        ANALYSESDB: str = os.path.join(APPHOME, "Analyses")

        self.foils_db: Database_2D = Database_2D(APPHOME, DB2D)
        self.vehicles_db: Database_3D = Database_3D(APPHOME, DB3D)
        self.analyses_db: AnalysesDB = AnalysesDB(APPHOME, ANALYSESDB)

    @property
    def DB2D(self) -> str:
        return self.foils_db.DB2D

    @DB2D.setter
    def DB2D(self, value: str) -> None:
        self.foils_db.DB2D = value

    @property
    def DB3D(self) -> str:
        return self.vehicles_db.DB3D

    @DB3D.setter
    def DB3D(self, value: str) -> None:
        self.vehicles_db.DB3D = value

    @property
    def ANALYSESDB(self) -> str:
        return self.analyses_db.ANALYSESDB

    @ANALYSESDB.setter
    def ANALYSESDB(self, value: str) -> None:
        self.analyses_db.ANALYSESDB = value

    def load_all_data(self) -> None:
        """Loads all the data from the databases"""
        self.foils_db.load_all_data()
        self.vehicles_db.load_all_data()

    ########## Airfoils Database ##########
    def get_airfoil(self, name: str) -> Airfoil:
        return self.foils_db.get_airfoil(name)

    def get_airfoil_polars(self, airfoil: str | Airfoil, solver: str | None = None) -> AirfoilPolars:
        return self.foils_db.get_polars(airfoil, solver=solver)

    def get_or_compute_airfoil_polars(
        self,
        airfoil: Airfoil,
        reynolds: float,
        aoa: list[float] | FloatArray,
        solver_name: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
        REYNOLDS_BINS: list[float] | FloatArray | None = None,
    ) -> AirfoilPolars:
        return self.foils_db.get_or_compute_polars(
            airfoil=airfoil,
            reynolds=reynolds,
            solver_name=solver_name,
            aoa=aoa,
            REYNOLDS_BINS=REYNOLDS_BINS,
        )

    def get_airfoil_data(self, airfoil: str | Airfoil) -> AirfoilData:
        return self.foils_db.get_airfoil_data(airfoil)

    def get_airfoil_names(self) -> list[str]:
        return self.foils_db.get_airfoil_names()

    def load_airfoil_data(self, airfoil: str | Airfoil) -> None:
        self.foils_db.load_airfoil_data(airfoil)

    @staticmethod
    def generate_airfoil_directories(
        airfoil: Airfoil,
        reynolds: float,
        angles: list[float] | FloatArray,
    ) -> tuple[str, str, str, list[str]]:
        return Database_2D.generate_airfoil_directories(airfoil, reynolds, angles)

    @property
    def airfoils(self) -> Struct:
        return self.foils_db.airfoils

    @property
    def airfoil_polars(self) -> dict[str, AirfoilData]:
        return self.foils_db.polars

    ########## Vehicles Database ##########
    def load_vehicle(self, name: str, file: str) -> Airplane | None:
        return self.vehicles_db.load_vehicle(name, file)

    def get_vehicle(self, name: str) -> Airplane:
        return self.vehicles_db.get_vehicle(name)

    def get_vehicle_polars(self, vehicle: str | Airplane, solver: str | None = None) -> DataFrame:
        return self.vehicles_db.get_polars(vehicle)

    def get_vehicle_case_directory(self, airplane: Airplane, state: State, solver: str, case: str | None = None) -> str:
        return self.vehicles_db.get_case_directory(airplane, state, solver, case)

    def get_vehicle_names(self) -> list[str]:
        return self.vehicles_db.get_vehicle_names()

    def get_vehicle_states(self, vehicle: str | Airplane) -> dict[str, State]:
        return self.vehicles_db.get_states(vehicle)

    def load_vehicle_solver_data(self, vehicle: Airplane, state: State, folder: str, solver: str) -> None:
        self.vehicles_db.load_solver_data(
            vehicle=vehicle,
            state=state,
            folder=folder,
            solver=solver,
        )

    ########## UTILS ##########
    def __str__(self) -> str:
        return "Master Database"

    def inspect(self) -> None:
        """Prints the content of the database"""
        print("Master Database Contents:")
        print()
        print("------------------------------------------------")
        print(f"|        {self.foils_db}                          |")
        print("------------------------------------------------")
        for foil in self.foils_db.polars.keys():
            string = f"|{foil}\t\t\t\t\t|\n"
            for solver in self.foils_db.polars[foil].solvers:
                string += f"|  - {solver}:"
                reyns = list(self.foils_db.polars[foil].get_solver_reynolds(solver))
                reyns_num = [float(reyn) for reyn in reyns]
                string += f"\t Re: {min(reyns_num)} - {max(reyns_num)} "
                string += "\t|\n"
            string += "|\t\t\t\t\t\t|\n|\t\t\t\t\t\t|"
            print(string)
        print("-----------------------------------------")
        print()

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
        print()

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
