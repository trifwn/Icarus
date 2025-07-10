from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pandas import DataFrame
from rich import box
from rich.table import Table

from ICARUS import ICARUS_CONSOLE
from ICARUS.airfoils import AirfoilData
from ICARUS.airfoils import AirfoilPolarMap
from ICARUS.core.types import FloatArray

from .database2D import Database_2D
from .database3D import Database_3D

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


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

    def __init__(self, DB_PATH: str) -> None:
        """Initializes the Database
        Args:
            DB_PATH (str): The path to the database directory
        """
        DB_PATH = os.path.abspath(DB_PATH)
        self.DB_PATH: str = DB_PATH
        self.EXTERNAL_DB: str = os.path.join(DB_PATH, "3d_Party")
        DB2D: str = os.path.join(DB_PATH, "2D")
        DB3D: str = os.path.join(DB_PATH, "3D")

        self.foils_db: Database_2D = Database_2D(DB2D)
        self.vehicles_db: Database_3D = Database_3D(DB3D)

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

    def load_all_data(self) -> None:
        """Loads all the data from the databases"""
        self.foils_db.load_all_data()
        self.vehicles_db.load_all_data()

    ########## Airfoils Database ##########
    def get_airfoil(self, name: str) -> Airfoil:
        return self.foils_db.get_airfoil(name)

    def get_airfoil_polars(
        self,
        airfoil: str | Airfoil,
        solver: str | None = None,
    ) -> AirfoilPolarMap:
        return self.foils_db.get_polars(airfoil, solver=solver)

    def get_or_compute_airfoil_polars(
        self,
        airfoil: Airfoil,
        reynolds: float,
        aoa: list[float] | FloatArray,
        solver_name: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
        REYNOLDS_BINS: list[float] | FloatArray | None = None,
    ) -> AirfoilPolarMap:
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

    def load_airfoil_data(self, airfoil: Airfoil) -> None:
        self.foils_db.load_airfoil_data(airfoil)

    @staticmethod
    def generate_airfoil_directories(
        airfoil: Airfoil,
        reynolds: float | None = None,
        angles: float | list[float] | FloatArray = [],
    ) -> tuple[str, str, list[str]]:
        return Database_2D.generate_airfoil_directories(airfoil, reynolds, angles)

    @property
    def airfoils(self) -> dict[str, Airfoil]:
        return self.foils_db.airfoils

    @property
    def airfoil_polars(self) -> dict[str, AirfoilData]:
        return self.foils_db.polars

    ########## Vehicles Database ##########
    def load_vehicle(self, name: str, file: str) -> Airplane | None:
        return self.vehicles_db.load_vehicle(name, file)

    def get_vehicle(self, name: str) -> Airplane:
        return self.vehicles_db.get_vehicle(name)

    def get_vehicle_polars(
        self,
        vehicle: str | Airplane,
        solver: str | None = None,
    ) -> DataFrame:
        return self.vehicles_db.get_polars(vehicle)

    def get_vehicle_case_directory(
        self,
        airplane: Airplane,
        state: State,
        solver: str,
        case: str | None = None,
    ) -> str:
        return self.vehicles_db.get_case_directory(airplane, state, solver, case)

    def get_vehicle_names(self) -> list[str]:
        return self.vehicles_db.get_vehicle_names()

    def get_vehicle_states(self, vehicle: str | Airplane) -> dict[str, State]:
        return self.vehicles_db.get_states(vehicle)

    def load_vehicle_solver_data(
        self,
        vehicle: Airplane,
        state: State,
        folder: str,
        solver: str,
    ) -> None:
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
        """Prints the content of the airfoil and vehicle database using rich tables."""
        console = ICARUS_CONSOLE
        # === Airfoil Database ===
        console.rule(f"[bold cyan]Airfoil Database: {self.foils_db}")

        table = Table(title="Airfoils and Polars", box=box.SQUARE, expand=True)
        table.add_column("Airfoil", style="bold")
        table.add_column("Solver", style="cyan")
        table.add_column("Reynolds Range", justify="right")

        for foil_name, data in self.foils_db.polars.items():
            solvers = data.solvers
            for i, solver in enumerate(solvers):
                reynolds_list = list(data.get_solver_reynolds(solver))
                reynolds_float = [float(r) for r in reynolds_list]
                re_range = f"{min(reynolds_float):.1e} - {max(reynolds_float):.1e}"
                table.add_row(
                    foil_name if i == 0 else "",
                    solver,
                    re_range,
                )
        console.print(table)

        # === Vehicle Database ===
        console.rule(f"[bold green]Vehicle Database: {self.vehicles_db}")

        vehicle_table = Table(title="Vehicles and Solvers", box=box.SQUARE, expand=True)
        vehicle_table.add_column("Vehicle", style="bold")
        vehicle_table.add_column("Solver(s)", style="green")

        for vehicle_name, solver_dict in self.vehicles_db.polars.items():
            solvers = list(solver_dict.keys())
            solvers_str = ", ".join(solvers)
            vehicle_table.add_row(vehicle_name, solvers_str)

        console.print(vehicle_table)
