from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import pandas as pd
from pandas import DataFrame

from ICARUS import APPHOME
from ICARUS.computation.solvers.GenuVP.post_process.convergence import (
    get_error_convergence,
)
from ICARUS.computation.solvers.GenuVP.post_process.convergence import (
    get_loads_convergence,
)
from ICARUS.core.struct import Struct
from ICARUS.flight_dynamics.state import State

from . import DB3D

if TYPE_CHECKING:
    from ICARUS.vehicle.plane import Airplane

jsonpickle_pd.register_handlers()


class Database_3D:
    """Class to represent the 3D Database. It contains all the information and results
    of the 3D analysis of the vehicles."""

    def __init__(self) -> None:
        self.HOMEDIR: str = APPHOME
        self.DATADIR: str = DB3D
        self.forces: Struct = Struct()
        self.polars: Struct = Struct()
        self.planes: Struct = Struct()
        self.states: Struct = Struct()
        self.convergence_data: Struct = Struct()

    def get_planenames(self) -> list[str]:
        """
        Returns the list of planenames in the database.

        Returns:
            list[str]: List of planenames
        """
        return list(self.planes.keys())

    def load_data(self) -> None:
        self.read_all_data()

    def get_state(self, vehicle: str, state: str) -> State:
        if vehicle not in self.states.keys():
            # Try to Load Vehicle object
            self.get_vehicle(vehicle)
            self.states[vehicle] = {}

        if state in self.states[vehicle].keys():
            state_obj: State = self.states[vehicle][state]
            return state_obj
        else:
            try:
                self.read_plane_data(vehicle)
                state_obj = self.states[vehicle][state]
                return state_obj
            except KeyError:
                raise ValueError(f"No State found for {state}")

    def get_polars(self, name: str) -> DataFrame:
        if name in self.polars.keys():
            polar_obj: DataFrame = self.polars[name]
            return polar_obj
        else:
            self.read_plane_data(name)
            try:
                polar_obj = self.polars[name]
                return polar_obj
            except KeyError:
                raise ValueError(f"No Polars found for {name}")

    def get_vehicle(self, name: str) -> Airplane:
        if name in self.planes.keys():
            plane_object: Airplane = self.planes[name]
            return plane_object
        else:
            # Try to Load Vehicle object
            file_plane: str = os.path.join(DB3D, name, f"{name}.json")
            plane_obj: Airplane | None = self.load_plane(name, file_plane)
            print(f"Loaded Plane {plane_obj}")
            if plane_obj is None:
                raise ValueError(f"No Vehicle Object Found at {file_plane}")

            self.planes[name] = plane_obj
            return plane_obj

    def read_all_data(self) -> None:
        if not os.path.isdir(DB3D):
            print(f"Creating DB3D directory at {DB3D}...")
            os.makedirs(DB3D, exist_ok=True)

        vehicle_folders: list[str] = next(os.walk(DB3D))[1]
        for vehicle_folder in vehicle_folders:  # For each plane vehicle == folder name
            self.read_plane_data(vehicle_folder)

    def read_plane_data(self, vehicle_folder: str) -> None:
        logging.info(f"Adding Vehicle at {vehicle_folder}")
        # Enter DB3D
        DIRNOW = os.getcwd()
        os.chdir(DB3D)
        vehicle_folder_path = os.path.join(DB3D, vehicle_folder)
        os.chdir(vehicle_folder_path)

        # Load Vehicle object
        file_plane: str = os.path.join(DB3D, vehicle_folder, f"{vehicle_folder}.json")
        plane_obj: Airplane | None = self.load_plane(name=vehicle_folder, file=file_plane)
        if plane_obj is None:
            vehicle_name = vehicle_folder
            logging.debug(f"No Plane Object Found at {vehicle_folder_path}")
        else:
            # print(f"Loaded Plane {plane_obj}")
            # plane_obj.visualize()
            self.planes[plane_obj.name] = plane_obj
            vehicle_name = plane_obj.name

        # Load Vehicle State
        state_obj: State | None = self.load_plane_state(vehicle_folder_path)
        if state_obj is None:
            logging.debug(f"No State Object Found at {vehicle_folder_path}")
        else:
            try:
                self.states[vehicle_name][state_obj.name] = state_obj
            except KeyError:
                self.states[vehicle_name] = {}
                self.states[vehicle_name][state_obj.name] = state_obj

        solver_folders = next(os.walk(os.path.join(DB3D, vehicle_folder)))[1]
        for solver_folder in solver_folders:
            logging.debug(f"Entering {solver_folder}")
            if solver_folder == "GenuVP3":
                self.load_gnvp_data(
                    plane=plane_obj,
                    state=state_obj,
                    vehicle_folder=vehicle_folder,
                    gnvp_version=3,
                )
            elif solver_folder == "GenuVP7":
                self.load_gnvp_data(
                    plane=plane_obj,
                    state=state_obj,
                    vehicle_folder=vehicle_folder,
                    gnvp_version=7,
                )
            elif solver_folder == "LSPT":
                self.load_lspt_data(plane=plane_obj, state=state_obj, vehicle_folder=vehicle_folder)
            elif solver_folder == "AVL":
                self.load_avl_data(plane=plane_obj, state=state_obj, vehicle_folder=vehicle_folder)
            # elif solver_folder == "XFLR5":
            #     self.load_xflr5_data(vehicle_name, gnvp_version=3)
            else:
                logging.debug(f"Unknow Solver directory {solver_folder}")
                pass

            os.chdir(vehicle_folder_path)

        os.chdir(DIRNOW)

    def load_plane(self, name: str, file: str) -> Airplane | None:
        """Function to get Plane Object from file and decode it.

        Args:
            name (str): planename
            file (str): filename

        Returns:
            bool : whether the plane was found or not
        """
        plane = None
        try:
            with open(file, encoding="UTF-8") as f:
                json_obj: str = f.read()
                try:
                    plane: Airplane | None = jsonpickle.decode(json_obj)  # type: ignore
                    if plane is not None:
                        self.planes[plane.name] = plane
                    else:
                        raise ValueError(f"Error Decoding Plane object {name}")
                except Exception as error:
                    logging.debug(f"Error decoding Plane object {name}! Got error {error}")
                    print(f"Error decoding Plane object {name}! Got error {error}")
        except FileNotFoundError:
            logging.debug(f"FileNotFound No Plane object found in {name} folder at {file}!")
        return plane

    def load_plane_state(self, directory: str) -> State | None:
        """
        Function to load the states of the plane from the states.json file.

        Args:
            plane (str): Plane Name
            case (str): Case Directory
        """
        files: list[str] = next(os.walk(directory))[2]
        state: State | None = None
        for file in files:
            if file.endswith("_state.json"):
                file_path = os.path.join(directory, file)
                with open(file_path, encoding="UTF-8") as f:
                    json_obj: str = f.read()

                obj: Any = jsonpickle.decode(json_obj)
                if isinstance(obj, State):
                    state = obj
                else:
                    raise TypeError(f"Expected State object, got {type(obj)}")
        return state

    def load_gnvp_data(
        self,
        plane: Airplane | None,
        state: State | None,
        vehicle_folder: str,
        gnvp_version: int,
    ) -> None:
        genudir = os.path.join(DB3D, vehicle_folder, f"GenuVP{gnvp_version}")
        os.chdir(genudir)
        cases: list[str] = next(os.walk("."))[1]

        if plane is None:
            vehicle_name = vehicle_folder
        else:
            vehicle_name = plane.name

        # Load Forces from forces file and store them in the raw_data dict.
        # If the file doesn't exist it tries to create it by loading the plane object
        # and running the make_polars function. If that fails as weall it logs an error.
        forces_file: str = os.path.join("..", f"forces.gnvp{gnvp_version}")
        try:
            forces_df = pd.read_csv(forces_file)
            forces_df = forces_df.sort_values("AoA").reset_index(drop=True)
            self.add_forces(vehicle_name, forces_df)
            for name in [
                f"GenuVP{gnvp_version} Potential",
                f"GenuVP{gnvp_version} 2D",
                f"GenuVP{gnvp_version} ONERA",
            ]:
                self.add_polars_from_forces(plane=plane, state=state, forces=forces_df, prefix=name)
        except FileNotFoundError:
            logging.debug(
                f"No forces.gnvp{gnvp_version} file found in {vehicle_folder} folder at {DB3D}!\nNo polars Created as well",
            )

        for case in cases:
            # Load States
            if case == "Dynamics":
                continue

            # Loads the convergence data from gnvp.out and LOADS_aer.dat and stores it in the
            # convergence_data dict. If LOADS_aer.dat exists it tries to load it and then load
            # the convergence data from gnvp.out. If successfull it adds the error data to the
            # dataframe containing the loads and stores it in the convergence_data dict.
            RESULTS_DIR = os.path.join(DB3D, vehicle_folder, f"GenuVP{gnvp_version}", case)
            loads_file: str = os.path.join(RESULTS_DIR, "LOADS_aer.dat")
            log_file = os.path.join(RESULTS_DIR, f"gnvp{gnvp_version}.out")

            # Load Convergence
            load_convergence: DataFrame = get_loads_convergence(loads_file, gnvp_version)
            convergence: DataFrame = get_error_convergence(log_file, load_convergence, gnvp_version)
            if vehicle_name not in self.convergence_data.keys():
                self.convergence_data[vehicle_name] = Struct()
            self.convergence_data[vehicle_name][case] = convergence

    def load_lspt_data(
        self,
        plane: Airplane | None,
        state: State | None,
        vehicle_folder: str,
    ) -> None:
        if plane is None:
            vehicle_name = vehicle_folder
        else:
            vehicle_name = plane.name

        file_lspt: str = os.path.join(DB3D, vehicle_folder, "forces.lspt")
        try:
            forces_df = pd.read_csv(file_lspt)
            self.add_forces(vehicle_name, forces_df)
            logging.info(f"Loading Forces from {file_lspt}")
            for name in [f"LSPT Potential", "LSPT 2D"]:
                self.add_polars_from_forces(plane=plane, state=state, forces=forces_df, prefix=name)
        except FileNotFoundError:
            logging.debug(f"No forces.lspt file found in {vehicle_folder} folder at {DB3D}!\nNo polars Created as well")

    def load_avl_data(
        self,
        plane: Airplane | None,
        state: State | None,
        vehicle_folder: str,
    ) -> None:
        if plane is None:
            vehicle_name = vehicle_folder
        else:
            vehicle_name = plane.name

        file_avl: str = os.path.join(DB3D, vehicle_folder, "forces.avl")
        try:
            forces_df = pd.read_csv(file_avl)
            self.add_forces(vehicle_name, forces_df)
            logging.info(f"Loading AVL Forces from {file_avl}")
            for name in [f"AVL"]:
                self.add_polars_from_forces(plane=plane, state=state, forces=forces_df, prefix=name)
        except FileNotFoundError:
            logging.debug(f"No forces.avl file found in {vehicle_folder} folder at {DB3D}!\nNo polars Created as well")

    def add_polars_from_forces(
        self,
        plane: Airplane | None,
        state: State | None,
        forces: DataFrame,
        prefix: str,
    ) -> None:
        if plane is None:
            logging.info("Could not Create Polars")
            return
        if state is None:
            logging.info("Could not Create Polars")
            return

        Q = state.dynamic_pressure
        S: float = plane.S
        MAC: float = plane.mean_aerodynamic_chord

        AoA = forces["AoA"]
        df: DataFrame = pd.DataFrame()
        df[f"AoA"] = AoA
        df[f"{prefix} CL"] = forces[f"{prefix} Fz"] / (Q * S)
        df[f"{prefix} CD"] = forces[f"{prefix} Fx"] / (Q * S)
        df[f"{prefix} Cm"] = forces[f"{prefix} My"] / (Q * S * MAC)
        if plane.name not in self.polars.keys():
            self.polars[plane.name] = df
        else:
            for col in df.keys():
                if col in self.polars[plane.name].columns and col != "AoA":
                    self.polars[plane.name].drop(col, axis=1, inplace=True)

            # Merge the df with the old data on the AoA column
            self.polars[plane.name] = self.polars[plane.name].merge(
                df,
                on="AoA",
                how="outer",
            )
            # Sort the dataframe by AoA
            self.polars[plane.name].sort_values(by="AoA", inplace=True)

    def add_forces(self, planename: str, forces: DataFrame) -> None:
        if f"{planename}" not in self.forces.keys():
            self.forces[f"{planename}"] = deepcopy(forces)
        else:
            for col in forces.columns:
                # Drop old forces except AoA
                if col in self.forces[planename].columns and col != "AoA":
                    self.forces[planename].drop(col, axis=1, inplace=True)

            # Merge the df with the old data on the AoA column
            self.forces[f"{planename}"] = self.forces[f"{planename}"].merge(
                forces,
                on="AoA",
                how="outer",
            )
            # Sort the dataframe by AoA
            self.forces[f"{planename}"].sort_values(by="AoA", inplace=True)

    def __str__(self) -> str:
        return f"Vehicle Database at {DB3D}"

    def get_case_directory(
        self, 
        airplane: Airplane,
        solver: str,
        case: str | None = None,
        subcase: str | None = None
    ) -> str:
        if solver not in ["GenuVP3", "GenuVP7", "LSPT", "AVL"]:
            raise ValueError(f"Solver {solver} not recognized")
    
        if case is None:
            return os.path.join(DB3D, airplane.directory, solver)
        else:
            if subcase is not None:
                return os.path.join(DB3D, airplane.directory, solver, case, subcase)
            else:
                return os.path.join(DB3D, airplane.directory, solver, case)
