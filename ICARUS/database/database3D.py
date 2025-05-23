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

from ICARUS.core.struct import Struct

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

jsonpickle_pd.register_handlers()


class Database_3D:
    """Class to represent the 3D Database. It contains all the information and results
    of the 3D analysis of the vehicles.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Database_3D:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, APPHOME: str, location: str) -> None:
        self.HOMEDIR: str = APPHOME
        self.DB3D: str = location
        self.planes: dict[str, Airplane] = {}
        self.states: dict[str, dict[str, State]] = {}

        self.forces: dict[str, DataFrame] = {}
        self.polars: dict[str, DataFrame] = {}
        self.transient_data: dict[str, Struct] = {}

        if not os.path.isdir(self.DB3D):
            os.makedirs(self.DB3D)

    def get_vehicle_names(self) -> list[str]:
        """Returns the list of planenames in the database.

        Returns:
            list[str]: List of planenames

        """
        return list(self.planes.keys())

    def load_all_data(self) -> None:
        self.read_all_data()

    def get_state(self, vehicle: str | Airplane, state: str) -> State:
        if isinstance(vehicle, Airplane):
            vehicle = vehicle.name

        if vehicle not in self.states.keys():
            # Try to Load Vehicle object
            self.get_vehicle(vehicle)
            self.states[vehicle] = {}

        if state in self.states[vehicle].keys():
            state_obj: State = self.states[vehicle][state]
            return state_obj
        try:
            self.read_plane_data(vehicle)
            state_obj = self.states[vehicle][state]
            return state_obj
        except KeyError:
            raise ValueError(f"No State found for {state}")

    def get_polars(self, vehicle: str | Airplane, solver: str | None = None) -> DataFrame:
        from ICARUS.vehicle import Airplane

        if isinstance(vehicle, Airplane):
            vehicle = vehicle.name

        if vehicle in self.polars.keys():
            polar_obj: DataFrame = self.polars[vehicle]
            pol = polar_obj
        else:
            self.read_plane_data(vehicle)
            try:
                polar_obj = self.polars[vehicle]
                pol = polar_obj
            except KeyError:
                raise ValueError(f"No Polars found for {vehicle}")

        if solver is not None:
            return pol[[col for col in pol.columns if col.startswith(solver) or col == "AoA"]]
        return pol

    def get_forces(self, vehicle: str | Airplane) -> DataFrame:
        from ICARUS.vehicle import Airplane

        if isinstance(vehicle, Airplane):
            vehicle = vehicle.name

        if vehicle in self.forces.keys():
            forces_obj: DataFrame = self.forces[vehicle]
            return forces_obj
        self.read_plane_data(vehicle)
        try:
            forces_obj = self.forces[vehicle]
            return forces_obj
        except KeyError:
            raise ValueError(f"No Forces found for {vehicle}")

    def get_vehicle(self, name: str) -> Airplane:
        if name in self.planes.keys():
            plane_object: Airplane = self.planes[name]
            return plane_object
        # Try to Load Vehicle object
        file_plane: str = os.path.join(self.DB3D, name, f"{name}.json")
        plane_obj: Airplane | None = self.load_vehicle(name, file_plane)
        print(f"Loaded Plane {plane_obj}")
        if plane_obj is None:
            raise ValueError(f"No Vehicle Object Found at {file_plane}")

        self.planes[name] = plane_obj
        return plane_obj

    def get_states(self, vehicle: str | Airplane) -> dict[str, State]:
        # Import Airplane here to avoid circular imports
        from ICARUS.vehicle import Airplane

        if isinstance(vehicle, Airplane):
            vehicle = vehicle.name

        print(f"Getting States for {vehicle}")
        if vehicle in self.states.keys():
            return self.states[vehicle]
        self.read_plane_data(vehicle)
        try:
            return self.states[vehicle]
        except KeyError:
            raise ValueError(f"No States found for {vehicle}")

    def read_all_data(self) -> None:
        if not os.path.isdir(self.DB3D):
            print(f"Creating self.DB3D directory at {self.DB3D}...")
            os.makedirs(self.DB3D, exist_ok=True)

        vehicle_folders: list[str] = next(os.walk(self.DB3D))[1]
        for vehicle_folder in vehicle_folders:  # For each plane vehicle == folder name
            try:
                self.read_plane_data(vehicle_folder)
            except Exception as error:
                print(f"Error reading {vehicle_folder}! Got error {error}")
                raise error

    def read_plane_data(self, vehicle_folder: str) -> None:
        logging.info(f"Adding Vehicle at {vehicle_folder}")
        # Enter self.DB3D
        DIRNOW = os.getcwd()
        os.chdir(self.DB3D)
        vehicle_folder_path = os.path.join(self.DB3D, vehicle_folder)
        os.chdir(vehicle_folder_path)

        # Load Vehicle object
        file_plane: str = os.path.join(
            self.DB3D,
            vehicle_folder,
            f"{vehicle_folder}.json",
        )
        plane_obj: Airplane | None = self.load_vehicle(
            name=vehicle_folder,
            file=file_plane,
        )
        if plane_obj is None:
            logging.debug(f"No Plane Object Found at {vehicle_folder_path}")
            return None
        self.planes[plane_obj.name] = plane_obj
        vehicle_name = plane_obj.name

        # Load Vehicle State

        state_folders = next(os.walk(os.path.join(self.DB3D, vehicle_folder)))[1]
        for state_folder in state_folders:
            solver_folders = next(os.walk(os.path.join(vehicle_folder_path, state_folder)))[1]
            for solver_folder in solver_folders:
                solver_folder_path = os.path.join(vehicle_folder_path, state_folder, solver_folder)

                state_obj: State | None = self.load_plane_state(solver_folder_path)
                if state_obj is None:
                    logging.debug(f"No State Object Found at {solver_folder_path}")
                    continue

                state_name = f"{state_obj.name}_{solver_folder}"
                if vehicle_name not in self.states.keys():
                    self.states[vehicle_name] = {}
                self.states[vehicle_name][state_name] = state_obj

                # Load Solver Data
                try:
                    self.load_solver_data(
                        vehicle=plane_obj,
                        state=state_obj,
                        folder=solver_folder_path,
                        solver=solver_folder,
                    )
                except ValueError:
                    print(f"Unknown Solver {solver_folder} for {vehicle_name}")
                os.chdir(vehicle_folder_path)
        os.chdir(DIRNOW)

    def load_vehicle(self, name: str, file: str) -> Airplane | None:
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
                    logging.debug(
                        f"Error decoding Plane object {name}! Got error {error}",
                    )
                    raise (error)
                    print(f"Error decoding Plane object {name}! Got error {error}")
        except FileNotFoundError:
            logging.debug(
                f"FileNotFound No Plane object found in {name} folder at {file}!",
            )
        return plane

    def load_plane_state(self, directory: str) -> State | None:
        """Function to load the states of the plane from the states.json file.

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
                from ICARUS.flight_dynamics import State

                if isinstance(obj, State) or obj.__class__.__name__ == "State":
                    state = obj
                else:
                    raise TypeError(f"Expected State object, got {type(obj)}")
        return state

    def load_solver_data(self, vehicle: Airplane, state: State, folder: str, solver: str) -> None:
        if solver == "GenuVP3":
            self.load_gnvp_data(
                vehicle=vehicle,
                state=state,
                folder=folder,
                gnvp_version=3,
            )
        elif solver == "GenuVP7":
            self.load_gnvp_data(
                vehicle=vehicle,
                state=state,
                folder=folder,
                gnvp_version=7,
            )
        elif solver == "LSPT":
            self.load_lspt_data(
                vehicle=vehicle,
                state=state,
                folder=folder,
            )
        elif solver == "AVL":
            self.load_avl_data(
                vehicle=vehicle,
                state=state,
                folder=folder,
            )
        else:
            raise ValueError(f"Solver {solver} not recognized")
        logging.info(f"Added Polars for {vehicle.name} {solver}")

    def load_gnvp_data(
        self,
        vehicle: Airplane,
        state: State,
        folder: str,
        gnvp_version: int,
    ) -> None:
        vehicle_name = vehicle.name
        # Load Forces from forces file and store them in the raw_data dict.
        # If the file doesn't exist it tries to create it by loading the plane object
        # and running the make_polars function. If that fails as weall it logs an error.
        forces_file: str = os.path.join(folder, f"forces.gnvp{gnvp_version}")
        try:
            forces_df = pd.read_csv(forces_file)
            forces_df = forces_df.sort_values("AoA").reset_index(drop=True)
            self.add_forces(vehicle_name, forces_df)
            for name in [
                f"GenuVP{gnvp_version} Potential",
                f"GenuVP{gnvp_version} 2D",
                f"GenuVP{gnvp_version} ONERA",
            ]:
                self.add_polars_from_forces(
                    plane=vehicle,
                    state=state,
                    forces=forces_df,
                    prefix=name,
                )
        except FileNotFoundError:
            logging.debug(
                f"No forces.gnvp{gnvp_version} file found in {folder} folder at {self.DB3D}!\nNo polars Created as well",
            )
            print(
                f"No forces.gnvp{gnvp_version} file found in {folder} folder at {self.DB3D}!\nNo polars Created as well",
            )
        cases: list[str] = next(os.walk(folder))[1]
        if "Dynamics" in cases:
            cases.remove("Dynamics")
            dynamic_cases = next(os.walk(os.path.join(folder, "Dynamics")))[1]
            cases.extend([f"Dynamics/{case}" for case in dynamic_cases])
            pertrubations_file = os.path.join(folder, "Dynamics", f"pertrubations.gnvp{gnvp_version}")
            pertrubations_df = pd.read_csv(pertrubations_file)
            try:
                state.set_pertrubation_results(pertrubations_df)
                state.stability_fd()
            except Exception as error:
                print(f"Error setting pertrubation results {error} for {vehicle_name} GenuVP{gnvp_version}")
                # raise(error)
                logging.debug(f"Error setting pertrubation results {error}")
                state.pertrubation_results = pertrubations_df

        if gnvp_version == 7:
            return
        for case in cases:
            from ICARUS.computation.solvers.GenuVP.post_process.convergence import (
                get_error_convergence,
            )
            from ICARUS.computation.solvers.GenuVP.post_process.convergence import (
                get_loads_convergence,
            )

            # Loads the convergence data from gnvp.out and LOADS_aer.dat and stores it in the
            # convergence_data dict. If LOADS_aer.dat exists it tries to load it and then load
            # the convergence data from gnvp.out. If successfull it adds the error data to the
            # dataframe containing the loads and stores it in the convergence_data dict.
            run_directory = os.path.join(folder, case)
            loads_file: str = os.path.join(run_directory, "LOADS_aer.dat")
            log_file = os.path.join(run_directory, f"gnvp{gnvp_version}.out")

            # Load Convergence
            load_convergence: DataFrame = get_loads_convergence(
                loads_file,
                gnvp_version,
            )
            # print(load_convergence)
            convergence: DataFrame = get_error_convergence(
                log_file,
                load_convergence,
                gnvp_version,
            )
            if vehicle_name not in self.transient_data.keys():
                self.transient_data[vehicle_name] = Struct()
            self.transient_data[vehicle_name][case] = convergence

    def load_lspt_data(
        self,
        vehicle: Airplane,
        state: State,
        folder: str,
    ) -> None:
        vehicle_name = vehicle.name
        file_lspt: str = os.path.join(folder, "forces.lspt")
        try:
            forces_df = pd.read_csv(file_lspt)
            self.add_forces(vehicle_name, forces_df)
            logging.info(f"Loading Forces from {file_lspt}")
            for name in ["LSPT Potential", "LSPT 2D"]:
                self.add_polars_from_forces(
                    plane=vehicle,
                    state=state,
                    forces=forces_df,
                    prefix=name,
                )
        except FileNotFoundError:
            logging.debug(
                f"No forces.lspt file found in {folder} folder at {self.DB3D}!\nNo polars Created as well",
            )

    def load_avl_data(
        self,
        vehicle: Airplane,
        state: State,
        folder: str,
    ) -> None:
        vehicle_name = vehicle.name
        forces_file: str = os.path.join(folder, "forces.avl")
        try:
            forces_df = pd.read_csv(forces_file)
            self.add_forces(vehicle_name, forces_df)
            logging.info(f"Loading AVL Forces from {forces_file}")
            for name in ["AVL"]:
                self.add_polars_from_forces(
                    plane=vehicle,
                    state=state,
                    forces=forces_df,
                    prefix=name,
                )

            # Check if a Dynamics Folder exists
            dynamics_folder = os.path.join(folder, "Dynamics")
            pertrubations_file = os.path.join(folder, "Dynamics", "pertrubations.avl")
            if os.path.isdir(dynamics_folder) and os.path.isfile(pertrubations_file):
                # Load Dynamics Data
                pertrubations_df = pd.read_csv(pertrubations_file)
                try:
                    state.set_pertrubation_results(pertrubations_df, "AVL")
                    state.stability_fd()
                except (ValueError, KeyError) as error:
                    print(f"Error setting pertrubation results {error} for {vehicle_name} AVL")
                    logging.debug(f"Error setting pertrubation results {error}")
                    state.pertrubation_results = pertrubations_df

        except FileNotFoundError:
            logging.debug(
                f"No forces.avl file found in {folder} folder at {self.DB3D}!\nNo polars Created as well",
            )

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
        df["AoA"] = AoA
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
        try:
            if prefix not in state.get_polar_prefixes():
                print(f"\tAdding Polars for {plane.name} {prefix} to State {state.name}")
                state.add_polar(
                    polar=forces,
                    polar_prefix=prefix,
                    is_dimensional=True,
                    verbose=False,
                )
        except Exception as error:
            logging.debug(f"Error adding polar {error}")

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
        return f"Vehicle Database at {self.DB3D}"

    def get_plane_directory(self, plane: Airplane) -> str:
        return os.path.join(self.DB3D, plane.directory)

    def get_case_directory(
        self,
        airplane: Airplane,
        state: State,
        solver: str,
        case: str | None = None,
    ) -> str:
        if solver not in ["GenuVP3", "GenuVP7", "LSPT", "AVL"]:
            raise ValueError(f"Solver {solver} not recognized")

        airplane_path = os.path.join(self.DB3D, airplane.directory)
        state_path = os.path.join(airplane_path, state.name)
        solver_path = os.path.join(state_path, solver)
        case_path = os.path.join(solver_path, case) if case is not None else solver_path

        if not os.path.isdir(case_path):
            os.makedirs(case_path, exist_ok=True)
        return case_path
