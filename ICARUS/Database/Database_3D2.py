import logging
import os

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd

from . import DB3D
from ICARUS import APPHOME
from ICARUS.Core.struct import Struct

jsonpickle_pd.register_handlers()


class Database_3D:
    """Class to represent the 3D Database. It contains all the information and results
    of the 3D analysis of the vehicles."""

    def __init__(self) -> None:
        self.HOMEDIR: str = APPHOME
        self.DATADIR: str = DB3D
        self.forces = Struct()
        self.polars = Struct()
        self.planes = Struct()
        self.states = Struct()
        self.convergence_data = Struct()

    def load_data(self) -> None:
        self.scan_and_make_data()

    def scan_and_make_data(self) -> None:
        if not os.path.isdir(DB3D):
            print(f"Creating DB3D directory at {DB3D}...")
            os.makedirs(DB3D, exist_ok=True)

        veh_folders: list[str] = next(os.walk(DB3D))[1]
        for vehicle in veh_folders:  # For each plane vehicle == folder name
            # Load Vehicle object
            file_plane: str = os.path.join(DB3D, vehicle, f"{vehicle}.json")
            plane_found: bool = self.load_plane_from_file(vehicle, file_plane)

            solver_folders = next(os.walk(os.path.join(DB3D, vehicle)))[1]

            # solver_directories = ["GenuVP3", "GenuVP7", "LSPT", "AVL"]
            for solver_folder in solver_folders:
                if solver_folder == "GenuVP3":
                    # load_gnvp_data(vehicle, gnvp_version = 3)
                    pass
                elif solver_folder == "GenuVP7":
                    # load_gnvp_data(vehicle, gnvp_version = 7)
                    pass
                elif solver_folder == "LSPT":
                    # load_lspt_data(vehicle, gnvp_version = 3)
                    pass
                elif solver_folder == "AVL":
                    # load_avl_data(vehicle, gnvp_version = 3)
                    pass
                elif solver_folder == "XFLR5":
                    # load_xflr5_data(vehicle, gnvp_version = 3)
                    pass
                else:
                    logging.debug(f"Unknow Solver directory {solver_folder}")
                    pass

    def load_plane_from_file(self, name: str, file: str) -> bool:
        """Function to get Plane Object from file and decode it.

        Args:
            name (str): planename
            file (str): filename

        Returns:
            bool : whether the plane was found or not
        """
        try:
            with open(file, encoding="UTF-8") as f:
                json_obj: str = f.read()
                try:
                    self.planes[name] = jsonpickle.decode(json_obj)
                except Exception as error:
                    logging.debug(f"Error decoding Plane object {name}! Got error {error}")
                    pass
            plane_found = True
        except FileNotFoundError:
            logging.debug(f"No Plane object found in {name} folder at {file}!")
            plane_found = False
        return plane_found

    # @staticmethod
    # def make_polars_from_forces(plane,state, df, prefix)

    def __str__(self) -> str:
        return f"Vehicle Database at {DB3D}"
