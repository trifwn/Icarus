import os

import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.metrics.polars import AirfoilPolar
from ICARUS.database import Database


def save_polar_results(
    airfoil: Airfoil | list[Airfoil],
    reynolds: list[float],
    results: np.ndarray,
) -> None:
    """Saves the polar results to the database."""

    DB = Database.get_instance()

    if isinstance(airfoil, Airfoil):
        all_airfoils: list[Airfoil] = [airfoil]
    elif isinstance(airfoil, list):
        all_airfoils = airfoil
    else:
        raise TypeError(f"Expected Airfoil or list[Airfoil], got {type(airfoil)}")

    # Is the results is 1D make it 2D (1, n)
    if results.ndim == 1:
        results = results.reshape(1, -1)

    if results.ndim != 2:
        raise ValueError(
            f"Results must be a 2D array with at least 4 columns, got shape {results.shape}",
        )

    assert results.shape[1] == len(
        all_airfoils,
    ), f"Results shape {results.shape[0]} does not match number of airfoils {len(all_airfoils)}"

    assert results.shape[0] >= len(reynolds), (
        f"Results shape {results.shape[1]} does not match number of Reynolds {len(reynolds)}",
    )

    for j, airfoil in enumerate(all_airfoils):
        for i, reyn in enumerate(reynolds):
            if results.shape[1] <= j:
                raise ValueError(
                    f"Results array does not have enough columns for Reynolds {reyn}. "
                    f"Expected at least {len(reynolds)}, got {results.shape[1]}",
                )
            polar = results[i, j]
            if not isinstance(polar, AirfoilPolar):
                raise TypeError(
                    f"Expected AirfoilPolar, got {type(polar)} at index {i}, airfoil {airfoil.name}",
                )

            # Check if the DataFrame is empty by checking if the CL column is empty
            if polar.is_empty():
                print(f"\t\tReynolds {polar.reynolds} failed to converge to a solution")
                continue

            airfoil_dir, reynolds_dir, _ = Database.generate_airfoil_directories(
                airfoil=airfoil,
                reynolds=polar.reynolds,
            )
            polar.save(reynolds_dir, "polar.xfoil")

        # If the airfoil doesn't exist in the DB, save it
        files_in_folder = os.listdir(airfoil_dir)
        if airfoil.file_name not in files_in_folder:
            airfoil.save_selig(airfoil_dir)

        # Add Results to Database
        DB.load_airfoil_data(airfoil)
