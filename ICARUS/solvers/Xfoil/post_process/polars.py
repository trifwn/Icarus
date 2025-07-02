import os

from pandas import DataFrame

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.database import Database


def save_polar_results(
    airfoil: Airfoil,
    reynolds: list[float],
    results: FloatArray,
) -> None:
    """Saves the polar results to the database."""

    DB = Database.get_instance()
    reyn_dicts: list[dict[str, FloatArray]] = []
    for reyn_data in results:
        tempDict: dict[str, FloatArray] = {}
        for aoa_data in reyn_data:
            tempDict[str(aoa_data[0])] = aoa_data[1:4]
        reyn_dicts.append(tempDict)

    for i, reyn_data in enumerate(reyn_dicts):
        if len(reyn_data) == 0:
            continue

        _, airfoil_dir, reynolds_dir, _ = Database.generate_airfoil_directories(
            airfoil=airfoil,
            reynolds=reynolds[i],
        )
        os.makedirs(airfoil_dir, exist_ok=True)

        df: DataFrame = DataFrame(reyn_data).T.rename(
            columns={"index": "AoA", 0: "CL", 1: "CD", 2: "Cm"},
        )
        # Check if the DataFrame is empty by checking if the CL column is empty
        if df["CL"].empty:
            print(f"Reynolds {reynolds[i]} failed to converge to a solution")
            continue

        fname = os.path.join(reynolds_dir, "clcd.xfoil")
        df.to_csv(fname, sep="\t", index=True, index_label="AoA")

    # If the airfoil doesn't exist in the DB, save it
    files_in_folder = os.listdir(airfoil_dir)
    if airfoil.file_name not in files_in_folder:
        airfoil.save_selig(airfoil_dir)

    # Add Results to Database
    DB.load_airfoil_data(airfoil)
