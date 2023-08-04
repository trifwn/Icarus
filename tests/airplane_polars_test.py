import os

from pandas import DataFrame

from Data.Planes.simple_wing import airplane as plane
from ICARUS.Database.Database_3D import Database_3D
from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import read_polars_3d


def airplane_polars(plot: bool | None = False) -> tuple[DataFrame, DataFrame]:
    """
    Function to test the airplane polars.

    Args:
        plot (bool | None, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple[DataFrame, DataFrame]: Returns the desired and actual results.
    """
    print("Testing Airplane Polars...")

    db = DB()
    db.load_data()

    db3d: Database_3D = db.vehiclesDB
    planenames: list[str] = [plane.name]
    BMARKLOC: str = os.path.join(db.HOMEDIR, "Data", "XFLR5", "bmark.txt")
    read_polars_3d(db3d, BMARKLOC, "bmark")
    planenames.append("XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplane.gnvp_polars import plot_airplane_polars

        plot_airplane_polars(db3d.data, planenames, ["2D", "Potential"], size=(10, 10))

    desired: DataFrame = db3d.data["XFLR_bmark"]
    actual: DataFrame = db3d.data["bmark"]
    return desired, actual
