import os

from pandas import DataFrame

from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane
from ICARUS.Database import DB
from ICARUS.Database.Database_3D import Database_3D
from ICARUS.Input_Output.XFLR5.polars import read_polars_3d


def airplane_polars(plot: bool | None = False) -> tuple[DataFrame, list[DataFrame]]:
    """
    Function to test the airplane polars.

    Args:
        plot (bool | None, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple[DataFrame, DataFrame]: Returns the desired and actual results.
    """

    planenames: list[str] = ["bmark"]
    BMARKLOC: str = os.path.join(DB.HOMEDIR, "Data", "XFLR5", "bmark.txt")
    read_polars_3d(BMARKLOC, "bmark")
    planenames.append("XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplane.db_polars import plot_airplane_polars

        plot_airplane_polars(
            airplane_names=planenames,
            solvers=["All"],
            size=(10, 10),
            title="Benchmark Airplane Polars",
        )

    desired: DataFrame = DB.vehicles_db.data["XFLR_bmark"]
    actuals: list[DataFrame] = [
        DB.vehicles_db.data["bmark"],
    ]

    return desired, actuals
