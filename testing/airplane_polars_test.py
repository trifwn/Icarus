import os

from pandas import DataFrame

from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Input_Output.XFLR5.polars import read_polars_3d


def airplane_polars(plot: bool = False) -> tuple[DataFrame, list[DataFrame]]:
    """
    Function to test the airplane polars.

    Args:
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple[DataFrame, DataFrame]: Returns the desired and actual results.
    """

    planenames: list[str] = ["bmark"]
    BMARKLOC: str = os.path.join(EXTERNAL_DB, "bmark.txt")
    read_polars_3d(BMARKLOC, "bmark")
    planenames.append("XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplane.db_polars import plot_airplane_polars

        plot_airplane_polars(
            airplane_names=planenames,
            solvers=["GNVP3 Potential", "GNVP3 2D"],
            size=(10, 10),
            title="Benchmark Airplane Polars",
        )

    desired: DataFrame = DB.vehicles_db.data["XFLR_bmark"]
    actuals: list[DataFrame] = [
        DB.vehicles_db.data["bmark"],
    ]

    return desired, actuals
