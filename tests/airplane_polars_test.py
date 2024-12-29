import numpy as np
from pandas import DataFrame
from pandas import Series

from ICARUS.database.db import Database


def airplane_polars(plot: bool = False) -> None:
    """Function to test the airplane polars.

    Args:
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple[DataFrame, DataFrame]: Returns the desired and actual results.

    """
    planenames: list[str] = ["bmark"]
    # BMARKLOC: str = os.path.join(EXTERNAL_DB, "bmark.txt")
    # read_polars_3d(BMARKLOC, "bmark")

    # planenames.append("XFLR_bmark")
    if plot:
        from ICARUS.visualization.airplane.db_polars import plot_airplane_polars

        plot_airplane_polars(
            airplane_names=planenames,
            solvers=[
                "GenuVP3 2D",
                "GenuVP3 Potential",
                "GenuVP7 2D",
                "GenuVP7 Potential",
                "LSPT 2D",
                "LSPT Potential",
                "AVL",
            ],
            size=(10, 10),
            title="Benchmark Airplane Polars",
        )

    DB = Database.get_instance()
    desired: DataFrame = DB.vehicles_db.polars["XFLR_bmark"]
    actuals: list[DataFrame] = [
        DB.vehicles_db.polars["bmark"],
    ]

    solvers = ["GNVP3 2D", "GNVP7 2D", "LSPT 2D"]
    for pol in solvers:
        for act in actuals:
            try:
                AoA_d: Series[float] = desired["AoA"].astype(float)
                AoA: Series[float] = act["AoA"].astype(float)

                CL_d: Series[float] = desired["CL"].astype(float)
                CL: Series[float] = act[f"{pol} CL"].astype(float)

                CD_d: Series[float] = desired["CD"].astype(float)
                CD: Series[float] = act[f"{pol} CD"].astype(float)

                Cm_d: Series[float] = desired["Cm"].astype(float)
                Cm: Series[float] = act[f"{pol}Cm"].astype(float)
            except KeyError:
                print("--------ATTENTION----------")
                print(f"{pol} not found")
                continue
            # Compare All Values tha correspond to same AoA
            # to x decimal places (except AoA)
            dec_prec = 2
            for a in AoA:
                for x, x_d in zip([CL, CD, Cm], [CL_d, CD_d, Cm_d]):
                    np.testing.assert_almost_equal(
                        actual=x[AoA == a].to_numpy(),
                        desired=x_d[AoA_d == a].to_numpy(),
                        decimal=dec_prec,
                    )
