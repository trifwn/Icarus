import numpy as np
import pytest
from pandas import Series


@pytest.mark.integration
def test_airplane_polars(database_instance):
    """Test the airplane polars comparison between different solvers."""
    planenames: list[str] = ["bmark"]

    solvers = ["GNVP3 2D", "GNVP7 2D", "LSPT 2D"]

    for pol in solvers:
        computed = database_instance.get_vehicle_polars(planenames[0], pol)
        desired = database_instance.get_vehicle_polars(planenames[0], "AVL")

        try:
            AoA_d: Series[float] = desired["AoA"].astype(float)
            AoA: Series[float] = computed["AoA"].astype(float)

            CL_d: Series[float] = desired["CL"].astype(float)
            CL: Series[float] = computed[f"{pol} CL"].astype(float)

            CD_d: Series[float] = desired["CD"].astype(float)
            CD: Series[float] = computed[f"{pol} CD"].astype(float)

            Cm_d: Series[float] = desired["Cm"].astype(float)
            Cm: Series[float] = computed[f"{pol}Cm"].astype(float)
        except KeyError:
            pytest.skip(f"{pol} not found")

        # Compare All Values that correspond to same AoA
        # to x decimal places (except AoA)
        dec_prec = 2
        for a in AoA:
            for x, x_d in zip([CL, CD, Cm], [CL_d, CD_d, Cm_d]):
                np.testing.assert_almost_equal(
                    actual=x[AoA == a].to_numpy(),
                    desired=x_d[AoA_d == a].to_numpy(),
                    decimal=dec_prec,
                )


@pytest.mark.parametrize("plot", [False])
def test_airplane_polars_with_plot(database_instance, plot: bool):
    """Test airplane polars with optional plotting."""
    planenames: list[str] = ["bmark"]

    if plot:
        pytest.importorskip("matplotlib")
        from ICARUS.visualization.airplane import plot_airplane_polars

        plot_airplane_polars(
            airplanes=planenames,
            prefixes=[
                "GenuVP3 Potential",
                "GenuVP3 2D",
                "GenuVP7 Potential",
                "GenuVP7 2D",
                "LSPT Potential",
                "LSPT 2D",
                "AVL",
            ],
            size=(10, 10),
            title="Benchmark Airplane Polars",
        )

    # Run the main test logic
    test_airplane_polars(database_instance)


# Backward compatibility function
def airplane_polars(plot: bool = False) -> None:
    """Legacy function for backward compatibility."""
    # For backward compatibility, we need to initialize database manually
    import os

    from ICARUS.database import Database

    database_folder = os.path.join(os.path.dirname(__file__), "..", "Data")
    if not os.path.exists(database_folder):
        database_folder = ".\\Data"

    # Clear and initialize database
    Database._instance = None
    db = Database(database_folder)
    db.load_all_data()

    if plot:
        test_airplane_polars_with_plot(db, plot)
    else:
        test_airplane_polars(db)
