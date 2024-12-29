import unittest

from tests.airplane_polars_test import airplane_polars
from tests.avl_run_test import avl_run
from tests.gnvp3_polar_test import gnvp3_run
from tests.solver_geom_test import gnvp3_geometry
from tests.wing_geom_test import geom


class BaseAirplaneTests(unittest.TestCase):
    # def test0_xfoil_run(self) -> None:
    #     xfoil_run()

    def test1_geom(self) -> None:
        geom()

    def test2_gnvp3_run(self) -> None:
        gnvp3_run(run_parallel=True)

    def test3_geometry_gnvp3(self) -> None:
        gnvp3_geometry(plot=True)

    # def test5_gnvp7_run(self) -> None:
    #     gnvp7_run(run_parallel=True)

    # def test6_geometry_gnvp7(self) -> None:
    #     gnvp7_geometry(plot=True)

    # def test7_lspt_run(self) -> None:
    #     lspt_run()

    def test8_avl_run(self) -> None:
        avl_run()

    def test_9_3d_polars(self) -> None:
        airplane_polars(plot=True)


if __name__ == "__main__":
    from ICARUS.database import Database

    DB = Database("./Data")
    unittest.TestLoader.sortTestMethodsUsing = None  # type: ignore
    unittest.main()
