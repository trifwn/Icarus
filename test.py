import unittest


class BaseAirplaneTests(unittest.TestCase):
    # def test0_xfoil_run(self) -> None:
    #     xfoil_run()

    def test1_geom(self) -> None:
        from tests.wing_geom_test import geom

        geom()

    def test2_gnvp3_run(self) -> None:
        from tests.gnvp3_polar_test import gnvp3_run

        gnvp3_run(run_parallel=True)

    def test3_geometry_gnvp3(self) -> None:
        from tests.solver_geom_test import gnvp3_geometry

        gnvp3_geometry(plot=True)

    def test5_gnvp7_run(self) -> None:
        from tests.gnvp7_polar_test import gnvp7_run

        gnvp7_run(run_parallel=False)

    def test6_geometry_gnvp7(self) -> None:
        from tests.solver_geom_test import gnvp7_geometry

        gnvp7_geometry(plot=True)

    # def test7_lspt_run(self) -> None:
    #     lspt_run()

    def test8_avl_run(self) -> None:
        from tests.avl_run_test import avl_run

        avl_run()

    def test_9_3d_polars(self) -> None:
        from tests.airplane_polars_test import airplane_polars

        airplane_polars(plot=True)


if __name__ == "__main__":
    from ICARUS.database import Database

    DB = Database("./Data")
    unittest.TestLoader.sortTestMethodsUsing = None  # type: ignore
    unittest.main()
