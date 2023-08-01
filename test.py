import unittest
from typing import Any

import numpy as np
from numpy import dtype
from numpy import ndarray
from pandas import Series

import tests.wing_test as wing_test
from tests.airplane_polars_test import airplane_polars
from tests.solver_geom_test import gnvp_geometry
from tests.solver_run_test import gnvprun


class BaseAirplaneTests(unittest.TestCase):
    def test1_geom(self) -> None:
        S_act: tuple[float] = (4.0,)
        MAC_act: tuple[float] = (0.8,)
        AREA_act: tuple[float] = (4.0608,)
        CG_act: ndarray[Any, dtype[Any]] = np.array([0.163, 0.0, 0.0])
        I_act: ndarray[Any, dtype[Any]] = np.array(
            [2.082, 0.017, 2.099, 0.0, 0.139, 0.0],
        )

        S, MAC, AREA, CG, INERTIA = wing_test.geom()

        np.testing.assert_almost_equal(S, S_act, decimal=4)
        np.testing.assert_almost_equal(MAC, MAC_act, decimal=4)
        np.testing.assert_almost_equal(AREA, AREA_act, decimal=4)
        np.testing.assert_almost_equal(CG, CG_act, decimal=3)
        np.testing.assert_almost_equal(INERTIA, I_act, decimal=3)

    def test2_gnvp_run(self) -> None:
        gnvprun("Serial")
        # gnvprun("Parallel")
        # pass

    def test3_3d_polars(self) -> None:
        des, act = airplane_polars(plot=True)
        prefered_pol = "2D"

        AoA_d: Series[float] = des["AoA"].astype(float)
        AoA: Series[float] = act["AoA"].astype(float)

        CL_d: Series[float] = des["CL"].astype(float)
        CL: Series[float] = act[f"CL_{prefered_pol}"].astype(float)

        CD_d: Series[float] = des["CD"].astype(float)
        CD: Series[float] = act[f"CD_{prefered_pol}"].astype(float)

        Cm_d: Series[float] = des["Cm"].astype(float)
        Cm: Series[float] = act[f"Cm_{prefered_pol}"].astype(float)

        # Compare All Values tha correspond to same AoA
        # to x decimal places (except AoA)
        dec_prec = 1
        for a in AoA:
            for x, x_d in zip([CL, CD, Cm], [CL_d, CD_d, Cm_d]):
                np.testing.assert_almost_equal(
                    actual=x[AoA == a].values,
                    desired=x_d[AoA_d == a].values,
                    decimal=dec_prec,
                )
            # np.testing.assert_almost_equal(
            #     CL_d[AoA_d == a].values,
            #     CL[AoA == a].values,
            #     decimal=dec_prec,
            # )
            # np.testing.assert_almost_equal(
            #     CD_d[AoA_d == a].values,
            #     CD[AoA == a].values,
            #     decimal=dec_prec,
            # )
            # np.testing.assert_almost_equal(
            #     Cm_d[AoA_d == a].values,
            #     Cm[AoA == a].values,
            #     decimal=dec_prec,
            # )

    def test4_geometry_gnvp(self) -> None:
        gridAP, gridGNVP = gnvp_geometry(plot=True)
        np.testing.assert_almost_equal(gridAP, gridGNVP, decimal=3)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None  # type: ignore
    unittest.main()
