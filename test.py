import unittest
from typing import Any

import numpy as np
from numpy import dtype
from numpy import ndarray

import tests.wing_test as wing_test
from tests.airplane_polars_test import airPolars
from tests.solver_geom_test import gnvpGeom
from tests.solver_run_test import gnvprun


class TestAdd(unittest.TestCase):
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
        # gnvprun("Serial")
        gnvprun("Parallel")
        # pass

    def test3_airPolars(self) -> None:
        des, act = airPolars(plot=False)
        preffered_pol = "2D"

        AoA_d = des["AoA"].astype(float)
        AoA = act["AoA"].astype(float)

        CL_d = des["CL"]
        CL = act[f"CL_{preffered_pol}"]

        CD_d = des["CD"]
        CD = act[f"CD_{preffered_pol}"]

        Cm_d = des["Cm"]
        Cm = act[f"Cm_{preffered_pol}"]

        # Compare All Values tha correspond to same AoA
        # to x decimal places (except AoA)
        dec_prec = 1
        for a in AoA:
            np.testing.assert_almost_equal(
                CL_d[AoA_d == a].values,
                CL[AoA == a].values,
                decimal=dec_prec,
            )
            np.testing.assert_almost_equal(
                CD_d[AoA_d == a].values,
                CD[AoA == a].values,
                decimal=dec_prec,
            )
            np.testing.assert_almost_equal(
                Cm_d[AoA_d == a].values,
                Cm[AoA == a].values,
                decimal=dec_prec,
            )

    def test4_gnvpGeom(self) -> None:
        gridAP, gridGNVP = gnvpGeom(plot=False)
        np.testing.assert_almost_equal(gridAP, gridGNVP, decimal=3)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None  # type: ignore
    unittest.main()
