import os

import numpy as np
from pandas import DataFrame

from ICARUS.Computation.Solvers.AVL import Dir
from ICARUS.Core.types import FloatArray
from ICARUS.Database.utils import angle_to_case
from ICARUS.Vehicle.plane import Airplane


#
def polar_postprocess(PLANEDIR: str, angles: FloatArray) -> DataFrame:
    """POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM

    Args:
        PLANEDIR (str): Path to plane directory
        plane (Airplane): Airplane object
        angles (FloatArray): Array of angles of attack

    Returns:
        FloatArray: Array with the following order of vectors: AOA,CL,CD,CM
    """

    AoAs = []
    CLs = []
    CDs = []
    Cms = []
    for angle in angles:
        p = os.path.join(PLANEDIR, f"{angle_to_case(angle)}.txt")

        with open(p, encoding="utf-8") as f:
            con = f.readlines()

        CL = con[23]
        CD = con[24]
        Cm = con[20]

        AoAs.append(angle)
        if CL[11] == "-":
            CLs.append(float(CL[11:19]))
        else:
            CLs.append(float(CL[12:19]))
        if CD[11] == "-":
            CDs.append(float(CD[11:19]))
        else:
            CDs.append(float(CD[12:19]))
        if Cm[33] == "-":
            Cms.append(float(Cm[33:41]))
        else:
            Cms.append(float(Cm[34:41]))
    polar_df = DataFrame(np.array([AoAs, CLs, CDs, Cms]).T, columns=["AOA", "CL", "CD", "Cm"])
    return polar_df


# POST PROCESSING OF FINITE DIFFERENCE RUNS - RETURNS PERTURBATION ANALYSIS RESULTS IN THE FORM OF DICTIONARIES
# - NECESSARY FOR FINITE-DIFFERENCES BASED DYNAMIC ANALYSIS
# USEFUL FOR SENSITIVITY ANALYSIS WITH RESPECT TO STATE VARIABLE INCREMENTS


def finite_difs_post(plane: Airplane, inc_ar, var):
    dif_path = f"{plane.M}_difs"

    CZ_li = []
    CX_li = []
    Cm_li = []
    CY_li = []
    Cl_li = []
    Cn_li = []

    inc_dict = {
        "plane": f"{plane.M}",
        "axes": "notyet",
        "var": f"{var}",
        "vals": np.concatenate((-inc_ar, inc_ar)),
    }
    if var in ["w", "u", "q"]:
        for inc_v in inc_ar:
            bp = f"{Dir}/{dif_path}/{plane.M}_dif_b_{var}_{inc_v}"
            bf = open(bp)
            con = bf.readlines()

            temp_CZ = con[21]
            temp_CX = con[19]
            temp_Cm = con[20]

            CZ_li.append(float(temp_CZ[11:19]))

            CX_li.append(float(temp_CX[11:19]))

            if temp_Cm[33] == "-":
                Cm_li.append(float(temp_Cm[33:41]))
            else:
                Cm_li.append(float(temp_Cm[34:41]))

            bf.close()

        for inc_v in inc_ar:
            fp = f"{Dir}/{dif_path}/{plane.M}_dif_f_{var}_{inc_v}"
            ff = open(fp)
            con = ff.readlines()

            temp_CZ = con[21]
            temp_CX = con[19]
            temp_Cm = con[20]

            CZ_li.append(float(temp_CZ[11:19]))

            CX_li.append(float(temp_CX[11:19]))

            if temp_Cm[33] == "-":
                Cm_li.append(float(temp_Cm[33:41]))
            else:
                Cm_li.append(float(temp_Cm[34:41]))

            ff.close()

            inc_dict["axes"] = "longitudinal"
            inc_dict["CX"] = np.array(CX_li)
            inc_dict["CZ"] = np.array(CZ_li)
            inc_dict["Cm"] = np.array(Cm_li)
    else:
        for inc_v in inc_ar:
            bp = f"{Dir}/{dif_path}/{plane.M}_dif_b_{var}_{inc_v}"
            bf = open(bp)
            con = bf.readlines()

            temp_CY = con[20]
            temp_Cl = con[19]
            temp_Cn = con[21]

            CY_li.append(float(temp_CY[11:19]))

            Cl_li.append(float(temp_Cl[33:41]))

            if temp_Cn[33] == "-":
                Cn_li.append(float(temp_Cn[33:41]))
            else:
                Cn_li.append(float(temp_Cn[34:41]))

            bf.close()

        for inc_v in inc_ar:
            fp = f"{Dir}/{dif_path}/{plane.M}_dif_f_{var}_{inc_v}"
            ff = open(fp)
            con = ff.readlines()

            temp_CY = con[20]
            temp_Cl = con[19]
            temp_Cn = con[21]

            CY_li.append(float(temp_CY[11:19]))

            Cl_li.append(float(temp_Cl[33:41]))

            if temp_Cn[33] == "-":
                Cn_li.append(float(temp_Cn[33:41]))
            else:
                Cn_li.append(float(temp_Cn[34:41]))

            ff.close()

            inc_dict["axes"] = "lateral"
            inc_dict["CY"] = np.array(CY_li)
            inc_dict["Cl"] = np.array(Cl_li)
            inc_dict["Cn"] = np.array(Cn_li)

    return inc_dict
