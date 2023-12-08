from avl import Dir
import numpy as np


# POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM
def polar_postprocess(plane, angles):
    polar_dir = f"{plane.name}_polarD"

    a_li = []
    CL_li = []
    CD_li = []
    Cm_li = []
    for i, angle in enumerate(angles):
        if angle >= 0:
            p = f"{Dir}/{polar_dir}/{plane.name}_res_{angle}.txt"
        else:
            p = f"{Dir}/{polar_dir}/{plane.name}_res_M{np.abs(angle)}.txt"

        f = open(p)
        con = f.readlines()
        temp_CL = con[23]
        temp_CD = con[24]
        temp_Cm = con[20]

        a_li.append(angle)
        if temp_CL[11] == "-":
            CL_li.append(float(temp_CL[11:19]))
        else:
            CL_li.append(float(temp_CL[12:19]))
        if temp_CD[11] == "-":
            CD_li.append(float(temp_CD[11:19]))
        else:
            CD_li.append(float(temp_CD[12:19]))
        if temp_Cm[33] == "-":
            Cm_li.append(float(temp_Cm[33:41]))
        else:
            Cm_li.append(float(temp_Cm[34:41]))

        f.close()

    return np.array(a_li), np.array(CL_li), np.array(CD_li), np.array(Cm_li)


# POST PROCESSING OF FINITE DIFFERENCE RUNS - RETURNS PERTURBATION ANALYSIS RESULTS IN THE FORM OF DICTIONARIES
# - NECESSARY FOR FINITE-DIFFERENCES BASED DYNAMIC ANALYSIS
# USEFUL FOR SENSITIVITY ANALYSIS WITH RESPECT TO STATE VARIABLE INCREMENTS


def finite_difs_post(plane, inc_ar, var):
    dif_path = f"{plane.mass}_difs"

    CZ_li = []
    CX_li = []
    Cm_li = []
    CY_li = []
    Cl_li = []
    Cn_li = []

    inc_dict = {
        "plane": f"{plane.mass}",
        "axes": "notyet",
        "var": f"{var}",
        "vals": np.concatenate((-inc_ar, inc_ar)),
    }
    if var in ["w", "u", "q"]:
        for inc_v in inc_ar:
            bp = f"{Dir}/{dif_path}/{plane.mass}_dif_b_{var}_{inc_v}"
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
            fp = f"{Dir}/{dif_path}/{plane.mass}_dif_f_{var}_{inc_v}"
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
            bp = f"{Dir}/{dif_path}/{plane.mass}_dif_b_{var}_{inc_v}"
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
            fp = f"{Dir}/{dif_path}/{plane.mass}_dif_f_{var}_{inc_v}"
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
