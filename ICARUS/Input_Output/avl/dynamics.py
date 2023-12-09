import numpy as np
import os

from numpy import deg2rad, rad2deg

from avl import Dir


# EIGENVALUE ANALYSIS BASED ON THE IMPLICIT DIFFERENTIATION APPROACH OF M.DRELA AND AVL
def implicit_eigs(plane):
    pl_dir = f"{plane.name}_genD"
    polar_dir = f"{plane.name}_polarD"
    eig_dir = f"{plane.name}_eigD"
    log = f"{eig_dir}/{plane.name}_eiglog.txt"
    os.system(f"mkdir {eig_dir}")

    f_li = []
    f_li.append(f"load {pl_dir}/{plane.name}.avl")
    f_li.append(f"mass {pl_dir}/{plane.name}.mass")
    f_li.append("MSET")
    f_li.append("0")
    f_li.append("oper")
    f_li.append("1")
    f_li.append("a")
    f_li.append("pm")
    f_li.append("0")
    f_li.append("x")
    f_li.append("C1")
    f_li.append("b")
    f_li.append("0")
    f_li.append("    ")
    f_li.append("x")
    f_li.append("    ")
    f_li.append("mode")
    f_li.append("1")
    f_li.append("N")
    f_li.append("    ")
    f_li.append("quit")
    ar = np.array(f_li)
    np.savetxt(f"{eig_dir}/{plane.name}_mode_scr", ar, delimiter=" ", fmt="%s")
    os.system(f"./avl.exe < {eig_dir}/{plane.name}_mode_scr > {log}")

    os.getcwd()

    print(os.getcwd())
    char = "'{*}'"
    os.system(f"csplit -z {log} /1:/ {char}")
    p = "xx10"
    f = open(p)
    lines = f.readlines()
    f.close()
    print(os.getcwd())
    long_li = []
    lat_li = []
    ind_li = np.arange(0, 8, 1) * 6
    for i in ind_li:
        vals = eigpro(i, lines)
        if np.mean(np.abs(vals[0])) == 0:
            lat_li.append(vals[2])
        elif np.mean(np.abs(vals[1])) == 0:
            long_li.append(vals[2])
    os.system("rm -rf xx*")

    # )

    return long_li, lat_li


# NECESSARY FUNCTION FOR THE ABOVE
def eigpro(st, lines):
    st = int(st)
    mode = np.array([float(lines[st][10:22]), float(lines[st][24:32])])
    long_vecs = np.zeros((4, 2))
    lat_vecs = np.zeros((4, 2))
    for i in range(0, 4):
        long_vecs[i, :] = np.array(
            [float(lines[st + i + 1][9:15]), float(lines[st + i + 1][20:26])]
        )
        lat_vecs[i, :] = np.array(
            [float(lines[st + i + 1][41:47]), float(lines[st + i + 1][52:58])]
        )

    return long_vecs, lat_vecs, mode


# FUCNTION OBTAINING EIGENVALUES FROM PRE-RUN XFLR5 DYNAMIC ANALYSIS (FOR COMPARISON)
def xflr_eigs(plane):
    p = f"{Dir}/{plane.name}_x.eigs"
    f = open(p)
    lines = f.readlines()
    long_li = lines[105]
    lat_li = lines[116]
    eig_mat = np.zeros((5, 2))
    eig_mat[0, 0] = float(long_li[22:38].partition("+")[0])
    eig_mat[0, 1] = float(long_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[1, 0] = float(long_li[74:92].partition("+")[0])
    eig_mat[1, 1] = float(long_li[74:92].partition("+")[2].partition("i")[0])
    eig_mat[2, 0] = float(lat_li[22:38].partition("+")[0])
    eig_mat[2, 1] = float(lat_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[3, 0] = float(lat_li[46:67].partition("+")[0])
    eig_mat[3, 1] = float(lat_li[46:67].partition("+")[2].partition("i")[0])
    eig_mat[4, 0] = float(lat_li[99:119].partition("+")[0])
    eig_mat[4, 1] = float(lat_li[99:119].partition("+")[2].partition("i")[0])

    f.close()
    print("1) Short - Period, 2) Phygoid, 3) Roll-Damping, 4) Dutch Roll, 5) Spiral")
    return eig_mat


# CALCULATION OF AIRCRAFT TRIM AOA AND VELOCITY
def trim_conditions(plane):
    trim_path = f"{plane.name}_trimD"
    pl_dir = f"{plane.name}_genD"
    if os.path.isdir(f"{plane.name}_trimD"):
        os.system(f"rm -rf {trim_path}")
    os.system(f"mkdir {trim_path}")
    li = []

    li.append(f"load {pl_dir}/{plane.name}.avl")
    li.append(f"mass {pl_dir}/{plane.name}.mass")
    li.append(f"MSET")
    li.append("0")
    li.append(f"oper")
    li.append("1")
    li.append(f"a")
    li.append("pm")
    li.append("0")
    li.append("x")
    li.append("c1")
    li.append("b")
    li.append("0")
    li.append("    ")
    li.append("x")
    li.append("FT")
    li.append(f"{trim_path}/{plane.name}_trimmed_res.txt")
    if os.path.isfile(f"{trim_path}/{plane.name}_trimmed_res.txt"):
        li.append("O")
    li.append("    ")
    li.append("quit")
    ar = np.array(li)
    np.savetxt(f"{trim_path}/{plane.name}_trimmed_scr", ar, delimiter=" ", fmt="%s")
    log = f"{trim_path}/{plane.name}_trim_log.txt"
    os.system(f"./avl.exe < {trim_path}/{plane.name}_trimmed_scr > {log}")

    char = "'{*}'"
    os.system(f"csplit -z {log} /Setup/ {char}")
    p = f"xx02"
    f = open(p)
    lines = f.readlines()
    f.close()
    os.system("rm -rf xx*")
    V = float(lines[5][22:28])
    aoa = float(lines[68][10:19])
    return aoa, V


# FUNCTION FOR THE CALCULATION OF STABILITY DERIVATIVES VIA FINITED DIFEERENCE METHOD
def finite_difs(plane, u_ar, v_ar, w_ar, q_ar, p_ar, r_ar, trim_conditions):
    pl_dir = f"{plane.name}_genD"
    dif_path = f"{plane.name}_difs"
    case_num = 0

    if os.path.isdir(dif_path):
        os.system(f"rm -rf {dif_path}")
    os.system(f"mkdir {dif_path}")
    for w in w_ar:
        li = []
        li.append(f"load {pl_dir}/{plane.name}.avl")
        li.append(f"mass {pl_dir}/{plane.name}.mass")
        li.append("MSET")
        li.append("0")
        li.append("oper")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        f_ang = trim_conditions[0] + rad2deg(np.arctan(w / trim_conditions[1]))
        b_ang = trim_conditions[0] - rad2deg(np.arctan(w / trim_conditions[1]))
        li.append(f"{f_ang}")
        li.append("x")
        li.append("FT")
        f_w = f"{plane.name}_dif_f_w_{w}"
        li.append(f_w)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1

        li.append("a")
        li.append("a")
        li.append(f"{b_ang}")
        li.append("x")
        li.append("FT")
        b_w = f"{plane.name}_dif_b_w_{w}"
        li.append(b_w)

        ar_1 = np.array(li)

    for u in u_ar:
        li = []
        w_0 = np.tan(deg2rad(trim_conditions[0])) * trim_conditions[1]
        f_u_ang = rad2deg(np.arctan(w_0 / (trim_conditions[1] + u)))
        b_u_ang = rad2deg(np.arctan(w_0 / (trim_conditions[1] - u)))
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{f_u_ang}")
        li.append("x")
        li.append("FT")
        f_u = f"{plane.name}_dif_f_u_{u}"
        li.append(f_u)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{b_u_ang}")
        li.append("x")
        li.append("FT")
        b_u = f"{plane.name}_dif_b_u_{u}"
        li.append(b_u)

        ar_2 = np.array(li)

        print(u)

    for q in q_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{(q*plane.mean_aerodynamic_chord/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_q = f"{plane.name}_dif_f_q_{q}"
        li.append(f_q)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{-(q*plane.mean_aerodynamic_chord/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_q = f"{plane.name}_dif_b_q_{q}"
        li.append(b_q)

        ar_3 = np.array(li)
    for v in v_ar:
        li = []

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{0}")
        li.append("b")
        li.append("b")
        f_ang = rad2deg(np.arctan(v / trim_conditions[1]))
        b_ang = rad2deg(np.arctan(-v / trim_conditions[1]))
        li.append(f"{f_ang}")
        li.append("x")
        li.append("FT")
        f_v = f"{plane.name}_dif_f_v_{v}"
        li.append(f_v)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1

        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"{b_ang}")
        li.append("x")
        li.append("FT")
        b_v = f"{plane.name}_dif_b_v_{v}"
        li.append(b_v)
        ar_4 = np.array(li)

    for p in p_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"{(p*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_p = f"{plane.name}_dif_f_p_{p}"
        li.append(f_p)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"{-(p*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_p = f"{plane.name}_dif_b_p_{p}"
        li.append(b_p)
        ar_5 = np.array(li)

    for r in r_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"0")
        li.append("y")
        li.append("y")
        li.append(f"{(r*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_r = f"{plane.name}_dif_f_r_{r}"
        li.append(f_r)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("y")
        li.append("y")
        li.append(f"{-(r*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_r = f"{plane.name}_dif_b_r_{r}"
        li.append(b_r)
        li.append("    ")
        li.append("quit")
        ar_6 = np.array(li)
    ar_6 = np.concatenate((ar_1, ar_2, ar_3, ar_4, ar_5, ar_6))
    np.savetxt(f"{plane.name}_difs_script", ar_6, delimiter=" ", fmt="%s")

    os.system(f"./avl.exe < {plane.name}_difs_script ")

    os.system(
        f"mv {b_q} {f_q} {b_u} {f_u} {b_w} {f_w} {b_p} {f_p} {b_v} {f_v} {b_r} {f_r} {plane.name}_difs_script ./{dif_path}"
    )


# LONGITUDAL STATE MATRIX USING FINITE DIFFERENCE STABILITY DERIVATIVES -
# EXECUTION OF POSTPRROCESSING FUNCTION NECESSARY
def long_mat(
    plane, environment, aoa_trim, trim_Vel, u_dict, I_li, w_dict, q_dict, span=4.4
):
    mass = plane.M
    rho = environment.air_density
    g = environment.GRAVITY
    Area = plane.S
    chord = plane.mean_aerodynamic_chord
    span = plane.span
    # good inc = 0.005*U

    U_e = trim_Vel * np.cos(deg2rad(aoa_trim))
    W_e = trim_Vel * np.sin(deg2rad(aoa_trim))

    V_u_f = np.linalg.norm([u_dict["vals"][1] + U_e, W_e])
    V_u_b = np.linalg.norm([u_dict["vals"][0] + U_e, W_e])

    V_w_f = np.linalg.norm([U_e, W_e + w_dict["vals"][1]])
    V_w_b = np.linalg.norm([U_e, W_e + w_dict["vals"][1]])

    print(V_u_f**2 - V_u_b**2)

    X_u = (
        (u_dict["CX"][1] * V_u_f**2 - u_dict["CX"][0] * V_u_b**2)
        * (0.5 * rho * Area)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )
    Z_u = (
        (u_dict["CZ"][1] * V_u_f**2 - u_dict["CZ"][0] * V_u_b**2)
        * (0.5 * rho * Area)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )
    M_u = (
        (u_dict["Cm"][1] * V_u_f**2 - u_dict["Cm"][0] * V_u_b**2)
        * (0.5 * rho * Area * chord)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )

    X_w = (
        (w_dict["CX"][1] * V_w_f**2 - w_dict["CX"][0] * V_w_b**2)
        * (0.5 * rho * Area)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )
    Z_w = (
        (w_dict["CZ"][1] * V_w_f**2 - w_dict["CZ"][0] * V_w_b**2)
        * (0.5 * rho * Area)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )
    M_w = (
        (w_dict["Cm"][1] * V_w_f**2 - w_dict["Cm"][0] * V_w_b**2)
        * (0.5 * rho * Area * chord)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )

    X_q = (
        (q_dict["CX"][1] - q_dict["CX"][0])
        * (0.5 * rho * trim_Vel**2 * Area)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )
    Z_q = (
        (q_dict["CZ"][1] - q_dict["CZ"][0])
        * (0.5 * rho * trim_Vel**2 * Area)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )
    M_q = (
        (q_dict["Cm"][1] - q_dict["Cm"][0])
        * (0.5 * rho * trim_Vel**2 * Area * chord)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )

    state_mat = np.zeros((4, 4))

    state_mat[0, 0] = X_u / mass
    state_mat[1, 0] = Z_u / mass
    state_mat[2, 0] = M_u / I_li[1]

    state_mat[0, 1] = X_w / mass
    state_mat[1, 1] = Z_w / mass
    state_mat[2, 1] = M_w / I_li[1]

    state_mat[0, 2] = (X_q - mass * W_e) / mass
    state_mat[1, 2] = (Z_q + mass * U_e) / mass
    state_mat[2, 2] = M_q / I_li[1]
    state_mat[3, 0] = 0
    state_mat[3, 1] = 0
    state_mat[3, 2] = 1

    state_mat[0, 3] = -g
    state_mat[1, 3] = 0
    # print(Z_q)
    # print(-Z_q)

    return state_mat


# LATERAL STATE MATRIX USING FINITE DIFFERENCE STABILITY DERIVATIVES
def lat_mat(
    plane, environment, aoa_trim, trim_Vel, v_dict, I_li, p_dict, r_dict, span=4.4
):
    # good inc = 0.005*U

    mass = plane.M
    rho = environment.air_density
    g = environment.GRAVITY
    Area = plane.S
    chord = plane.mean_aerodynamic_chord
    span = plane.span

    U_e = trim_Vel * np.cos(deg2rad(aoa_trim))
    W_e = trim_Vel * np.sin(deg2rad(aoa_trim))

    Y_v = (
        (v_dict["CY"][1] - v_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    l_v = (
        (v_dict["Cl"][1] - v_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    n_v = (
        (v_dict["Cn"][1] - v_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    Y_p = (
        (p_dict["CY"][1] - p_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    l_p = (
        (p_dict["Cl"][1] - p_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    n_p = (
        (p_dict["Cn"][1] - p_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    Y_r = (
        (r_dict["CY"][1] - r_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    l_r = (
        (r_dict["Cl"][1] - r_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    n_r = (
        (r_dict["Cn"][1] - r_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    state_mat = np.zeros((4, 4))

    state_mat[0, 0] = Y_v / mass
    state_mat[1, 0] = (I_li[2] * l_v + (I_li[3]) * n_v) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )
    state_mat[2, 0] = (I_li[0] * n_v + (I_li[3]) * l_v) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )

    state_mat[0, 1] = (Y_p + mass * W_e) / mass
    state_mat[1, 1] = (I_li[2] * l_p + (I_li[3]) * n_p) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )
    state_mat[2, 1] = (I_li[0] * n_p + -(I_li[3]) * l_p) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )

    state_mat[0, 2] = (Y_r - mass * U_e) / mass
    state_mat[1, 2] = (I_li[2] * l_r + (I_li[3]) * n_r) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )
    state_mat[2, 2] = (I_li[0] * n_r + (I_li[3]) * l_r) / (
        I_li[0] * I_li[2] - I_li[3] ** 2
    )

    state_mat[3, 0] = 0
    state_mat[3, 1] = 1
    state_mat[3, 2] = 0
    state_mat[3, 3] = 0

    state_mat[0, 3] = g
    state_mat[1, 3] = 0

    return state_mat
