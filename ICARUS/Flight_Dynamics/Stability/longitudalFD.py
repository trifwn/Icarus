import numpy as np
from ICARUS.Software.GenuVP3.postProcess.forces import rotateForces


def longitudalStability(plane, mode="2D"):
    """This Function Requires the results from perturbation analysis
    For the Longitudinal Motion, in addition to the state space variables an analysis with respect to the derivative of w perturbation is needed.
    These derivatives are in this function are added externally and called Xw_dot,Zw_dot,Mw_dot. Depending on the Aerodynamics Solver,
    these w_dot derivatives can either be computed like the rest derivatives, or require an approximation concerning the downwash velocity
    that the main wing induces on the tail wing
    """

    pertr = plane.pertubResults
    eps = plane.epsilons
    m = plane.pln.M
    U = plane.trim["U"]   # TRIM
    theta = plane.trim["AoA"] * np.pi / 180   # TRIM
    Ue = np.abs(U * np.cos(theta))
    We = np.abs(U * np.sin(theta))

    G = - 9.81
    Ix, Iy, Iz, Ixz, Ixy, Iyz = plane.pln.I

    X = {}
    Z = {}
    M = {}
    pertr = pertr.sort_values(by=["Epsilon"]).reset_index(drop=True)
    trimState = pertr[pertr["Type"] == "Trim"]

    for var in ["u",  "q", "w", "theta"]:
        if plane.scheme == "Central":
            front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            de = 2 * eps[var]
        elif plane.scheme == "Forward":
            front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            back = trimState
            de = eps[var]
        elif plane.scheme == "Backward":
            front = trimState
            back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            de = eps[var]

        if var == 'theta':
            de *= - np.pi / 180
        elif var == 'q':
            de *= - np.pi / 180

        back = rotateForces(back, plane.trim["AoA"])
        front = rotateForces(front, plane.trim["AoA"])
        Xf = float(front[f"Fx_{mode}"].to_numpy())
        Xb = float(back[f"Fx_{mode}"].to_numpy())
        X[var] = (Xf - Xb)/de

        Zf = float(front[f"Fz_{mode}"].to_numpy())
        Zb = float(back[f"Fz_{mode}"].to_numpy())
        Z[var] = (Zf - Zb)/de

        Mf = float(front[f"M_{mode}"].to_numpy())
        Mb = float(back[f"M_{mode}"].to_numpy())
        M[var] = (Mf - Mb)/de

    X["w_dot"] = 0
    Z["w_dot"] = 0
    M["w_dot"] = 0

    xu = X["u"]/m  # + (X["w_dot"] * Z["u"])/(m*(M-Z["w_dot"]))
    xw = X["w"]/m  # + (X["w_dot"] * Z["w"])/(m*(M-Z["w_dot"]))
    xq = (X['q'] - m * We)/(m)
    xth = G*np.cos(theta)

    # xq += (X["w_dot"] * (Z["q"] + m * Ue))/(m*(m-Z["w_dot"]))
    # xth += - (X["w_dot"]*G * np.sin(theta))/((m-Z["w_dot"]))

    zu = Z['u']/(m-Z["w_dot"])
    zw = Z['w']/(m-Z["w_dot"])
    zq = (Z['q']+m*Ue)/(m-Z["w_dot"])
    zth = (m*G*np.sin(theta))/(m-Z["w_dot"])

    mu = M['u']/Iy + Z['u']*M["w_dot"]/(Iy*(m-Z["w_dot"]))
    mw = M['w']/Iy + Z['w']*M["w_dot"]/(Iy*(m-Z["w_dot"]))
    mq = M['q']/Iy + ((Z['q']+m*Ue) *
                      M["w_dot"])/(Iy*(m-Z["w_dot"]))
    mth = - (m*G*np.sin(theta)*M["w_dot"])/(Iy*(m-Z["w_dot"]))

    plane.AstarLong = np.array([[X["u"], X["w"], X["q"], X["theta"]],
                                [Z['u'], Z['w'], Z['q'], Z['theta']],
                                [M['u'], M['w'], M['q'], M['theta']],
                                [0, 0, 1, 0]])

    plane.Along = np.array([[xu, xw, xq, xth],
                            [zu, zw, zq, zth],
                            [mu, mw, mq, mth],
                            [0, 0, 1, 0]])

    print("Longitudal Derivatives")
    print(f"Xu=\t{X['u']}\t\tCxu=\t{xu/(plane.Q*plane.S)}")
    print(f"Xw=\t{X['w']}\t\tCxa=\t{xth/(plane.Q*plane.S)}")
    # print(
    # f"Xth/Ue=\t{X['theta']/(U*np.cos(theta))}")

    print(f"Zu=\t{Z['u']}\t\tCzu=\t{zu/(plane.Q*plane.S)}")
    print(f"Zw=\t{Z['w']}\t\tCLa=\t{Z['theta']/(plane.Q*plane.S)}")
    # print(f"Zth/Ue=\t{Z['theta']/(U*np.cos(theta))}")
    print(f"Zq=\t{Z['q']}\t\tCLq=\t{Z['q']/(plane.Q*plane.S)}")

    print(f"Mu=\t{M['u']}\t\tCmu=\t{M['u']/(plane.Q*plane.S*plane.MAC)}")
    print(
        f"Mw=\t{M['w']}\t\tCma=\t{M['theta']/(plane.Q*plane.S*plane.MAC)}")
    # print(f"Mth/Ue=\t{M['theta']/(U*np.cos(theta))}")
    print(f"Mq=\t{M['q']}\t\tCmq=\t{M['q']/(plane.Q*plane.S*plane.MAC)}\n")
