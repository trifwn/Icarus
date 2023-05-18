import numpy as np

from ICARUS.Flight_Dynamics.dynamic_plane import Dynamic_Airplane
from ICARUS.Software.GenuVP3.postProcess.forces import rotate_forces


def longitudalStability(plane: Dynamic_Airplane, mode: str = "2D"):
    """This Function Requires the results from perturbation analysis
    For the Longitudinal Motion, in addition to the state space variables
    an analysis with respect to the derivative of w perturbation is needed.
    These derivatives are in this function are added externally and called
    Xw_dot,Zw_dot,Mw_dot. Depending on the aerodynamics Solver, these w_dot
    derivatives can either be computed like the rest derivatives, or require
    an approximation concerning the downwash velocity that the main wing
    induces on the tail wing
    """

    pertr = plane.pertubResults
    eps = plane.epsilons
    m = plane.M
    U = plane.trim["U"]  # TRIM
    theta = plane.trim["AoA"] * np.pi / 180  # TRIM
    Ue = np.abs(U * np.cos(theta))
    We = np.abs(U * np.sin(theta))

    G = -9.81
    Ix, Iy, Iz, Ixz, Ixy, Iyz = plane.I

    X = {}
    Z = {}
    M = {}
    pertr = pertr.sort_values(by=["Epsilon"]).reset_index(drop=True)
    trimState = pertr[pertr["Type"] == "Trim"]

    for var in ["u", "q", "w", "theta"]:
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
        else:
            raise ValueError(f"Unknown Scheme {plane.scheme}")

        if var == "theta":
            de *= -np.pi / 180
        elif var == "q":
            de *= -np.pi / 180

        back = rotate_forces(back, plane.trim["AoA"])
        front = rotate_forces(front, plane.trim["AoA"])
        Xf = float(front[f"Fx_{mode}"].to_numpy())
        Xb = float(back[f"Fx_{mode}"].to_numpy())
        X[var] = (Xf - Xb) / de

        Zf = float(front[f"Fz_{mode}"].to_numpy())
        Zb = float(back[f"Fz_{mode}"].to_numpy())
        Z[var] = (Zf - Zb) / de

        Mf = float(front[f"M_{mode}"].to_numpy())
        Mb = float(back[f"M_{mode}"].to_numpy())
        M[var] = (Mf - Mb) / de

    X["w_dot"] = 0
    Z["w_dot"] = 0
    M["w_dot"] = 0

    xu = X["u"] / m  # + (X["w_dot"] * Z["u"])/(m*(M-Z["w_dot"]))
    xw = X["w"] / m  # + (X["w_dot"] * Z["w"])/(m*(M-Z["w_dot"]))
    xq = (X["q"] - m * We) / (m)
    xth = G * np.cos(theta)

    # xq += (X["w_dot"] * (Z["q"] + m * Ue))/(m*(m-Z["w_dot"]))
    # xth += - (X["w_dot"]*G * np.sin(theta))/((m-Z["w_dot"]))

    zu = Z["u"] / (m - Z["w_dot"])
    zw = Z["w"] / (m - Z["w_dot"])
    zq = (Z["q"] + m * Ue) / (m - Z["w_dot"])
    zth = (m * G * np.sin(theta)) / (m - Z["w_dot"])

    mu = M["u"] / Iy + Z["u"] * M["w_dot"] / (Iy * (m - Z["w_dot"]))
    mw = M["w"] / Iy + Z["w"] * M["w_dot"] / (Iy * (m - Z["w_dot"]))
    mq = M["q"] / Iy + ((Z["q"] + m * Ue) * M["w_dot"]) / (Iy * (m - Z["w_dot"]))
    mth = -(m * G * np.sin(theta) * M["w_dot"]) / (Iy * (m - Z["w_dot"]))

    plane.AstarLong = np.array(
        [
            [X["u"], X["w"], X["q"], X["theta"]],
            [Z["u"], Z["w"], Z["q"], Z["theta"]],
            [M["u"], M["w"], M["q"], M["theta"]],
            [0, 0, 1, 0],
        ],
    )

    plane.Along = np.array(
        [[xu, xw, xq, xth], [zu, zw, zq, zth], [mu, mw, mq, mth], [0, 0, 1, 0]],
    )

    return X, Z, M
