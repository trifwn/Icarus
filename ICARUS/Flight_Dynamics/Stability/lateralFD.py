import numpy as np

from ICARUS.Software.GenuVP3.postProcess.forces import rotate_forces


def lateralStability(plane, mode="2D"):
    """This Function Requires the results from perturbation analysis"""
    pertr = plane.pertubResults
    eps = plane.epsilons
    Mass = plane.M
    U = plane.trim["U"]
    theta = plane.trim["AoA"] * np.pi / 180
    G = -9.81

    Ix, Iy, Iz, Ixz, Ixy, Iyz = plane.I

    Y = {}
    L = {}
    N = {}
    pertr = pertr.sort_values(by=["Epsilon"]).reset_index(drop=True)
    trimState = pertr[pertr["Type"] == "Trim"]
    for var in ["v", "p", "r", "phi"]:
        if plane.scheme == "Central":
            back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            de = 2 * eps[var]
        elif plane.scheme == "Forward":
            back = trimState
            front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            de = eps[var]
        elif plane.scheme == "Backward":
            back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            front = trimState
            de = eps[var]
        else:
            raise ValueError(f"Unknown Scheme {plane.scheme}")

        if var != "v":
            de *= np.pi / 180

        back = rotate_forces(back, plane.trim["AoA"])
        front = rotate_forces(front, plane.trim["AoA"])

        Yf = float(front[f"Fy_{mode}"].to_numpy())
        Yb = float(back[f"Fy_{mode}"].to_numpy())
        Y[var] = (Yf - Yb) / de

        Lf = float(front[f"L_{mode}"].to_numpy())
        Lb = float(back[f"L_{mode}"].to_numpy())
        L[var] = (Lf - Lb) / de

        Nf = float(front[f"N_{mode}"].to_numpy())
        Nb = float(back[f"N_{mode}"].to_numpy())
        N[var] = (Nf - Nb) / de

    yv = Y["v"] / Mass
    yp = (Y["p"] + Mass * U * np.sin(theta)) / Mass
    yr = (Y["r"] - Mass * U * np.cos(theta)) / Mass
    yphi = -G * np.cos(theta)

    lv = (Iz * L["v"] + Ixz * N["v"]) / (Ix * Iz - Ixz**2)
    lp = (Iz * L["p"] + Ixz * N["p"]) / (Ix * Iz - Ixz**2)
    lr = (Iz * L["r"] + Ixz * N["r"]) / (Ix * Iz - Ixz**2)
    lphi = 0

    nv = (Ix * N["v"] + Ixz * L["v"]) / (Ix * Iz - Ixz**2)
    n_p = (Ix * N["p"] + Ixz * L["p"]) / (Ix * Iz - Ixz**2)
    nr = (Ix * N["r"] + Ixz * L["r"]) / (Ix * Iz - Ixz**2)
    nph = 0

    plane.AstarLat = np.array(
        [
            [Y["v"], Y["p"], Y["r"], Y["phi"]],
            [L["v"], L["p"], L["r"], L["phi"]],
            [N["v"], N["p"], N["r"], N["phi"]],
            [0, 1, 0, 0],
        ],
    )

    plane.Alat = np.array(
        [[yv, yp, yr, yphi], [lv, lp, lr, lphi], [nv, n_p, nr, nph], [0, 1, 0, 0]],
    )

    return Y, L, N
    # print("Lateral Derivatives")
    # print(f"Yv=\t{Y['v']}")
    # print(f"Yp=\t{Y['p']}")
    # print(f"Yr=\t{Y['r']}")
    # print(f"Lv=\t{L['v']}")
    # print(f"Lp=\t{L['p']}")
    # print(f"Lr=\t{L['r']}")
    # print(f"Nv=\t{N['v']}")
    # print(f"Np=\t{N['p']}")
    # print(f"Nr=\t{N['r']}")
