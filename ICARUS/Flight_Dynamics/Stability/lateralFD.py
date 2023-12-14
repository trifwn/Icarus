from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame


if TYPE_CHECKING:
    from ICARUS.Flight_Dynamics.state import State


def lateral_stability_fd(
    state: "State",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """This Function Requires the results from perturbation analysis"""
    pertr: DataFrame = state.pertrubation_results.sort_values(
        by=["Epsilon"],
    ).reset_index(drop=True)
    eps: dict[str, float] = state.epsilons
    mass: float = state.mass
    U: float = state.trim["U"]
    theta: float = state.trim["AoA"] * np.pi / 180
    G: float = 9.81

    Ix, Iy, Iz, Ixz, Ixy, Iyz = state.inertia

    Y: dict[str, float] = {}
    L: dict[str, float] = {}
    N: dict[str, float] = {}
    trimState: DataFrame = pertr[pertr["Type"] == "Trim"]
    for var in ["v", "p", "r", "phi"]:
        if state.scheme == "Central":
            back: DataFrame = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            front: DataFrame = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            de: float = 2 * eps[var]
        elif state.scheme == "Forward":
            back = trimState
            front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
            de = eps[var]
        elif state.scheme == "Backward":
            back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            front = trimState
            de = eps[var]
        else:
            raise ValueError(f"Unknown Scheme {state.scheme}")

        Yf = float(front[f"Fy"].to_numpy())
        Yb = float(back[f"Fy"].to_numpy())
        Y[var] = (Yf - Yb) / de

        Lf = float(front[f"L"].to_numpy())
        Lb = float(back[f"L"].to_numpy())
        L[var] = (Lf - Lb) / de

        Nf = float(front[f"N"].to_numpy())
        Nb = float(back[f"N"].to_numpy())
        N[var] = (Nf - Nb) / de

    yv: float = Y["v"] / mass
    yp: float = (Y["p"] + mass * U * np.sin(theta)) / mass
    yr: float = (Y["r"] - mass * U * np.cos(theta)) / mass
    yphi: float = -G * np.cos(theta)

    lv: float = (Iz * L["v"] + Ixz * N["v"]) / (Ix * Iz - Ixz**2)
    lp: float = (Iz * L["p"] + Ixz * N["p"]) / (Ix * Iz - Ixz**2)
    lr: float = (Iz * L["r"] + Ixz * N["r"]) / (Ix * Iz - Ixz**2)
    lphi: float = 0

    nv: float = (Ix * N["v"] + Ixz * L["v"]) / (Ix * Iz - Ixz**2)
    n_p: float = (Ix * N["p"] + Ixz * L["p"]) / (Ix * Iz - Ixz**2)
    nr: float = (Ix * N["r"] + Ixz * L["r"]) / (Ix * Iz - Ixz**2)
    nph: float = 0

    state.lateral.stateSpace.A = np.array(
        [
            [Y["v"], Y["p"], Y["r"], Y["phi"]],
            [L["v"], L["p"], L["r"], L["phi"]],
            [N["v"], N["p"], N["r"], N["phi"]],
            [0, 1, 0, 0],
        ],
    )

    state.lateral.stateSpace.A_DS = np.array(
        [[yv, yp, yr, yphi], [lv, lp, lr, lphi], [nv, n_p, nr, nph], [0, 1, np.tan(theta), 0]],
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
