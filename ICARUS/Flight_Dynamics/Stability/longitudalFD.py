from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame

from ICARUS.Computation.Solvers.GenuVP.post_process.forces import rotate_forces

if TYPE_CHECKING:
    from ICARUS.Flight_Dynamics.state import State


def longitudal_stability_fd(
    state: "State",
    mode: str = "2D",
) -> tuple[dict[Any, float], dict[Any, float], dict[Any, float]]:
    """This Function Requires the results from perturbation analysis
    For the Longitudinal Motion, in addition to the state space variables
    an analysis with respect to the derivative of w perturbation is needed.
    These derivatives are in this function are added externally and called
    Xw_dot,Zw_dot,Mw_dot. Depending on the aerodynamics Solver, these w_dot
    derivatives can either be computed like the rest derivatives, or require
    an approximation concerning the downwash velocity that the main wing
    induces on the tail wing

    Args:
        mode (str, optional): Type of forces to be used "2D", "Onera", "Potential". Defaults to "2D".
    """

    pert: DataFrame = state.pertrubation_results
    eps: dict[str, float] = state.epsilons
    m: float = state.mass
    trim_velocity: float = state.trim["U"]
    trim_angle: float = state.trim["AoA"] * np.pi / 180
    u_e: float = np.abs(trim_velocity * np.cos(trim_angle))
    w_e: float = np.abs(trim_velocity * np.sin(trim_angle))

    G: float = -9.81
    Ix, Iy, Iz, Ixz, Ixy, Iyz = state.inertia

    X: dict[Any, float] = {}
    Z: dict[Any, float] = {}
    M: dict[Any, float] = {}
    pert = pert.sort_values(by=["Epsilon"]).reset_index(drop=True)
    trim_state: DataFrame = pert[pert["Type"] == "Trim"]

    for var in ["u", "q", "w", "theta"]:
        if state.scheme == "Central":
            front: DataFrame = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)]
            back: DataFrame = pert[(pert["Type"] == var) & (pert["Epsilon"] < 0)]
            de: float = 2 * eps[var]
        elif state.scheme == "Forward":
            front = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)]
            back = trim_state
            de = eps[var]
        elif state.scheme == "Backward":
            front = trim_state
            back = pert[(pert["Type"] == var) & (pert["Epsilon"] < 0)]
            de = eps[var]
        else:
            raise ValueError(f"Unknown Scheme {state.scheme}")

        if var == "theta":
            de *= -np.pi / 180
        elif var == "q":
            de *= -np.pi / 180

        back = rotate_forces(back, state.trim["AoA"])
        front = rotate_forces(front, state.trim["AoA"])
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
    xq = (X["q"] - m * w_e) / (m)
    xth = G * np.cos(trim_angle)

    # xq += (X["w_dot"] * (Z["q"] + m * Ue))/(m*(m-Z["w_dot"]))
    # xth += - (X["w_dot"]*G * np.sin(theta))/((m-Z["w_dot"]))

    zu = Z["u"] / (m - Z["w_dot"])
    zw = Z["w"] / (m - Z["w_dot"])
    zq = (Z["q"] + m * u_e) / (m - Z["w_dot"])
    zth = (m * G * np.sin(trim_angle)) / (m - Z["w_dot"])

    mu = M["u"] / Iy + Z["u"] * M["w_dot"] / (Iy * (m - Z["w_dot"]))
    mw = M["w"] / Iy + Z["w"] * M["w_dot"] / (Iy * (m - Z["w_dot"]))
    mq = M["q"] / Iy + ((Z["q"] + m * u_e) * M["w_dot"]) / (Iy * (m - Z["w_dot"]))
    mth = -(m * G * np.sin(trim_angle) * M["w_dot"]) / (Iy * (m - Z["w_dot"]))

    state.longitudal.stateSpace.A = np.array(
        [
            [X["u"], X["w"], X["q"], X["theta"]],
            [Z["u"], Z["w"], Z["q"], Z["theta"]],
            [M["u"], M["w"], M["q"], M["theta"]],
            [0, 0, 1, 0],
        ],
    )

    state.longitudal.stateSpace.A_DS = np.array(
        [[xu, xw, xq, xth], [zu, zw, zq, zth], [mu, mw, mq, mth], [0, 0, 1, 0]],
    )

    return X, Z, M
