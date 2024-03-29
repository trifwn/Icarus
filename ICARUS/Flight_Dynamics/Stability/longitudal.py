from typing import Any
from typing import TYPE_CHECKING

from pandas import DataFrame

from ICARUS.flight_dynamics.stability.state_space import LongitudalStateSpace
from ICARUS.flight_dynamics.stability.state_space import StateSpace

if TYPE_CHECKING:
    from ICARUS.flight_dynamics.state import State


def longitudal_stability_finite_differences(
    state: "State",
) -> "LongitudalStateSpace":
    """This Function Requires the results from perturbation analysis
    For the Longitudinal Motion, in addition to the state space variables
    an analysis with respect to the derivative of w perturbation is needed.
    These derivatives are in this function are added externally and called
    Xw_dot,Zw_dot,Mw_dot. Depending on the aerodynamics Solver, these w_dot
    derivatives can either be computed like the rest derivatives, or require
    an approximation concerning the downwash velocity that the main wing
    induces on the tail wing

    Args:
        State (State): State of the airplane
    """

    pert: DataFrame = state.pertrubation_results
    eps: dict[str, float] = state.epsilons

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

        # back = rotate_forces(back, state.trim["AoA"])
        # front = rotate_forces(front, state.trim["AoA"])
        Xf = float(front[f"Fx"].to_numpy())
        Xb = float(back[f"Fx"].to_numpy())
        X[var] = (Xf - Xb) / de

        Zf = float(front[f"Fz"].to_numpy())
        Zb = float(back[f"Fz"].to_numpy())
        Z[var] = (Zf - Zb) / de

        Mf = float(front[f"My"].to_numpy())
        Mb = float(back[f"My"].to_numpy())
        M[var] = (Mf - Mb) / de

    X["w_dot"] = 0
    Z["w_dot"] = 0
    M["w_dot"] = 0

    longitudal_state_space = LongitudalStateSpace(state, X, Z, M)
    return longitudal_state_space
