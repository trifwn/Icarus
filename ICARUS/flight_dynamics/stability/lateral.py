from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from ICARUS.flight_dynamics.stability.state_space import LateralStateSpace

if TYPE_CHECKING:
    from ICARUS.flight_dynamics.state import State


def lateral_stability_finite_differences(
    state: State,
) -> LateralStateSpace:
    """This Function Requires the results from perturbation analysis"""
    pertr: DataFrame = state.pertrubation_results.sort_values(
        by=["Epsilon"],
    ).reset_index(drop=True)
    eps: dict[str, float] = state.epsilons

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

        Yf = float(front["Fy"].to_numpy())
        Yb = float(back["Fy"].to_numpy())
        Y[var] = (Yf - Yb) / de

        Lf = float(front["Mx"].to_numpy())
        Lb = float(back["Mx"].to_numpy())
        L[var] = (Lf - Lb) / de

        Nf = float(front["Mz"].to_numpy())
        Nb = float(back["Mz"].to_numpy())
        N[var] = (Nf - Nb) / de

    lateral_state_space = LateralStateSpace(state, Y, L, N)

    return lateral_state_space
