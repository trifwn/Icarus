from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from . import LateralStateSpace

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State


def lateral_stability_finite_differences(
    state: State,
) -> LateralStateSpace:
    """This Function Requires the results from perturbation analysis"""
    pert: DataFrame = state.pertrubation_results.sort_values(
        by=["Epsilon"],
    ).reset_index(drop=True)
    # eps: dict[str, float] = state.epsilons

    Y: dict[str, float] = {}
    L: dict[str, float] = {}
    N: dict[str, float] = {}
    trimState: DataFrame = pert[pert["Type"] == "Trim"]
    for var in ["v", "p", "r", "phi"]:
        if state.scheme == "Central":
            back: DataFrame = pert[(pert["Type"] == var) & (pert["Epsilon"] < 0)]
            front: DataFrame = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)]
            de: float = (
                2
                * pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)][
                    "Epsilon"
                ].to_numpy()[0]
            )
        elif state.scheme == "Forward":
            back = trimState
            front = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)]
            de = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)][
                "Epsilon"
            ].to_numpy()[0]
        elif state.scheme == "Backward":
            back = pert[(pert["Type"] == var) & (pert["Epsilon"] < 0)]
            front = trimState
            de = pert[(pert["Type"] == var) & (pert["Epsilon"] > 0)][
                "Epsilon"
            ].to_numpy()[0]
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
