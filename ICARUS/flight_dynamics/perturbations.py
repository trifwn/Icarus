from __future__ import annotations

from typing import TYPE_CHECKING

from .disturbances import Disturbance as dst

if TYPE_CHECKING:
    from ICARUS.flight_dynamics.state import State


def longitudal_pertrubations(
    state: State,
    scheme: str,
    epsilon: dict[str, float] | None = None,
) -> list[dst]:
    """Function to add all longitudinal perturbations
    needed to compute the aero derivatives
    Inputs:
    - variable: string with the variable to perturb
    - amplitude: amplitude of the perturbation
    """
    disturbances: list[dst] = []
    if epsilon is None:
        del epsilon
        epsilon = {"u": 0.01, "w": 0.01, "q": 0.25, "theta": 0.01}  # /plane.trim["U"]

    for var in ["u", "w", "q", "theta"]:
        state.epsilons[var] = epsilon[var]
        if scheme == "Central":
            disturbances.append(dst(var, epsilon[var]))
            disturbances.append(dst(var, -epsilon[var]))
        elif scheme == "Forward":
            disturbances.append(dst(var, epsilon[var]))
        elif scheme == "Backward":
            disturbances.append(dst(var, -epsilon[var]))
        else:
            raise ValueError("Scheme must be 'Central', 'Forward' or 'Backward'")
    return disturbances


def lateral_pertrubations(
    state: State,
    scheme: str,
    epsilon: dict[str, float] | None = None,
) -> list[dst]:
    """Function to add all lateral perturbations
    needed to compute the aero derivatives
    Inputs:
    - variable: string with the variable to perturb
    - amplitude: amplitude of the perturbation
    """
    disturbances: list[dst] = []
    if epsilon is None:
        del epsilon
        epsilon = {"v": 0.01, "p": 0.25, "r": 0.25, "phi": 0.01}

    for var in ["v", "p", "r", "phi"]:
        state.epsilons[var] = epsilon[var]
        if scheme == "Central":
            disturbances.append(dst(var, epsilon[var]))
            disturbances.append(dst(var, -epsilon[var]))
        elif scheme == "Forward":
            disturbances.append(dst(var, epsilon[var]))
        elif scheme == "Backward":
            disturbances.append(dst(var, -epsilon[var]))
        else:
            raise ValueError("Scheme must be 'Central', 'Forward' or 'Backward'")
    return disturbances
