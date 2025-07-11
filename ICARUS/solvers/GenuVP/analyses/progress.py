from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ICARUS.computation.core import ProgressEvent
from ICARUS.computation.core import Task
from ICARUS.core.utils import tail
from ICARUS.database import Database
from ICARUS.database import angle_to_directory
from ICARUS.database import disturbance_to_directory
from ICARUS.flight_dynamics import Disturbance
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

if TYPE_CHECKING:
    from ICARUS.solvers.GenuVP import GenuVP3Parameters
    from ICARUS.solvers.GenuVP import GenuVP7Parameters


class GenuVPAnalysisError(Exception):
    """Custom exception for analysis errors."""

    pass


@dataclass
class GenuVPCaseProgress:
    case: str | None = None
    error: bool = False
    iteration: int | None = None
    max_iterations: int | None = None

    def success(self) -> bool:
        """Check if the angle analysis is complete."""
        return (
            self.iteration is not None
            and self.max_iterations is not None
            and self.iteration >= self.max_iterations
            and not self.error
        )

    def finished(self) -> bool:
        """Check if the angle analysis is finished."""
        return (
            self.iteration is not None
            and self.max_iterations is not None
            and self.iteration >= self.max_iterations
        ) or self.error

    def has_started(self) -> bool:
        """Check if the angle analysis has started."""
        return self.iteration is not None and self.iteration > 0

    def is_running(self) -> bool:
        """Check if the angle analysis is still running."""
        return self.has_started() and not self.finished()


def get_aseq_progress(
    task: Task,
    plane: Airplane,
    state: State,
    angles: float,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> ProgressEvent:
    """Get the latest iteration of F2W

    Args:
        REYNDIR (str): Directory where it is run

    Returns:
        list[GenuVPCaseProgress]: List of progress for each angle
    """
    DB = Database.get_instance()
    base_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{solver_parameters.genu_version}",
    )
    case_directory: str = os.path.join(base_directory, angle_to_directory(angles))

    latest_t, error = latest_time(
        CASEDIR=case_directory,
        gnvp_version=solver_parameters.genu_version,
    )

    GenuVPCaseProgress(
        case=f"{angles} deg",
        error=error,
        iteration=latest_t,
        max_iterations=solver_parameters.iterations,
    )
    if error:
        exc = GenuVPAnalysisError(
            f"Error in GenuVP analysis for {plane.name} at angle {angles} with state {state.name}. Check the output files in {case_directory}.",
        )
    event = ProgressEvent(
        task_id=task.id,
        name=f"Plane {plane.name} - Angle {angles} deg",
        current_step=latest_t if latest_t is not None else 0,
        total_steps=solver_parameters.iterations,
        error=exc if error else None,
    )

    return event


def get_stability_progress(
    task: Task,
    plane: Airplane,
    state: State,
    disturbances: Disturbance,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> ProgressEvent:
    """Get the latest iteration of F2W

    Args:
        REYNDIR (str): Directory where it is run

    Returns:
        list[GenuVPCaseProgress]: List of progress for each angle
    """
    DB = Database.get_instance()
    base_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{solver_parameters.genu_version}",
        case="Dynamics",
    )
    case_directory: str = os.path.join(
        base_directory,
        disturbance_to_directory(disturbances),
    )

    latest_t, error = latest_time(
        CASEDIR=case_directory,
        gnvp_version=solver_parameters.genu_version,
    )

    GenuVPCaseProgress(
        case=f"{disturbances}",
        error=error,
        iteration=latest_t,
        max_iterations=solver_parameters.iterations,
    )
    if error:
        exc = GenuVPAnalysisError(
            f"Error in GenuVP analysis for {plane.name} at {disturbances} with state {state.name}. Check the output files in {case_directory}.",
        )
    event = ProgressEvent(
        task_id=task.id,
        name=f"Plane {plane.name} - DST {disturbances}",
        current_step=latest_t if latest_t is not None else 0,
        total_steps=solver_parameters.iterations,
        error=exc if error else None,
    )

    return event


def latest_time(
    CASEDIR: str,
    gnvp_version: int,
) -> tuple[int | None, bool]:
    """Get the latest iteration of GNVP3

    Args:
        CASEDIR (str): Directory where it is run
        gnvp_version (int): Version of GNVP

    Returns:
        Tuple[Optional[int], Optional[float], bool]: Tuple containing IBLM iteration, the angle where the simulation is, and an error flag.

    """
    filename: str = os.path.join(CASEDIR, f"gnvp{gnvp_version}.out")
    try:
        with open(filename, "rb") as f:
            data_b: list[bytes] = tail(f, 300)
        data: list[str] = [line.decode() for line in data_b]
    except FileNotFoundError:
        return None, False

    # ERROR
    error: bool = any(re.search(r"forrtl", x) for x in data)

    # ITERATION
    matches: list[str] = [x.split("   ")[1] for x in data if re.search(r"NTIME =", x)]
    times: list[int] = [int(x) for x in matches]
    try:
        latest_t: int = max(times)
        return latest_t, error
    except ValueError:
        return None, error
