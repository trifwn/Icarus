from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ICARUS.airfoils import Airfoil
from ICARUS.computation.core import ProgressEvent
from ICARUS.computation.core import Task
from ICARUS.core.utils import tail
from ICARUS.database import Database
from ICARUS.database import directory_to_angle

if TYPE_CHECKING:
    from ICARUS.solvers.Foil2Wake import Foil2WakeSolverParameters


class Foil2WakeAnalysisError(Exception):
    """Custom exception for analysis errors."""

    pass


@dataclass
class Foil2WakeAngleProgress:
    angle: float | None = None
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
    airfoil: Airfoil,
    reynolds: float,
    solver_parameters: Foil2WakeSolverParameters,
) -> ProgressEvent:
    """Get the latest iteration of F2W

    Args:
        REYNDIR (str): Directory where it is run

    Returns:
        list[Foil2WakeAngleProgress]: List of progress for each angle
    """
    DB = Database.get_instance()
    _, REYNDIR, _ = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
    )
    folders: list[str] = next(os.walk(REYNDIR))[1]
    progresses: list[Foil2WakeAngleProgress] = []
    for folder in folders:
        angle = directory_to_angle(folder)
        filename: str = os.path.join(REYNDIR, folder, "f2w.out")

        # Check if the file exists
        if not os.path.exists(filename):
            progresses.append(Foil2WakeAngleProgress(angle=angle))
            continue

        with open(filename, "rb") as f:
            data_b: list[bytes] = tail(f, 500)
        data: list[str] = [line.decode() for line in data_b]
        data = [x.strip() for x in data if x.strip()]
        data = [x for x in data if x.startswith("Ntime")]
        times: list[int] = [int(x[9:]) for x in data if x.startswith("Ntime")]

        if not times:
            progresses.append(Foil2WakeAngleProgress(angle=angle))
            continue

        latest_t: int = max(times)
        error: bool = any(re.search(r"forrtl", x) for x in data)
        error = error or any(re.search(r"Backtrace", x) for x in data)

        progresses.append(
            Foil2WakeAngleProgress(
                angle=angle,
                iteration=latest_t,
                error=error,
                max_iterations=solver_parameters.iterations,
            ),
        )

    task_error = Foil2WakeAnalysisError() if any(p.error for p in progresses) else None

    total_sub_tasks = len(folders)
    completed_sub_tasks = [p for p in progresses if p.finished()]
    running_tasks = [p for p in progresses if p.is_running()]

    event = ProgressEvent(
        task_id=task.id,
        name=f"Foil2Wake Progress for {airfoil.name} at Re={reynolds}",
        current_step=len(completed_sub_tasks),
        total_steps=total_sub_tasks,
        error=task_error,
        message=f"(running aoa = {running_tasks[0].angle})" if running_tasks else "",
    )

    return event
