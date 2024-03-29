from typing import Any

from inquirer import List
from inquirer import prompt
from inquirer import Text

from ICARUS.computation.solvers.solver import Solver
from ICARUS.core.struct import Struct


def set_solver_parameters(solver: Solver) -> None:
    parameters: Struct = solver.get_solver_parameters(verbose=True)
    change_prompt: list[List] = [
        List(
            "change",
            message="Do you want to change any of the solver parameters",
            choices=["Yes", "No"],
        ),
    ]
    choice: dict[Any, Any] | None = prompt(change_prompt)
    if choice is None:
        print("Exited by User")
        exit()
    if choice["change"] == "Yes":
        parameters_quest: list[Text] = [
            Text(
                f"{parameter}",
                message=f"{parameter} = ",
            )
            for parameter in parameters.keys()
        ]
        answer: dict[Any, Any] | None = prompt(parameters_quest)
        if answer is None:
            print("Exited by User")
            exit()
        try:
            solver.set_solver_parameters(answer)
        except:
            print("Unable to set parameters! Try Again")
            return set_solver_parameters(solver)
