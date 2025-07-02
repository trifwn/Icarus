from typing import Any

from inquirer import List
from inquirer import prompt

from cli.options import get_option
from cli.options import input_options
from ICARUS.airfoils import Airfoil
from ICARUS.computation import Solver
from ICARUS.computation.analyses.analysis import Analysis
from ICARUS.vehicle import Airplane


def set_analysis(solver: Solver) -> None:
    analyses: list[Analysis] = solver.get_analyses(verbose=True)
    analyses_quest: list[List] = [
        List(
            "analysis",
            message="Which analysis do you want to perform",
            choices=[analysis for analysis in analyses],
        ),
    ]
    answer: dict[Any, Any] | None = prompt(analyses_quest)
    if answer is None:
        print("Exited by User")
        exit()
    if answer["analysis"] in [analysis.name for analysis in analyses]:
        solver.select_analysis(answer["analysis"])
    else:
        print("Error")
        return set_analysis(solver)


def set_analysis_options(solver: Solver, obj: Airplane | Airfoil) -> None:
    # _ = solver.get_analysis_input(verbose=True)
    raise NotImplementedError("This function is not implemented yet after updating computation..")
    options = solver.analyses[solver.mode].input_type
    answers: dict[str, Any] = {}
    for option in options.values():
        if option.name == "airfoil":
            answers[option.name] = obj
            continue
        if option.name == "state" and isinstance(obj, Airplane):
            from cli.airplane_cli import get_state

            answers[option.name] = get_state(obj)
            continue
        if option.name == "solver2D":
            from cli.airplane_cli import get_2D_polars_solver

            answers[option.name] = get_2D_polars_solver()
            continue

        try:
            question_type = input_options[option.value_type]
        except KeyError:
            print(f"Option {option} has an invalid type")
            continue

        answer = get_option(option.name, question_type)
        answers[option.name] = answer[option.name]

    try:
        solver.set_analysis_input(answers)
        print("Options set")
        _ = solver.get_analysis_input(verbose=True)

    except Exception:
        print("Unable to set options! Try Again")
        return set_analysis_options(solver, obj)
