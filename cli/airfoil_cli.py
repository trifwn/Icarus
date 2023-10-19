import time
from typing import Any

from inquirer import Checkbox
from inquirer import List
from inquirer import Path
from inquirer import prompt
from inquirer import Text
from tqdm.asyncio import tqdm

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.struct import Struct
from ICARUS.Core.Units import calc_mach
from ICARUS.Core.Units import calc_reynolds
from ICARUS.Database.db import DB
from ICARUS.Solvers.Airfoil.f2w_section import get_f2w_section
from ICARUS.Solvers.Airfoil.open_foam import get_open_foam
from ICARUS.Solvers.Airfoil.xfoil import get_xfoil
from ICARUS.Workers.solver import Solver


def ask_num_airfoils() -> int:
    no_question: list[Text] = [
        Text("num_airfoils", message="How many airfoils Do you want to run", autocomplete=1),
    ]
    answer: dict[Any, Any] | None = prompt(no_question)
    if answer is None:
        print("Exited by User")
        exit()
    try:
        if int(answer["num_airfoils"]) < 1:
            print(f"Answer {answer['num_airfoils']} doesnt make sense")
            return ask_num_airfoils()
        return int(answer["num_airfoils"])
    except ValueError:
        print(f"Answer {answer['num_airfoils']} doesnt make sense")
        return ask_num_airfoils()


def get_airfoil_file() -> Airfoil:
    file_question: list[Path] = [
        Path(
            "airf_file",
            message="Specify the Path to the airfoil File",
            path_type=Path.FILE,
        ),
    ]
    answers: dict[Any, Any] | None = prompt(file_question)
    if answers is None:
        print("Exited by User")
        exit()
    else:
        try:
            return Airfoil.load_from_file(answers["airf_file"])
        except Exception as e:
            print(e)
            return get_airfoil_file()


def get_airfoil_NACA() -> Airfoil:
    naca_question: list[Text] = [
        Text("naca_dig", message="NACA digits (4 or 5):"),
    ]
    answers: dict[Any, Any] | None = prompt(naca_question)
    if answers is None:
        print("Exited by User")
        exit()
    else:
        try:
            return Airfoil.naca(answers["naca_dig"])
        except Exception as e:
            print(e)
            return get_airfoil_NACA()


def get_airfoil_db(db: DB) -> Airfoil:
    airfoils: Struct = db.foilsDB.set_available_airfoils()
    airfoil_question: list[List] = [
        List(
            "airfoil",
            message="Which airfoil do you want to load",
            choices=[airf_name for airf_name in airfoils.keys()],
        ),
    ]
    answer: dict[Any, Any] | None = prompt(airfoil_question)
    if answer is None:
        print("Exited by User")
        exit()

    try:
        airfoil: Airfoil = airfoils[answer["airfoil"]]
        return airfoil
    except Exception as e:
        print(e)
        return get_airfoil_db(db)


def select_airfoil_source(db: DB) -> Airfoil:
    airfoil_source_question: list[List] = [
        List(
            "airfoil_source",
            message="Where do you want to get the airfoils from",
            choices=["File", "NACA Digits", "Database"],
        ),
    ]

    answer: dict[Any, Any] | None = prompt(airfoil_source_question)
    if answer is None:
        print("Exited by User")
        exit()
    if answer["airfoil_source"] == "File":
        return get_airfoil_file()
    elif answer["airfoil_source"] == "NACA Digits":
        return get_airfoil_NACA()
    elif answer["airfoil_source"] == "Database":
        return get_airfoil_db(db)
    else:
        print("Error")
        return select_airfoil_source(db)


def set_analysis(solver: Solver) -> None:
    analyses: list[str] = solver.available_analyses_names(verbose=True)
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
    if answer["analysis"] in analyses:
        solver.set_analyses(answer["analysis"])
    else:
        print("Error")
        return set_analysis(solver)


input_options = {
    float: "float",
    int: "int",
    bool: "bool",
    str: "text",
    list[float]: "list_float",
    list[int]: "list_int",
    list[str]: "list_str",
    list[bool]: "list_bool",
    list[str]: "list_str",
}


def get_option(option_name: str, question_type: str) -> dict[str, Any]:
    if question_type.startswith("list_"):
        quest: Text = Text(
            f"{option_name}",
            message=f"{option_name} (Multiple Values Must be seperated with ',') = ",
        )
    else:
        quest = Text(
            f"{option_name}",
            message=f"{option_name} = ",
        )

    answer: dict[str, Any] | None = prompt([quest])
    if answer is None:
        print("Exited by User")
        exit()

    try:
        if question_type == "float":
            answer[option_name] = float(answer[option_name])
        elif question_type == "int":
            answer[option_name] = int(answer[option_name])
        elif question_type == "bool":
            answer[option_name] = bool(answer[option_name])
        elif question_type == "text":
            answer[option_name] = str(answer[option_name])
        elif question_type == "list_float":
            answer[option_name] = [float(x) for x in answer[option_name].split(",")]
        elif question_type == "list_int":
            answer[option_name] = [int(x) for x in answer[option_name].split(",")]
        elif question_type == "list_bool":
            answer[option_name] = [bool(x) for x in answer[option_name].split(",")]
        elif question_type == "list_str":
            answer[option_name] = [str(x) for x in answer[option_name].split(",")]
    except:
        print(answer)
        print("Error Getting Answer! Try Again")
        import sys

        sys.exit()
    return answer


def set_analysis_options(solver: Solver, db: DB, airfoil: Airfoil) -> None:
    all_options: Struct = solver.get_analysis_options(verbose=True)
    options = Struct()
    answers = {}
    for option in all_options.keys():
        if option == "db":
            answers[option] = db
            continue
        elif option == "airfoil":
            answers[option] = airfoil
            continue
        options[option] = all_options[option]

        try:
            question_type = input_options[options[option].option_type]
        except KeyError:
            print(f"Option {option} has an invalid type")
            continue

        answer = get_option(option, question_type)
        answers[option] = answer[option]

    try:
        solver.set_analysis_options(answers)
        print("Options set")
        _: Struct = solver.get_analysis_options(verbose=True)

    except:
        print("Unable to set options! Try Again")
        return set_analysis_options(solver, db, airfoil)


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


def airfoil_cli(db: DB) -> None:
    """2D CLI"""
    start_time: float = time.time()

    N: int = ask_num_airfoils()
    if N < 1:
        print("N must be greater than 0")
        return

    airfoils: list[Airfoil] = []

    calc_f2w: dict[str, bool] = {}
    f2w_solvers: dict[str, Solver] = {}

    calc_xfoil: dict[str, bool] = {}
    xfoil_solvers: dict[str, Solver] = {}

    calc_of: dict[str, bool] = {}
    open_foam_solvers: dict[str, Solver] = {}

    for i in range(N):
        print("\n")
        airfoil: Airfoil = select_airfoil_source(db)
        airfoils.append(airfoil)

        select_solver_quest: list[Checkbox] = [
            Checkbox(
                "solver",
                message="Which solvers do you want to run",
                choices=["Foil2Wake", "XFoil", "OpenFoam"],
            ),
        ]
        answer: dict[Any, Any] | None = prompt(select_solver_quest)
        if answer is None:
            print("Error")
            return

        if "Foil2Wake" in answer["solver"]:
            calc_f2w[airfoil.name] = True

            # Get Solver
            f2w_solvers[airfoil.name] = get_f2w_section(db)
            set_analysis(f2w_solvers[airfoil.name])
            set_analysis_options(f2w_solvers[airfoil.name], db, airfoil)
            set_solver_parameters(f2w_solvers[airfoil.name])
        else:
            calc_f2w[airfoil.name] = False

        if "XFoil" in answer["solver"]:
            calc_xfoil[airfoil.name] = True

            # Get Solver
            xfoil_solvers[airfoil.name] = get_xfoil(db)
            set_analysis(xfoil_solvers[airfoil.name])
            set_analysis_options(xfoil_solvers[airfoil.name], db, airfoil)
            set_solver_parameters(xfoil_solvers[airfoil.name])
        else:
            calc_xfoil[airfoil.name] = False

        if "OpenFoam" in answer["solver"]:
            calc_of[airfoil.name] = True

            # Get Solver
            open_foam_solvers[airfoil.name] = get_open_foam(db)
            set_analysis(open_foam_solvers[airfoil.name])
            set_analysis_options(open_foam_solvers[airfoil.name], db, airfoil)
            set_solver_parameters(open_foam_solvers[airfoil.name])
        else:
            calc_of[airfoil.name] = False

    #####################################  START Calculations #####################################
    for airfoil in airfoils:
        airfoil_stime: float = time.time()
        print(f"\nRunning airfoil {airfoil.name}\n")

        # Foil2Wake
        if calc_f2w[airfoil.name]:
            f2w_stime: float = time.time()

            # Set Solver Options and Parameters
            f2w_s: Solver = f2w_solvers[airfoil.name]
            f2w_options: Struct = f2w_s.get_analysis_options(verbose=True)
            f2w_solver_parameters: Struct = f2w_s.get_solver_parameters()
            f2w_options.airfoil.value = airfoil

            # Run Solver
            f2w_solvers[airfoil.name].run()

            # Get Results
            _ = f2w_solvers[airfoil.name].get_results()

            f2w_etime: float = time.time()
            print(f"Foil2Wake completed in {f2w_etime - f2w_stime} seconds")

        # XFoil
        if calc_xfoil[airfoil.name]:
            xfoil_stime: float = time.time()

            # Set Solver Options and Parameters
            xfoil: Solver = xfoil_solvers[airfoil.name]
            xfoil_options: Struct = xfoil.get_analysis_options(verbose=True)
            xfoil_options.airfoil.value = airfoil

            # Run Solver and Get Results
            xfoil_solvers[airfoil.name].run()

            xfoil_etime: float = time.time()
            print(f"XFoil completed in {xfoil_etime - xfoil_stime} seconds")

        # OpenFoam
        if calc_of[airfoil.name]:
            of_stime: float = time.time()

            # Set Solver Options and Parameters
            open_foam: Solver = open_foam_solvers[airfoil.name]
            open_foam_options: Struct = open_foam.get_analysis_options(verbose=True)
            open_foam_options.airfoil.value = airfoil

            for reyn in open_foam_options.reynolds.value:
                print(f"Running OpenFoam for Re={reyn}")

                open_foam_options.reynolds.value = reyn

                # Run Solver and Get Results
                open_foam_solvers[airfoil.name].run()

            of_etime: float = time.time()
            print(f"OpenFoam completed in {of_etime - of_stime} seconds")

        airfoil_etime: float = time.time()
        print(
            f"Airfoil {airfoil} completed in {airfoil_etime - airfoil_stime} seconds",
        )
    ##################################### END Calculations ########################################

    end_time: float = time.time()
    print(f"Total time: {end_time - start_time}")
    print("########################################################################")
    print("Program Terminated")
    print("########################################################################")


if __name__ == "__main__":
    db: DB = DB()
    db.load_data()

    airfoil_cli(db)
