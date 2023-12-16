"""
ICARUS CLI for computing Airfoils

Returns:
    None: None
"""
import time
from typing import Any

from inquirer import Checkbox
from inquirer import List
from inquirer import Path
from inquirer import prompt
from inquirer import Text

from .cli_home import cli_home
from cli.analysis import set_analysis
from cli.analysis import set_analysis_options
from cli.solver import set_solver_parameters
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Computation.Solvers.Foil2Wake.f2w_section import Foil2Wake
from ICARUS.Computation.Solvers.OpenFoam.open_foam import OpenFoam
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Computation.Solvers.Xfoil.xfoil import Xfoil
from ICARUS.Core.struct import Struct
from ICARUS.Database import DB


def ask_num_airfoils() -> int:
    no_question: list[Text] = [
        Text(
            "num_airfoils",
            message="How many airfoils Do you want to run",
            autocomplete=1,
        ),
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


def get_airfoil_db() -> Airfoil:
    airfoils: Struct = DB.foils_db.set_available_airfoils()
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
        return get_airfoil_db()


def select_airfoil_source() -> Airfoil:
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
        return get_airfoil_db()
    else:
        print("Error")
        return select_airfoil_source()


def airfoil_cli(return_home: bool = False) -> None:
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
        airfoil: Airfoil = select_airfoil_source()
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
            f2w_solvers[airfoil.name] = Foil2Wake()
            set_analysis(f2w_solvers[airfoil.name])
            set_analysis_options(f2w_solvers[airfoil.name], airfoil)
            set_solver_parameters(f2w_solvers[airfoil.name])
        else:
            calc_f2w[airfoil.name] = False

        if "XFoil" in answer["solver"]:
            calc_xfoil[airfoil.name] = True

            # Get Solver
            xfoil_solvers[airfoil.name] = Xfoil()
            set_analysis(xfoil_solvers[airfoil.name])
            set_analysis_options(xfoil_solvers[airfoil.name], airfoil)
            set_solver_parameters(xfoil_solvers[airfoil.name])
        else:
            calc_xfoil[airfoil.name] = False

        if "OpenFoam" in answer["solver"]:
            calc_of[airfoil.name] = True

            # Get Solver
            open_foam_solvers[airfoil.name] = OpenFoam()
            set_analysis(open_foam_solvers[airfoil.name])
            set_analysis_options(open_foam_solvers[airfoil.name], airfoil)
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
            f2w_options.airfoil = airfoil

            # Run Solver
            f2w_solvers[airfoil.name].define_analysis(f2w_options, f2w_solver_parameters)
            f2w_solvers[airfoil.name].execute()

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
            xfoil_options.airfoil = airfoil

            # Run Solver and Get Results
            xfoil_solvers[airfoil.name].define_analysis(xfoil_options, {})
            xfoil_solvers[airfoil.name].execute()

            xfoil_etime: float = time.time()
            print(f"XFoil completed in {xfoil_etime - xfoil_stime} seconds")

        # OpenFoam
        if calc_of[airfoil.name]:
            of_stime: float = time.time()

            # Set Solver Options and Parameters
            open_foam: Solver = open_foam_solvers[airfoil.name]
            open_foam_options: Struct = open_foam.get_analysis_options(verbose=True)
            open_foam_options.airfoil = airfoil

            for reyn in open_foam_options.reynolds:
                print(f"Running OpenFoam for Re={reyn}")

                open_foam_options.reynolds = reyn

                # Run Solver and Get Results
                open_foam_solvers[airfoil.name].define_analysis(open_foam_options, {})
                open_foam_solvers[airfoil.name].execute()

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
    print("All analyses have terminated")
    print("########################################################################")

    if return_home:
        cli_home()


if __name__ == "__main__":
    airfoil_cli()
