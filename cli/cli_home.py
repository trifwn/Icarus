"""
ICARUS CLI                                                                               |
Allows the user to chose between 2d and 3d simulations.                                  |
1) In 2D, the user should choose airfoils and define an analysis. The analysis should    |
   be run for a range of angles of attack and Reynolds numbers the results should be     |
   saved in a database. The user should be able to choose solver options and parameters  |
    if needed. Reynolds and Mach can be provided as numers or as a function of Mission   |
    parameters (velocity and geometry)                                                   |
2) In 3D, the user should choose or load an airplane and define an analysis. The plane   |
    can be loaded from a json file (ICARUS object) or parsed from XFLR, or AVL. The      |
    analysis should be specified from loaded or defined solvers and solver options and   |
    parameters should be editable.                                                       |
3) In the end the user should be able to visualize the results of the analysis if he     |
    wishes. Visualization options should be provided. and saved to folder if needed      |

"""
import argparse
from argparse import Namespace
from typing import Any

import inquirer
from inquirer import List
from pyfiglet import Figlet

from ICARUS import __version__
from ICARUS.database import DB


def cli_home() -> None:
    # parse args
    parser = argparse.ArgumentParser(description="ICARUS Aerodynamics")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"ICARUS {__version__}",
    )
    parser.add_argument(
        "-a",
        "--airfoil",
        action="store_true",
        help="Run airfoil analysis",
    )
    parser.add_argument(
        "-p",
        "--plane",
        action="store_true",
        help="Run airplane analysis",
    )

    parser.add_argument(
        "-vis",
        "--visualization",
        action="store_true",
        help="Visualize results",
    )

    f = Figlet(font="slant")

    args: Namespace = parser.parse_args()
    print(f.renderText("ICARUS Aerodynamics"))

    # run airfoil analysis
    from cli.airfoil_cli import airfoil_cli
    from cli.airplane_cli import airplane_cli
    from cli.visualization_cli import visualization_cli

    if args.airfoil:
        airfoil_cli(return_home=True)

    elif args.plane:
        airplane_cli(return_home=True)

    elif args.visualization:
        visualization_cli(return_home=True)

    # No input
    else:
        print("Modes:")
        print("-2D) In 2D, the user should choose airfoils and define an analysis. The analysis should    |")
        print("     be run for a range of angles of attack and Reynolds numbers the results should be     |")
        print("     saved in a database. The user should be able to choose solver options and parameters  |")
        print("     if needed. Reynolds and Mach can be provided as numers or as a function of Mission    |")
        print("     parameters (velocity and geometry)                                                    |")
        print("-3D) In 3D, the user should choose or load an airplane and define an analysis. The plane   |")
        print("     can be loaded from a json file (ICARUS object) or parsed from XFLR, or AVL. The       |")
        print("     analysis should be specified from loaded or defined solvers and solver options and    |")
        print("     parameters should be editable.                                                        |")
        print("VIS) In the end the user should be able to visualize the results of the analysis if he     |")
        print("     wishes. Visualization options should be provided. and saved to folder if needed       |")
        print("                                                                                           |")

        questions: list[List] = [
            inquirer.List(
                "Mode",
                message="Choose modes: (Check All That Apply with space and then press enter)",
                choices=["2D", "3D", "Visualization"],
            ),
        ]
        answers: dict[str, Any] | None = inquirer.prompt(questions)
        if answers is not None:
            mode = answers["Mode"]
            if mode == "2D":
                airfoil_cli(return_home=True)
                DB.foils_db.load_data()
            elif mode == "3D":
                airplane_cli(return_home=True)
                DB.vehicles_db.load_data()
            elif mode == "Visualization":
                visualization_cli(return_home=True)
        else:
            exit()


if __name__ == "__main__":
    # Establish DB Connection
    cli_home()
