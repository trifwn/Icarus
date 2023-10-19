## ICARUS CLI                                                                               |
## Allows the user to chose between 2d and 3d simulations.                                  |
## 1) In 2D, the user should choose airfoils and define an analysis. The analysis should    |
##    be run for a range of angles of attack and Reynolds numbers the results should be     |
##    saved in a database. The user should be able to choose solver options and parameters  |
##     if needed. Reynolds and Mach can be provided as numers or as a function of Mission   |
##     parameters (velocity and geometry)                                                   |
## 2) In 3D, the user should choose or load an airplane and define an analysis. The plane   |
##     can be loaded from a json file (ICARUS object) or parsed from XFLR, or AVL. The      |
##     analysis should be specified from loaded or defined solvers and solver options and   |
##     parameters should be editable.                                                       |
## 3) In the end the user should be able to visualize the results of the analysis if he     |
##     wishes. Visualization options should be provided. and saved to folder if needed      |
##                                                                                          |
import argparse
from argparse import Namespace
from typing import Any

import inquirer
from inquirer import List
from pyfiglet import Figlet

from ICARUS import __version__
from ICARUS.Database.db import DB
from cli.airfoil_cli import airfoil_cli
from cli.airplane_cli import airplane_cli

# from runall_3d import run_3d


# Establish DB Connection
db = DB()
db.load_data()

if __name__ == "__main__":
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
    args: Namespace = parser.parse_args()

    f = Figlet(font="slant")
    print(f.renderText("ICARUS Aerodynamics"))

    # run airfoil analysis
    if args.airfoil:
        airfoil_cli(db=db)

    # # run airplane analysis
    # elif args.plane:
    #     run_3d(db=db, cli = True)

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
                airfoil_cli(db)
            elif mode == "3D":
                airplane_cli(db)
            elif mode == "Visualization":
                # visualize(cli= True)
                print("Visualization")
        else:
            exit()
