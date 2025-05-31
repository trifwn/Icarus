"""ICARUS CLI Visualization module"""

from typing import Any

from inquirer import List
from inquirer import prompt

from ICARUS.database import Database

from .airfoil_vis import aifoil_visualization_cli
from .cli_home import cli_home


def ask_visualization_category() -> str:
    # Ask what to visualize airplane or airfoil
    type_visualization_question = [
        List(
            "visualization_type",
            message="What do you want to visualize?",
            choices=["airfoil", "airplane"],
        ),
    ]
    answer: dict[Any, Any] | None = prompt(type_visualization_question)

    if answer is None:
        print("Exited by User")
        exit()
    try:
        if not isinstance(answer["visualization_type"], str):
            print(f"Answer {answer['visualization_type']} not recognized")
            return ask_visualization_category()
        return str(answer["visualization_type"])
    except ValueError:
        print(f"Answer {answer['visualization_type']} not recognized")
        return ask_visualization_category()


def visualization_cli(DB: Database, return_home: bool = True) -> None:
    """CLI for visualization of Computed data stored in the ICARUS Database

    Args:
        return_home (bool, optional): Whether to loop back to the home of the cli. Defaults to True.

    """
    # Ask what to visualize airplane or airfoil
    vis_category: str = ask_visualization_category()
    if vis_category == "airfoil":
        aifoil_visualization_cli()
    elif vis_category == "airplane":
        pass
    else:
        raise ValueError(f"Visualization Category {vis_category} not recognized")

    if return_home:
        cli_home()


if __name__ == "__main__":
    import os

    from ICARUS import INSTALL_DIR

    database_folder = os.path.join(INSTALL_DIR, "Data")
    DB = Database(database_folder)
    visualization_cli(DB=DB)
