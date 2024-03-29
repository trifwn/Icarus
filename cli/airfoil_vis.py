from typing import Any

from inquirer import Checkbox
from inquirer import prompt

from ICARUS.database import DB
from ICARUS.visualization.airfoil import __functions__ as airfoil_functions


def ask_visualization_function(functions: list[Any]) -> Any:
    # Ask what to visualize
    visualization_function = [
        Checkbox(
            "visualization_function",
            message="What do you want to visualize?",
            choices=[f.__name__ for f in functions],
        ),
    ]
    answer = prompt(visualization_function)

    if answer is None:
        print("Exited by User")
        exit()

    try:
        return answer["visualization_function"]
    except ValueError:
        print(f"Answer {answer['visualization_function']} not recognized")
        return ask_visualization_function(functions)


def aifoil_visualization_cli() -> None:
    functions = airfoil_functions

    # Ask what to visualize
    vis_functions: str = ask_visualization_function(functions=functions)

    for fun_name in vis_functions:
        # Get the function from the list of functions
        fun = [f for f in functions if f.__name__ == fun_name][0]

        print("====================================")
        print(type(fun))
        print("====================================")
        # Get the arguments for the function
        import inspect

        args = inspect.getfullargspec(fun).args
        kwargs = inspect.getfullargspec(fun).kwonlyargs
        print(f"Running {fun.__name__}")
        print(args)
        print(kwargs)
