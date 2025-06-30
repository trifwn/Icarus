"""Option input handling using Rich interface."""

from typing import Any

import numpy as np
from rich.console import Console
from rich.prompt import Prompt

# Console for rich output
console = Console()


def get_option(
    option_name: str,
    question_type: str,
) -> dict[str, Any]:
    """Get option value from user using Rich interface."""

    if question_type.startswith("list_"):
        message = f"{option_name} (Multiple values separated with ',' or range as a:b:c)"
    else:
        message = f"{option_name}"

    value = Prompt.ask(message)

    if not value:
        console.print("[red]No value provided[/red]")
        return {}

    try:
        if question_type == "float":
            return {option_name: float(value)}
        elif question_type == "int":
            return {option_name: int(value)}
        elif question_type == "bool":
            return {option_name: bool(value)}
        elif question_type == "text":
            return {option_name: str(value)}
        elif question_type == "list_float":
            # Check if the user specified a range
            if ":" in value:
                a, b, c = value.split(":")
                return {option_name: np.linspace(float(a), float(b), num=int(c))}
            else:
                return {option_name: [float(x) for x in value.split(",")]}
        elif question_type == "list_int":
            return {option_name: [int(x) for x in value.split(",")]}
        elif question_type == "list_bool":
            return {option_name: [bool(x) for x in value.split(",")]}
        elif question_type == "list_str":
            return {option_name: [str(x) for x in value.split(",")]}
        else:
            console.print(f"[red]Unknown option type: {question_type}[/red]")
            return {}
    except Exception as e:
        console.print(f"[red]Error parsing value: {e}[/red]")
        return {}


input_options = {
    float: "float",
    int: "int",
    bool: "bool",
    str: "text",
    list[float]: "list_float",
    list[int]: "list_int",
    list[str]: "list_str",
    list[bool]: "list_bool",
}
