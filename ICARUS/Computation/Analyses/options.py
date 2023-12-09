import io
from typing import Any


class Option:
    """
    Class that represents options for an analysis. It stores the name, value, description and type of the option in slots

    Args:
        name (str): Name of the option.
        value (Any): Value of the option.
        description (str): Description of the option.
        option_type (Any): Type of the option.

    Methods:
        __getstate__(): Returns a tuple with the name, value and description of the option.
        __setstate__(state): Sets the name, value and description of the option.
        __str__(): Returns a string representation of the option.
        __repr__(): Returns a string representation of the option.
    """

    __slots__ = ["name", "value", "description", "option_type"]

    def __init__(self, name: str, value: Any, description: str, option_type: Any) -> None:
        self.name = name
        self.value = value
        self.description: str = description
        self.option_type = option_type

    def __getstate__(self) -> tuple[str, Any, str]:
        return (self.name, self.value, self.description)

    def __setstate__(self, state: tuple[str, Any, str]) -> None:
        self.name, self.value, self.description = state

    def __str__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} : {self.value}\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()

    def __repr__(self) -> str:
        ss = io.StringIO()

        ss.write(f"{self.name} : {self.value}\n")
        ss.write(f"{self.description}\n")
        return ss.getvalue()
