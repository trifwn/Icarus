import sys
from typing import Any

import numpy as np
from inquirer import prompt
from inquirer import Text


def get_option(
    option_name: str,
    question_type: str,
) -> dict[str, Any]:
    if question_type.startswith("list_"):
        quest: Text = Text(
            f"{option_name}",
            message=f"{option_name} (Multiple Values Must be seperated with ','. You can also specify a range as a:b:c where a and b are the endpoints and c the step )",
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
            # Check if the user specified a range
            if ":" in answer[option_name]:
                a, b, c = answer[option_name].split(":")
                answer[option_name] = np.linspace(float(a), float(b), num=int(c))
            else:
                answer[option_name] = [float(x) for x in answer[option_name].split(",")]
        elif question_type == "list_int":
            answer[option_name] = [int(x) for x in answer[option_name].split(",")]
        elif question_type == "list_bool":
            answer[option_name] = [bool(x) for x in answer[option_name].split(",")]
        elif question_type == "list_str":
            answer[option_name] = [str(x) for x in answer[option_name].split(",")]
    except Exception as e:
        print(answer)
        print("Error Getting Answer! Try Again")
        print(f"Got error {e}")
        import sys

        sys.exit()
    return answer


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
