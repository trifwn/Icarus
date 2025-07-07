import types
from typing import Any
from typing import Callable


def serialize_function(func: Callable[[Any], Any]) -> dict[str, Any] | None:
    if isinstance(func, types.MethodType):
        func_name = func.__func__.__name__
        return {"py/method": [func.__self__, func_name]}
    if isinstance(func, types.FunctionType):
        if func.__name__ == "<lambda>":
            return {"py/lambda": func.__code__.co_code}
        return {"py/function": func.__module__ + "." + func.__name__}
    return None


def deserialize_function(
    func_dict: dict[str, Any] | None,
) -> Callable[[Any], Any] | None:
    if func_dict:
        func_type, func_info = list(func_dict.items())[0]
        if func_type == "py/method":
            obj, func_name = func_info
            function = getattr(obj, func_name)
        elif func_type == "py/function":
            module_name, func_name = func_info.rsplit(".", 1)
            module = __import__(module_name, fromlist=[func_name])
            function = getattr(module, func_name)
        else:
            function = None
    else:
        function = None
    return function
