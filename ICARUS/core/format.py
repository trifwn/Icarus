from functools import singledispatch
from typing import Any


@singledispatch
def short_format(val: Any) -> str:
    return str(val)


@short_format.register
def _(val: float) -> str:
    return f"{val:.3f}"


@short_format.register
def _(val: list[Any]) -> str:
    return f"[{len(val)} items]"


@short_format.register
def _(val: dict[Any, Any]) -> str:
    return f"{{{len(val)} items}}"


@short_format.register
def _(val: str) -> str:
    if len(val) > 10:
        return f"{val[:10]}..."
    return val
