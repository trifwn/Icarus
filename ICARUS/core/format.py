from functools import singledispatch


@singledispatch
def short_format(val):
    return str(val)


@short_format.register
def _(val: float):
    return f"{val:.3f}"


@short_format.register
def _(val: list):
    return f"[{len(val)} items]"


@short_format.register
def _(val: dict):
    return f"{{{len(val)} items}}"


@short_format.register
def _(val: str):
    if len(val) > 10:
        return f"{val[:10]}..."
    return val
