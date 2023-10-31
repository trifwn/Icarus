import os
import re
from typing import Optional

from ICARUS.Core.file_tail import tail


def latest_time(REYNDIR: str, name: str) -> tuple[Optional[int], Optional[float], bool]:
    """Get the latest iteration of F2W

    Args:
        REYNDIR (str): Directory where it is run
        name (str): pos.out or neg.out depending on run

    Returns:
        Tuple[Optional[int], Optional[float], bool]: Tuple containing IBLM iteration, the angle
                                                    where the simulation is, and an error flag.
    """

    def get_angle() -> Optional[float]:
        folders: list[str] = next(os.walk(REYNDIR))[1]
        angles: list[float] = []
        angle: float = 0
        for folder in folders:
            if name == "pos.out" and folder.startswith("m"):
                continue
            elif name == "pos.out" and not folder.startswith("m"):
                angle = float(folder)

            if name == "neg.out" and not folder.startswith("m"):
                continue
            elif name == "neg.out" and folder.startswith("m"):
                angle = float(folder[1:])
            angles.append(angle)

        if len(angles) == 0:
            return None
        return max(angles)

    filename: str = os.path.join(REYNDIR, name)
    try:
        with open(filename, "rb") as f:
            data_b: list[bytes] = tail(f, 300)
        data: list[str] = [line.decode() for line in data_b]
    except FileNotFoundError:
        return None, None, False

    # ANGLE
    angle: float | None = get_angle()
    # ERROR
    error: bool = any(re.search(r"forrtl", x) for x in data)

    # ITERATION
    try:
        times: list[int] = [int(x[9:]) for x in data if re.search(r"^  NTIME", x)]
        latest_t: int = max(times)
        return latest_t, angle, error
    except ValueError:
        return None, angle, error
