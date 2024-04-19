import os
import re

from ICARUS.core.file_tail import tail


def latest_time(
    CASEDIR: str,
    genu_version: int,
) -> tuple[int | None, bool]:
    """Get the latest iteration of GNVP3

    Args:
        CASEDIR (str): Directory where it is run
        genu_version (int): Version of GNVP

    Returns:
        Tuple[Optional[int], Optional[float], bool]: Tuple containing IBLM iteration, the angle where the simulation is, and an error flag.
    """

    filename: str = os.path.join(CASEDIR, f"gnvp{genu_version}.out")
    try:
        with open(filename, "rb") as f:
            data_b: list[bytes] = tail(f, 300)
        data: list[str] = [line.decode() for line in data_b]
    except FileNotFoundError:
        # print(f"File {filename} not found")
        return None, False

    # ERROR
    error: bool = any(re.search(r"forrtl", x) for x in data)

    # ITERATION
    matches: list[str] = [x.split("   ")[1] for x in data if re.search(r"NTIME =", x)]
    times: list[int] = [int(x) for x in matches]
    try:
        latest_t: int = max(times)
        return latest_t, error
    except ValueError:
        # print(times, CASEDIR)
        return None, error
