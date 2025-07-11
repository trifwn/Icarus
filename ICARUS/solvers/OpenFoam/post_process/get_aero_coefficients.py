from __future__ import annotations

import os

from ICARUS.database.utils import angle_to_directory


def get_coefficients(case_directory: str, angle: float) -> str | None:
    """Function to get coefficients from OpenFoam results for a given angle.

    Args:
        angle (float): Angle for which coefficients are required
    Returns:
        str | None: String Containing Coefficients or None if not found

    """
    folder = angle_to_directory(angle)

    folders: list[str] = next(os.walk(case_directory))[1]
    if folder in folders:
        folders = next(os.walk(os.path.join(case_directory, folder)))[1]

    if "postProcessing" in folders:
        forces_dir = os.path.join(case_directory, "postProcessing", "force_coefs")
        times: list[str] = next(os.walk(forces_dir))[1]
        times_num = [int(times[j]) for j in range(len(times)) if times[j].isdigit()]
        latestTime = max(times_num)
        latestTime_dir = os.path.join(forces_dir, str(latestTime))
        filen = os.path.join(latestTime_dir, "coefficient.dat")
        with open(filen, encoding="UTF-8", newline="\n") as file:
            data: list[str] = file.readlines()
    else:
        return None
    return data[-1]
