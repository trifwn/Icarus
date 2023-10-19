import os


def get_coefficients(angle: float) -> str | None:
    """Function to get coefficients from OpenFoam results for a given angle.
    
    Args:
        angle (float): Angle for which coefficients are required
    Returns:
        str | None: String Containing Coefficients or None if not found
    """
    if angle >= 0:
        folder: str = str(angle)[::-1].zfill(7)[::-1]
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
    parentDir: str = os.getcwd()
    folders: list[str] = next(os.walk("."))[1]
    if folder[:-1] in folders:
        os.chdir(folder)
        folders = next(os.walk("."))[1]
    if "postProcessing" in folders:
        os.chdir(os.path.join("postProcessing", "force_coefs"))
        times: list[str] = next(os.walk("."))[1]
        times_num = [int(times[j]) for j in range(len(times)) if times[j].isdigit()]
        latestTime = max(times_num)
        os.chdir(str(latestTime))
        filen = "coefficient.dat"
        with open(filen, encoding="UTF-8", newline="\n") as file:
            data: list[str] = file.readlines()
        os.chdir(parentDir)
    else:
        os.chdir(parentDir)
        return None
    return data[-1]
