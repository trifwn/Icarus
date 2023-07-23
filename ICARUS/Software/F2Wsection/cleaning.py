import os

def remove_results(CASEDIR: str, HOMEDIR: str, angles: list[float]) -> None:
    """
    Removes Simulation results for a given case
    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        angles (list[float]): Angles to remove
    """
    try:
        os.chdir(CASEDIR)
        os.remove("SOLOUTI*")
        os.remove("*.out")
        os.remove("PAKETO")
    except FileNotFoundError:
        pass
    parent_directory: str = os.getcwd()
    folders: list[str] = next(os.walk("."))[1]
    for angle in angles:
        if angle >= 0:
            folder: str = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        if folder[:-1] in folders:
            os.chdir(folder)
            os.remove(
                "AERLOAD.OUT AIRFOIL.OUT BDLAYER.OUT COEFPRE.OUT SEPWAKE.OUT TREWAKE.OUT clcd.out SOLOUTI.INI",
            )
            os.chdir(parent_directory)
    os.chdir(HOMEDIR)