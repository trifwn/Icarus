import os

from ICARUS.database.database2D import Database_2D


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
        folder: str = Database_2D.angle_to_dir(angle=angle)

        if folder[:-1] in folders:
            os.chdir(folder)
            os.remove(
                "AERLOAD.OUT AIRFOIL.OUT BDLAYER.OUT COEFPRE.OUT SEPWAKE.OUT TREWAKE.OUT clcd.out SOLOUTI.INI",
            )
            os.chdir(parent_directory)
    os.chdir(HOMEDIR)
