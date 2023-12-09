import os
import shutil


def clean_open_foam(HOMEDIR: str, CASEDIR: str) -> None:
    """Function to clean OpenFoam results

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    os.chdir(CASEDIR)
    for folder in next(os.walk("."))[1]:
        if folder.startswith("m") or folder.startswith(
            ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
        ):
            os.chdir(folder)
            iteration_folder: list[str] = next(os.walk("."))[1]
            iteration_num: list[int] = [
                int(iteration_folder[j]) for j in range(len(iteration_folder)) if iteration_folder[j].isdigit()
            ]
            iteration_num = sorted(iteration_num)
            iteration_folder = [str(i) for i in iteration_num]
            for folder_to_delete in iteration_folder[1:-1]:
                shutil.rmtree(folder_to_delete)
            os.chdir(CASEDIR)
    os.chdir(HOMEDIR)
