import shutil
from pathlib import Path


def clean_open_foam(casedir: str) -> None:
    """Cleans OpenFOAM results by removing intermediate iteration folders.

    Args:
        casedir (str): Path to the OpenFOAM case directory.
    """
    case_path = Path(casedir)

    for folder in case_path.iterdir():
        if folder.is_dir() and (folder.name.startswith("m") or folder.name[:1].isdigit()):
            subdirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()]
            iteration_nums = sorted(int(d.name) for d in subdirs)
            iteration_folders = [folder / str(n) for n in iteration_nums[1:-1]]  # Keep first and last

            for subfolder in iteration_folders:
                try:
                    shutil.rmtree(subfolder)
                except Exception as e:
                    print(f"Warning: Failed to remove {subfolder}: {e}")
