from pathlib import Path

from ICARUS.database.utils import angle_to_directory


def remove_results(casedir: str, angles: list[float]) -> None:
    """Removes simulation results for a given case.

    Args:
        casedir (str): Path to the case directory.
        angles (list[float]): List of angles (float) used to determine folders to clean.
    """
    case_path = Path(casedir)

    # Remove top-level files
    for pattern in ["SOLOUTI*", "*.out", "PAKETO"]:
        for file in case_path.glob(pattern):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to remove {file}: {e}")

    # Get list of subfolders in case directory
    folders = {f.name for f in case_path.iterdir() if f.is_dir()}

    for angle in angles:
        folder_name = angle_to_directory(angle=angle)
        trimmed_name = folder_name.rstrip("/")

        if trimmed_name in folders:
            folder_path = case_path / trimmed_name
            files_to_remove = [
                "AERLOAD.OUT",
                "AIRFOIL.OUT",
                "BDLAYER.OUT",
                "COEFPRE.OUT",
                "SEPWAKE.OUT",
                "TREWAKE.OUT",
                "clcd.out",
                "SOLOUTI.INI",
            ]
            for filename in files_to_remove:
                file_path = folder_path / filename
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Warning: Failed to remove {file_path}: {e}")
