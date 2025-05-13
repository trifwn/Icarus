import os
import re
from collections import defaultdict

import pandas as pd

from ICARUS.database import Database
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.airplane import Airplane


# Helper function to try converting a string to float.
def to_number(token: str):
    try:
        return float(token)
    except ValueError:
        return token


def get_strip_data(plane: Airplane, state: State, case: str) -> pd.DataFrame:
    DB = Database.get_instance()
    directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )
    files: list[str] = os.listdir(directory)
    for file in files:
        # Assume directory and file are defined
        if not file.startswith("fs"):
            continue

        if not file.endswith(f"{case}.txt"):
            continue

        # Build the filename (make sure directory and file are defined)
        filename: str = os.path.join(directory, file)
        print(f"Reading AVL strip data from {filename}")
        with open(filename, encoding="UTF-8") as f:
            data: list[str] = f.readlines()

        surfaces = {}  # Dictionary to store surface_name: DataFrame
        surface_counts = defaultdict(int)  # To make duplicate surface names unique
        current_surface = None  # Current surface key (e.g., "wing" or "wing_2")
        table_columns = None  # Column names for the current table
        table_rows = []  # Data rows for the current table
        inside_table = False  # Flag for whether we are reading table rows

        for line in data:
            # Remove the leading and trailing whitespace.
            line = line.strip()

            # Detect the beginning of a new surface block.
            if line.startswith("Surface #"):
                # If a previous surface block exists, save its DataFrame.
                if current_surface is not None and table_rows and table_columns:
                    df = pd.DataFrame(table_rows, columns=table_columns)
                    if current_surface not in surfaces.keys():
                        surfaces[current_surface] = df
                    else:
                        surfaces[current_surface] = pd.concat([surfaces[current_surface], df])
                    table_rows = []
                    table_columns = None
                    inside_table = False

                # Use regex to capture the surface name.
                # Example lines:
                #   "Surface # 1     wing                 | surface name stri"
                #   "Surface # 3     elevator             | surface name "
                m = re.search(r"Surface\s+#\s+\d+\s+(\S+)", line)
                if m:
                    base_name = m.group(1)
                    surface_counts[base_name] += 1
                    # If the same surface name appears more than once, append a counter.
                    if surface_counts[base_name] > 1:
                        current_surface = f"{base_name}"
                    else:
                        current_surface = base_name
                else:
                    current_surface = "unknown"
                continue

            # Look for the start of the strip forces table.
            if "Strip Forces referred to Strip Area, Chord" in line:
                inside_table = False  # Reset flag; header row comes next.
                continue

            # Look for the header row of the table. It should start with "j".
            if line.startswith("j     Xle      Yle      Zle"):
                table_columns = line.split()
                inside_table = True
                continue

            # If we're inside the table, process each row.
            if inside_table:
                # Stop reading if we hit an empty or a separator line.
                if not line or line.startswith("-"):
                    inside_table = False
                    continue
                # Split the row into tokens and convert numbers where possible.
                tokens = line.split()
                row = [to_number(token) for token in tokens]
                table_rows.append(row)

        # At the end, save any remaining table.
        if current_surface is not None and table_rows and table_columns:
            df = pd.DataFrame(table_rows, columns=table_columns)
            if current_surface not in surfaces.keys():
                surfaces[current_surface] = df
            else:
                surfaces[current_surface] = pd.concat([surfaces[current_surface], df])

    for key, df in surfaces.items():
        # Sort by j and reset the index.
        surfaces[key] = df.sort_values(by="j").reset_index(drop=True)

    master_df = pd.concat(surfaces.values(), keys=list(surfaces.keys()), names=["surface_name"])
    # ('wing', np.int64(0)) -> 'wing'
    # ('wing', np.int64(1)) -> 'wing_1'
    master_df.index = master_df.index.get_level_values(0)

    return master_df


strip_cols_AVL = [
    "j",
    "Xle",
    "Yle",
    "Zle",
    "Chord",
    "Area",
    "c_cl",
    "ai",
    "cl_norm",
    "cl",
    "cd",
    "cdv",
    "cm_c/4 ",
    "cm_LE",
    "C.P.x/c",
]
