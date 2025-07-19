import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

logger = logging.getLogger(__name__)


class AVLParseError(Exception):
    """Custom exception for AVL parsing errors"""

    pass


@dataclass
class SurfaceData:
    """Container for parsed surface data"""

    name: str
    number: int
    data: pd.DataFrame


class AVLStripDataParser:
    """Streamlined parser for AVL strip data files"""

    # Regex patterns
    SURFACE_PATTERN = re.compile(r"Surface #\s*(\d+)\s+(\S.*?)(?:\s*\|.*)?$")
    HEADER_PATTERN = re.compile(r"^\s*j\s+Xle\s+Yle\s+Zle")

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.surfaces: Dict[str, SurfaceData] = {}
        self._surface_counts: Dict[str, int] = {}

    def _clean_surface_name(self, raw_name: str) -> str:
        """Clean and ensure unique surface names"""
        # Remove everything after pipe and clean whitespace
        name = raw_name.split("|")[0].strip()
        name = re.sub(r"\s+", "_", name)

        # Handle duplicates
        count = self._surface_counts.get(name, 0)
        self._surface_counts[name] = count + 1

        return f"{name}_{count}" if count > 0 else name

    def _convert_numeric(self, value: str) -> Union[float, str]:
        """Convert string to float if possible"""
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return value

    def _parse_surface_header(self, line: str) -> tuple[str, int]:
        """Extract surface name and number from header line"""
        match = self.SURFACE_PATTERN.match(line)
        if not match:
            raise AVLParseError(f"Invalid surface header: {line}")

        number = int(match.group(1))
        name = self._clean_surface_name(match.group(2))
        return name, number

    def _process_data_row(
        self,
        line: str,
        columns: List[str],
    ) -> Optional[List[Union[str, float]]]:
        """Process a data row, returning None if invalid"""
        tokens = line.split()
        if not tokens:
            return None

        row = [self._convert_numeric(token) for token in tokens]

        # Validate row length
        if len(row) != len(columns):
            if self.strict_mode:
                raise AVLParseError(
                    f"Row length mismatch: expected {len(columns)}, got {len(row)}",
                )
            logger.warning(f"Skipping malformed row: {line}")
            return None

        return row

    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, SurfaceData]:
        """Parse AVL strip data file"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise AVLParseError(f"File not found: {filepath}")

        # Read file with encoding fallback
        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = filepath.read_text(encoding="latin-1")

        lines = content.splitlines()

        # Reset state
        self.surfaces = {}
        self._surface_counts = {}

        current_surface = None
        current_number = None
        columns = None
        rows: List[List[float | str]] = []
        in_table = False

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # New surface
                if line.startswith("Surface #"):
                    # Save previous surface
                    if current_surface and current_number and columns and rows:
                        self._save_surface(
                            current_surface,
                            current_number,
                            columns,
                            rows,
                        )

                    # Start new surface
                    current_surface, current_number = self._parse_surface_header(line)
                    columns = None
                    rows = []
                    in_table = False
                    continue

                # Table header
                if self.HEADER_PATTERN.match(line):
                    columns = line.split()
                    in_table = True
                    continue

                # Data rows
                if in_table and columns:
                    # Check for table end
                    if line.startswith("-") or "Strip Forces" in line:
                        in_table = False
                        continue

                    # Process data row
                    row = self._process_data_row(line, columns)
                    if row:
                        rows.append(row)

            except AVLParseError:
                raise
            except Exception as e:
                error_msg = f"Error at line {line_num}: {e}"
                if self.strict_mode:
                    raise AVLParseError(error_msg)
                logger.warning(error_msg)

        # Save final surface
        if current_surface and current_number and columns and rows:
            self._save_surface(current_surface, current_number, columns, rows)

        if not self.surfaces:
            raise AVLParseError("No valid surfaces found")

        logger.debug(f"Parsed {len(self.surfaces)} surfaces from {filepath}")
        return self.surfaces

    def _save_surface(
        self,
        name: str,
        number: int,
        columns: List[str],
        rows: List[List],
    ) -> None:
        """Save surface data to internal storage"""
        try:
            df = pd.DataFrame(rows, columns=columns)

            # Sort by strip number if available
            if "j" in df.columns:
                df = df.sort_values("j").reset_index(drop=True)

            # Combine with existing data if duplicate surface
            if name in self.surfaces:
                existing_df = self.surfaces[name].data
                df = pd.concat([existing_df, df], ignore_index=True)
                self.surfaces[name].data = df
            else:
                self.surfaces[name] = SurfaceData(name=name, number=number, data=df)

        except Exception as e:
            raise AVLParseError(f"Failed to create DataFrame for surface {name}: {e}")

    def create_master_dataframe(self) -> pd.DataFrame:
        """Create combined DataFrame with all surfaces"""
        if not self.surfaces:
            return pd.DataFrame()

        # Sort by surface number
        sorted_surfaces = sorted(self.surfaces.values(), key=lambda s: s.number)

        # Combine all surface DataFrames
        dfs = []
        surface_names = []

        for surface in sorted_surfaces:
            dfs.append(surface.data)
            surface_names.extend([surface.name] * len(surface.data))

        if not dfs or not surface_names:
            return pd.DataFrame()

        master_df = pd.concat(dfs, ignore_index=True)
        master_df.index = pd.Index(surface_names)
        master_df.index.name = "surface_name"

        return master_df


def get_strip_data(
    plane: Airplane,
    state: State,
    case: str,
    strict_mode: bool = False,
) -> pd.DataFrame:
    """
    Get strip data from AVL files for given aircraft configuration

    Args:
        plane: Airplane object
        state: State object
        case: Case identifier
        strict_mode: Whether to raise exceptions on warnings

    Returns:
        Combined DataFrame with strip data from all matching files

    Raises:
        AVLParseError: If parsing fails
        FileNotFoundError: If no matching files found
    """
    # Get directory from database
    DB = Database.get_instance()
    directory = Path(
        DB.get_vehicle_case_directory(
            airplane=plane,
            state=state,
            solver="AVL",
        ),
    )

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find matching files
    pattern = f"fs*{case}.txt"
    matching_files = list(directory.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

    logger.debug(f"Found {len(matching_files)} files matching '{pattern}'")

    # Parse all files
    parser = AVLStripDataParser(strict_mode=strict_mode)
    all_surfaces = {}

    for filepath in matching_files:
        try:
            surfaces = parser.parse_file(filepath)
            all_surfaces.update(surfaces)
            logger.debug(f"Parsed {filepath}: {len(surfaces)} surfaces")
        except AVLParseError as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            if strict_mode:
                raise
            continue

    if not all_surfaces:
        raise AVLParseError("No valid surfaces found in any file")

    # Create master DataFrame
    parser.surfaces = all_surfaces
    master_df = parser.create_master_dataframe()

    if master_df.empty:
        raise AVLParseError("No data found in parsed files")

    logger.debug(
        f"Created master DataFrame: {len(master_df)} rows, {len(all_surfaces)} surfaces",
    )
    return master_df


if __name__ == "__main__":
    # Example usage
    parser = AVLStripDataParser()
    try:
        surfaces = parser.parse_file("paste.txt")
        master_df = parser.create_master_dataframe()
        print(f"Parsed {len(surfaces)} surfaces")
        print(master_df)
    except AVLParseError as e:
        print(f"Parse error: {e}")
