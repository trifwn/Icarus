import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from typing import Union
from typing import cast

from pandas import DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AVLParseError(Exception):
    """Custom exception for AVL parsing errors"""

    pass


class AVLSectionType(Enum):
    """Enum for different AVL output sections"""

    HEADER = "header"
    CONFIGURATION = "configuration"
    FORCES = "forces"
    EIGENVALUES = "eigenvalues"
    STABILITY = "stability"


@dataclass
class AVLConfiguration:
    """Data class for AVL configuration parameters"""

    name: str
    surfaces: int
    strips: int
    vortices: int
    sref: float
    cref: float
    bref: float
    xref: float
    yref: float
    zref: float


@dataclass
class AVLForceResults:
    """Data class for AVL force and moment results"""

    alpha: float
    beta: float
    mach: float
    pb_2v: float
    qc_2v: float
    rb_2v: float
    cxtot: float
    cytot: float
    cztot: float
    cltot: float
    cmtot: float
    cntot: float
    cltotal: float
    cdtot: float
    cdvis: float
    cdind: float
    efficiency: float


@dataclass
class AVLEigenmode:
    """Data class for eigenmode results"""

    mode_number: int
    eigenvalue: complex
    eigenvector: dict[str, complex]


class AVLOutputParser:
    """Production-ready parser for AVL output files"""

    def __init__(self, file_path: str) -> None:
        """Initialize parser with file path

        Args:
            file_path: Path to AVL output file
        """
        self.file_path = file_path
        self.content = self._read_file()
        self.content = self._clean_content(self.content)
        self.lines = self.content.splitlines()

    def _clean_content(self, content: str) -> str:
        """Remove lines that are notes or warnings (e.g., starting with 'Note:')"""
        cleaned_lines = []
        for line in content.splitlines():
            if line.strip().startswith("Note:"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _read_file(self) -> str:
        """Read file with proper error handling"""
        try:
            with open(self.file_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise AVLParseError(f"File not found: {self.file_path}")
        except Exception as e:
            raise AVLParseError(f"Error reading file {self.file_path}: {e}")

    def _find_section(
        self,
        pattern: str,
        start_line: int = 0,
        case_insensitive: bool = True,
    ) -> Optional[int]:
        """Find a section by pattern matching (case-insensitive by default)

        Args:
            pattern: Regex pattern to search for
            start_line: Line number to start searching from
            case_insensitive: Whether to ignore case

        Returns:
            Line number where pattern is found, or None if not found
        """
        flags = re.IGNORECASE if case_insensitive else 0
        compiled_pattern = re.compile(pattern, flags)
        for i, line in enumerate(self.lines[start_line:], start_line):
            if compiled_pattern.search(line):
                return i
        return None

    def _extract_float(self, line: str, pattern: str) -> Optional[float]:
        """Extract float value using regex pattern

        Args:
            line: Line to search in
            pattern: Regex pattern with one capture group

        Returns:
            Extracted float value or None if not found
        """
        match = re.search(pattern, line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                logger.warning(f"Could not convert '{match.group(1)}' to float")
                return None
        return None

    def _extract_int(self, line: str, pattern: str) -> Optional[int]:
        """Extract integer value using regex pattern

        Args:
            line: Line to search in
            pattern: Regex pattern with one capture group

        Returns:
            Extracted integer value or None if not found
        """
        match = re.search(pattern, line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                logger.warning(f"Could not convert '{match.group(1)}' to int")
                return None
        return None

    def parse_configuration(self) -> AVLConfiguration:
        """Parse AVL configuration section, robust to format variations"""
        # Find configuration section
        config_line = self._find_section(r"Configuration:\s*(.+)")
        if config_line is None:
            raise AVLParseError("Configuration section not found")

        config_name = re.search(r"Configuration:\s*(.+)", self.lines[config_line])
        if not config_name:
            raise AVLParseError("Configuration name not found")

        # Find reference quantities section (allow for whitespace and case)
        ref_line = self._find_section(r"Sref\s*=", config_line)
        if ref_line is None:
            raise AVLParseError("Reference quantities section not found")

        # Parse reference quantities
        sref = self._extract_float(self.lines[ref_line], r"Sref\s*=\s*([\d.eE+-]+)")
        cref = self._extract_float(self.lines[ref_line], r"Cref\s*=\s*([\d.eE+-]+)")
        bref = self._extract_float(self.lines[ref_line], r"Bref\s*=\s*([\d.eE+-]+)")

        # Parse reference position (next line, or same line)
        xref = yref = zref = None
        if ref_line + 1 < len(self.lines):
            xref = self._extract_float(
                self.lines[ref_line + 1],
                r"Xref\s*=\s*([\d.eE+-]+)",
            )
            yref = self._extract_float(
                self.lines[ref_line + 1],
                r"Yref\s*=\s*([\d.eE+-]+)",
            )
            zref = self._extract_float(
                self.lines[ref_line + 1],
                r"Zref\s*=\s*([\d.eE+-]+)",
            )
        if xref is None or yref is None or zref is None:
            # Try to find in same line as Sref
            xref = xref or self._extract_float(
                self.lines[ref_line],
                r"Xref\s*=\s*([\d.eE+-]+)",
            )
            yref = yref or self._extract_float(
                self.lines[ref_line],
                r"Yref\s*=\s*([\d.eE+-]+)",
            )
            zref = zref or self._extract_float(
                self.lines[ref_line],
                r"Zref\s*=\s*([\d.eE+-]+)",
            )

        # Parse surfaces, strips, vortices (search up to 10 lines before Sref)
        surfaces = strips = vortices = None
        for i in range(max(0, ref_line - 10), ref_line + 1):
            if surfaces is None:
                surfaces = self._extract_int(self.lines[i], r"#\s*Surfaces\s*=\s*(\d+)")
            if strips is None:
                strips = self._extract_int(self.lines[i], r"#\s*Strips\s*=\s*(\d+)")
            if vortices is None:
                vortices = self._extract_int(self.lines[i], r"#\s*Vortices\s*=\s*(\d+)")

        # Fallback: search next 10 lines after Sref
        if surfaces is None or strips is None or vortices is None:
            for i in range(ref_line, min(len(self.lines), ref_line + 10)):
                if surfaces is None:
                    surfaces = self._extract_int(
                        self.lines[i],
                        r"#\s*Surfaces\s*=\s*(\d+)",
                    )
                if strips is None:
                    strips = self._extract_int(self.lines[i], r"#\s*Strips\s*=\s*(\d+)")
                if vortices is None:
                    vortices = self._extract_int(
                        self.lines[i],
                        r"#\s*Vortices\s*=\s*(\d+)",
                    )

        # Log if any are still missing
        if surfaces is None or strips is None or vortices is None:
            logger.warning(
                f"Could not find all surface/strip/vortex counts near line {ref_line}",
            )

        return AVLConfiguration(
            name=config_name.group(1).strip(),
            surfaces=surfaces or 0,
            strips=strips or 0,
            vortices=vortices or 0,
            sref=sref or 0.0,
            cref=cref or 0.0,
            bref=bref or 0.0,
            xref=xref or 0.0,
            yref=yref or 0.0,
            zref=zref or 0.0,
        )

    def parse_forces(self) -> list[AVLForceResults]:
        """Parse force and moment results from AVL output, robust to format variations"""
        force_results: list[AVLForceResults] = []

        # Find all force output sections (case-insensitive, allow for extra dashes/whitespace)
        force_sections = []
        start_line = 0
        while True:
            section_line = self._find_section(
                r"Vortex Lattice Output\s*-+\s*Total Forces",
                start_line,
            )
            if section_line is None:
                break
            force_sections.append(section_line)
            start_line = section_line + 1

        if not force_sections:
            logger.warning("No force sections found")
            return force_results

        for section_start in force_sections:
            try:
                # Find run case line (allow for whitespace)
                run_case_line = self._find_section(r"Run case\s*:", section_start)
                if run_case_line is None:
                    logger.warning(
                        f"No 'Run case:' found after force section at line {section_start}",
                    )
                    continue

                # Extract flight conditions
                alpha = beta = mach = pb_2v = qc_2v = rb_2v = 0.0

                # Look for flight condition lines (robust to order and whitespace)
                for i in range(
                    run_case_line + 1,
                    min(len(self.lines), run_case_line + 15),
                ):
                    line = self.lines[i]
                    if re.search(r"Alpha", line, re.IGNORECASE):
                        alpha = (
                            self._extract_float(line, r"Alpha\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                    if re.search(r"Beta", line, re.IGNORECASE):
                        beta = (
                            self._extract_float(line, r"Beta\s*=\s*([\d.eE+-]+)") or 0.0
                        )
                    if re.search(r"Mach", line, re.IGNORECASE):
                        mach = (
                            self._extract_float(line, r"Mach\s*=\s*([\d.eE+-]+)") or 0.0
                        )
                    if re.search(r"pb/2V", line, re.IGNORECASE):
                        pb_2v = (
                            self._extract_float(line, r"pb/2V\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                    if re.search(r"qc/2V", line, re.IGNORECASE):
                        qc_2v = (
                            self._extract_float(line, r"qc/2V\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                    if re.search(r"rb/2V", line, re.IGNORECASE):
                        rb_2v = (
                            self._extract_float(line, r"rb/2V\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )

                # Extract force and moment coefficients (robust to order and whitespace)
                cxtot = cytot = cztot = cltot = cmtot = cntot = 0.0
                cltotal = cdtot = cdvis = cdind = efficiency = 0.0
                found_any = False
                for i in range(
                    run_case_line + 1,
                    min(len(self.lines), run_case_line + 30),
                ):
                    line = self.lines[i]
                    if re.search(r"CXtot", line, re.IGNORECASE):
                        cxtot = (
                            self._extract_float(line, r"CXtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CYtot", line, re.IGNORECASE):
                        cytot = (
                            self._extract_float(line, r"CYtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CZtot", line, re.IGNORECASE):
                        cztot = (
                            self._extract_float(line, r"CZtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"Cltot", line, re.IGNORECASE):
                        cltot = (
                            self._extract_float(line, r"Cltot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"Cmtot", line, re.IGNORECASE):
                        cmtot = (
                            self._extract_float(line, r"Cmtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"Cntot", line, re.IGNORECASE):
                        cntot = (
                            self._extract_float(line, r"Cntot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CLtot", line, re.IGNORECASE):
                        cltotal = (
                            self._extract_float(line, r"CLtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CDtot", line, re.IGNORECASE):
                        cdtot = (
                            self._extract_float(line, r"CDtot\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CDvis", line, re.IGNORECASE):
                        cdvis = (
                            self._extract_float(line, r"CDvis\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"CDind", line, re.IGNORECASE):
                        cdind = (
                            self._extract_float(line, r"CDind\s*=\s*([\d.eE+-]+)")
                            or 0.0
                        )
                        found_any = True
                    if re.search(r"e\s*=", line, re.IGNORECASE):
                        efficiency = (
                            self._extract_float(line, r"e\s*=\s*([\d.eE+-]+)") or 0.0
                        )
                        found_any = True

                if not found_any:
                    logger.warning(
                        f"No force/moment coefficients found after run case at line {run_case_line}",
                    )

                force_results.append(
                    AVLForceResults(
                        alpha=alpha,
                        beta=beta,
                        mach=mach,
                        pb_2v=pb_2v,
                        qc_2v=qc_2v,
                        rb_2v=rb_2v,
                        cxtot=cxtot,
                        cytot=cytot,
                        cztot=cztot,
                        cltot=cltot,
                        cmtot=cmtot,
                        cntot=cntot,
                        cltotal=cltotal,
                        cdtot=cdtot,
                        cdvis=cdvis,
                        cdind=cdind,
                        efficiency=efficiency,
                    ),
                )

            except Exception as e:
                logger.warning(
                    f"Error parsing force section starting at line {section_start}: {e}",
                )
                continue

        return force_results

    def parse_eigenmodes(self) -> list[AVLEigenmode]:
        """Parse eigenmode results from AVL output, robust to format variations and all eigenvector components"""
        eigenmodes: list[AVLEigenmode] = []

        # Find all eigenmode section headers (case-insensitive, allow for whitespace)
        mode_lines = []
        for i, line in enumerate(self.lines):
            if re.search(
                r"mode\s+\d+\s*:\s*([\d.eE+-]+)\s+([\d.eE+-]+)",
                line,
                re.IGNORECASE,
            ):
                mode_lines.append(i)

        if not mode_lines:
            logger.warning("No eigenmode section found")
            return eigenmodes

        for idx, mode_line in enumerate(mode_lines):
            line = self.lines[mode_line]
            mode_match = re.search(
                r"mode\s+(\d+)\s*:\s*([\d.eE+-]+)\s+([\d.eE+-]+)",
                line,
                re.IGNORECASE,
            )
            if mode_match:
                mode_num = int(mode_match.group(1))
                real_part = float(mode_match.group(2))
                imag_part = float(mode_match.group(3))
                eigenvalue = complex(real_part, imag_part)

                # Dynamically parse all eigenvector components after the mode line
                eigenvector = {}
                j = 1
                while (mode_line + j) < len(self.lines):
                    var_line = self.lines[mode_line + j].strip()
                    # Find all var: real imag groups in the line
                    matches = re.findall(
                        r"([\w\d_]+)\s*:\s*([\-\d.eE+]+)\s+([\-\d.eE+]+)",
                        var_line,
                    )
                    if matches:
                        for var, real_str, imag_str in matches:
                            try:
                                real_val = float(real_str)
                                imag_val = float(imag_str)
                                eigenvector[var] = complex(real_val, imag_val)
                            except Exception as e:
                                logger.debug(
                                    f"Could not parse eigenvector component '{var}' in line: {var_line} ({e})",
                                )
                        j += 1
                    else:
                        break

                eigenmodes.append(
                    AVLEigenmode(
                        mode_number=mode_num,
                        eigenvalue=eigenvalue,
                        eigenvector=eigenvector,
                    ),
                )
            else:
                logger.debug(f"Could not parse eigenmode line: {line}")

        return eigenmodes

    @staticmethod
    def to_dataframe(force_results: list[AVLForceResults]) -> DataFrame:
        """Convert force results to pandas DataFrame

        Args:
            force_results: list of AVLForceResults objects

        Returns:
            DataFrame with force and moment data
        """
        if not force_results:
            return DataFrame()

        data = []
        for result in force_results:
            data.append(
                {
                    "Alpha": result.alpha,
                    "Beta": result.beta,
                    "Mach": result.mach,
                    "pb/2V": result.pb_2v,
                    "qc/2V": result.qc_2v,
                    "rb/2V": result.rb_2v,
                    "CX": result.cxtot,
                    "CY": result.cytot,
                    "CZ": result.cztot,
                    "Cl": result.cltot,
                    "Cm": result.cmtot,
                    "Cn": result.cntot,
                    "CL": result.cltotal,
                    "CD": result.cdtot,
                    "CDvis": result.cdvis,
                    "CDind": result.cdind,
                    "e": result.efficiency,
                },
            )

        return DataFrame(data)


def parse_avl_output(
    file_path: str,
) -> dict[str, Union[AVLConfiguration, list[AVLForceResults], list[AVLEigenmode]]]:
    """Main function to parse AVL output file

    Args:
        file_path: Path to AVL output file

    Returns:
        Dictionary containing parsed configuration, forces, and eigenmodes
    """
    parser = AVLOutputParser(file_path)

    try:
        config = parser.parse_configuration()
        forces = parser.parse_forces()
        eigenmodes = parser.parse_eigenmodes()

        return {"configuration": config, "forces": forces, "eigenmodes": eigenmodes}
    except Exception as e:
        logger.error(f"Error parsing AVL output: {e}")
        raise AVLParseError(f"Failed to parse AVL output: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the parser
    try:
        # Parse the file
        results = parse_avl_output("paste.txt")

        config = results.get("configuration")
        forces = results.get("forces")
        eigenmodes = results.get("eigenmodes")

        # Access configuration
        if isinstance(config, AVLConfiguration):
            print(f"Configuration: {config.name}")
            print(
                f"Surfaces: {config.surfaces}, Strips: {config.strips}, Vortices: {config.vortices}",
            )
        else:
            print("No configuration found or wrong type.")

        # Access force results
        valid_forces: Optional[list[AVLForceResults]] = None
        if (
            isinstance(forces, list)
            and forces
            and all(isinstance(f, AVLForceResults) for f in forces)
        ):
            valid_forces = cast(list[AVLForceResults], forces)
            print(f"Found {len(valid_forces)} force results")
            for i, force in enumerate(valid_forces):
                print(
                    f"Result {i + 1}: Alpha={force.alpha:.3f}, CL={force.cltotal:.6f}, CD={force.cdtot:.6f}",
                )
        else:
            print("No force results found or wrong type.")

        valid_modes: Optional[list[AVLEigenmode]] = None
        if (
            isinstance(eigenmodes, list)
            and eigenmodes
            and all(isinstance(m, AVLEigenmode) for m in eigenmodes)
        ):
            valid_modes = cast(list[AVLEigenmode], eigenmodes)
            print(f"Found {len(valid_modes)} eigenmodes")
            for mode in valid_modes:
                print(f"Mode {mode.mode_number}: {mode.eigenvalue}")
                for eigenvar, value in mode.eigenvector.items():
                    print(f"  {eigenvar}: {value}")
        else:
            print("No eigenmodes found or wrong type.")

        # Convert to DataFrame
        parser = AVLOutputParser("paste.txt")
        if valid_forces is not None:
            df = parser.to_dataframe(valid_forces)
            print("\nDataFrame shape:", df.shape)
            print(df.head())
        else:
            print("No valid force results to convert to DataFrame.")

    except AVLParseError as e:
        print(f"AVL Parse Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
