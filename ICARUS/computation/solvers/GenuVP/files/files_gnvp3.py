import os
from io import StringIO
from io import TextIOWrapper
from typing import Any

import numpy as np
from pandas import DataFrame

from ICARUS.data import AirfoilPolars
from ICARUS.core.types import FloatArray
from ICARUS.core.utils import ff2
from ICARUS.core.utils import ff3
from ICARUS.core.utils import ff4
from ICARUS.core.utils import ff5
from ICARUS.database import Database
from ICARUS.database import PolarsNotFoundError

from ..utils.genu_movement import GNVP_Movement
from ..utils.genu_parameters import GenuParameters
from ..utils.genu_surface import GenuSurface


def line(
    value: Any,
    var_name: str,
    description: str,
    file: TextIOWrapper | StringIO,
) -> None:
    # the value should be added on the first 15 columns after that we should have the
    # var_name on the 16 column and then the description on 25 column
    # Depending on the type of value we should use different formatting. We should also
    # complete the empty spaces with spaces

    # value formatting
    if isinstance(value, float):
        value = ff4(value)
    elif isinstance(value, int):
        value = str(value)
    elif isinstance(value, str):
        pass
    else:
        raise ValueError("The value should be a float, int or str")

    # value formatting
    value = value.ljust(15)

    # var_name formatting
    var_name = var_name.ljust(10)

    # final formatting
    line: str = value + var_name + description + "\n"
    file.write(line)


def blankline(f: StringIO | TextIOWrapper) -> None:
    f.write("                                                                <blank>\n")


def file_name(input: str, file_ext: str) -> str:
    """Trim the input to 12 characters so as to be read by fortran

    Args:
        input (str): file name
        file_ext (str): file extension

    Returns:
        str: trimmed file name

    """
    name_len = 12 - len(file_ext)
    if len(input) > name_len:
        input = input[0 : name_len - 1]
    input += f".{file_ext}"

    return input.ljust(12)


def input_file() -> None:
    """Creates the input file for GNVP3"""
    fname: str = "input"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("dfile.yours\n")
        f.write("0\n")
        f.write("1\n")
        f.write("1.     ! NTIMER\n")
        f.write("1.     ! NTIMEHYB\n")
        f.write("1.     ! ITERWAK\n")
        f.write("1.     ! ITERVEL\n")
        f.write("1.\n")
        f.write("1.\n")
        f.write("1.\n")
        f.write("1.\n")
        f.write("1.\n")


def case_file(params: GenuParameters) -> None:
    """Create Dfile for GNVP3

    Args:
        params (GenuParameters): An object containing all parameter values

    """
    fname: str = "dfile.yours"
    with open(fname, "w", encoding="utf-8") as f:
        line(0, "ISTART", "(=0 for a full run, =1 for a rerun)')", f)
        blankline(f)
        blankline(f)
        f.write(
            "Three lines follow where text can be included as HEADER of the main OUTPUT file\n",
        )
        f.write(
            "<------------------  maximum line length  ------------------------------------->\n",
        )
        f.write("SIMULATION---------------------------\n")
        f.write("-------------------------------------\n")
        f.write("-------------------------------------\n")
        blankline(f)
        f.write("Give the names of the OUTPUT files\n")
        f.write("<---------->\n")
        blankline(f)
        line(
            "YOURS.TOT",
            "OFILE",
            "Name of the OUTPUT file containing general  results",
            f,
        )
        line(
            "YOURS.WAK",
            "OWAKE",
            "Name of the OUTPUT file containing   wake   results",
            f,
        )
        line(
            "YOURS.PRE",
            "OPRES",
            "Name of the OUTPUT file containing pressure results",
            f,
        )
        line(
            "YOURS.BAK",
            "RCALL",
            "Name of the OUTPUT file containing backup   results",
            f,
        )
        line("YOURS.SAS", "SUPAS", "Name of the BINARY file containing AS", f)
        line("YOURS.SBU", "SUPBU", "Name of the BINARY file containing BU", f)
        line(
            "YOURS.CHW",
            "CHWAK",
            "Name of the OUTPUT file containing wake-sts results",
            f,
        )
        line(
            "YOURS.LOA",
            "LOADS",
            "Name of the OUTPUT file containing   loads  results",
            f,
        )
        blankline(f)
        blankline(f)
        f.write("Give the general data in FREE format\n")
        blankline(f)
        f.write("a. The BASIC parameters\n")
        blankline(f)
        line(1, "NSYMF", "=1,2,3 (no-symm, axi-symm, Y-symm)", f)
        line(params.nBods, "NBODT", "number of bodies", f)
        line(params.nBlades, "NBLADE", "number of blades", f)
        line(0, "IABSREF", "= 0 for GCS =1 for RCS", f)
        line(1, "IAXISRF", "=1,2,3 gives the axis of rotation if IABSREF=1", f)
        line(0.0, "OMEGAR", "is the rotation speed of the RCS", f)
        f.write("\n")
        f.write("b. The TIME parameters\n")
        blankline(f)
        line(
            params.maxiter,
            "NTIMER",
            "number of the last time step to be performed",
            f,
        )
        line(params.timestep, "DT", "time step", f)
        line(0, "IDT", "if IDT=1 then DT is the number of steps per rotation", f)
        line(1, "OMEAGT", "the rotation speed for the definition of the PERIOD", f)
        line(
            params.NMETH,
            "NMETHT",
            "=1 for Euler =2 for Adams Bashford time integrat. scheme",
            f,
        )
        line(
            params.NEMTIP,
            "NEMTIP",
            "=0,1. The latter means that tip-emission takes place",
            f,
        )
        line(params.NTIMET, "NTIMET", "time step that tip-emission begins", f)
        line(
            params.NEMSLE,
            "NEMSLE",
            "=0(no action), 1(leading-edge separ. takes place)",
            f,
        )
        line(
            params.NTIMEL,
            "NTIMEL",
            "time step that leading-edge separation starts",
            f,
        )
        line(0.0, "AZIMIN", "the initial azimuthal angle", f)
        blankline(f)
        f.write("c. The SOLUTION parameters\n")
        blankline(f)
        line(0, "IMAT", "=0 AS is calculated, =1 AS is read from disk", f)
        line(200, "ITERM", "maximum number of potential iterations", f)
        line(
            params.RELAXS,
            "RELAXS",
            "relaxation factor for the singularity distributions",
            f,
        )
        line(
            params.EPSDS,
            "EPSDS",
            "convergence tolerance of the potential calculations",
            f,
        )
        blankline(f)
        f.write("d. The MOVEMENT parameters\n")
        blankline(f)
        line(
            params.NLEVELT,
            "NLEVELT",
            "number of movements levels  ( 15 if tail rotor is considered )",
            f,
        )
        blankline(f)
        f.write("e. The FLOW parameters\n")
        blankline(f)
        line(params.u_freestream[0], "UINF(1)", "the velocity at infinity", f)
        line(params.u_freestream[1], "UINF(2)", ".", f)
        line(params.u_freestream[2], "UINF(3)", ".", f)
        line(0, "UREF", "the reference velocity", f)
        line(1, "ADIML", "the length scale used for the non-dimentionalisation", f)
        line(1, "ADIMT", "the  time  scale used for the non-dimentionalisation", f)
        line(0, "IUINFC", "0(no action), 1(UINF varies)", f)
        line(1, "IAXISUI", "=1,2,3 gives the direction of UINF that varies", f)
        line(
            0.00,
            "TIUINF(1)",
            "time parameters of the variation   *** 5 periods ***",
            f,
        )
        line(0.00, "TIUINF(2)", ".shear exponent", f)
        line(0.00, "TIUINF(3)", ".xronos pou arxizei to INWIND", f)
        line(0.00, "TIUINF(4)", ".tower impact factor", f)
        line(1, "TIUINF(5)", ".record pou arxeizei na diabazei", f)
        line(0.000, "AMUINF(1)", "", f)
        line(0.000, "AMUINF(2)", ".", f)
        line(0.000, "AMUINF(3)", ".", f)
        line(0.000, "AMUINF(4)", ".", f)
        line(0.000, "AMUINF(5)", ".", f)
        line(0.000, "AMUINF(6)", ".", f)
        line(0.000, "AMUINF(7)", ".", f)
        blankline(f)
        f.write("f. The EMISSION parameters\n")
        blankline(f)
        line("", "", "Number of vortex particles created within a time step", f)
        line(params.NNEVP0, "NNEVP0", "per near-wake element of a thin  wing", f)
        line(params.RELAXU, "RELAXU", "relaxation factor for the emission velocity", f)
        line(
            params.PARVEC,
            "PARVEC",
            "parameter for the minimum width of the near-wake elemen.",
            f,
        )
        line(params.NEMIS, "NEMISS", "=1,2 (See CREATE)", f)
        blankline(f)
        f.write("g. The DEFORMATION parameters\n")
        blankline(f)
        line(params.EPSFB, "EPSFB", "Cut-off length for the bound vorticity", f)
        line(params.EPSFW, "EPSFW", "Cut-off length for the near-wake vorticity", f)
        line(params.EPSSR, "EPSSR", "Cut-off length for source distributions", f)
        line(params.EPSDI, "EPSDI", "Cut-off length for source distributions", f)
        line(
            params.EPSVR,
            "EPSVR",
            "Cut-off length for the free vortex particles (final)",
            f,
        )
        line(
            params.EPSO,
            "EPSO",
            "Cut-off length for the free vortex particles (init.)",
            f,
        )
        line(params.EPSINT, "EPSINT", "", f)
        line(params.COEF, "COEF", "Factor for the disipation of particles", f)
        line(params.RMETM, "RMETM", "Upper bound of the deformation rate", f)
        line(
            params.IDEFW,
            "IDEFW",
            "Parameter for the deformation induced by the near wake",
            f,
        )
        line(
            params.REFLEN,
            "REFLEN",
            "Length used in VELEF for suppresing far-particle calc.",
            f,
        )
        line(params.IDIVVRP, "IDIVVRP", "Parameter for the subdivision of particles", f)
        line(
            params.FLENSC,
            "FLENSC",
            "Length scale for the subdivision of particles",
            f,
        )
        line(params.NREWAK, "NREWAK", "Parameter for merging of particles", f)
        line(params.NMER, "NMER", "Parameter for merging of particles", f)
        line(params.XREWAK, "XREWAK", "X starting distance of merging", f)
        line(params.RADMER, "RADMER", "Radius for merging", f)
        blankline(f)
        f.write("j. The MANAGEMENT parameters\n")
        blankline(f)
        line(1, "ITERPRE", "Write forces every ... time steps", f)
        line(1, "ITERWAK", "Write wake geometry every ... time steps", f)
        line(10000, "ITERVEL", "Write inflow velocities", f)
        line(10000, "ITERREC", "Take back-up every ... time steps", f)
        line(100, "ITERLOA", "Write loads every ... time steps", f)
        line(1, "ITERCHW", "Check the wake calculations every ... time steps", f)
        blankline(f)
        f.write("i. The FLUID parameters\n")
        blankline(f)
        line(params.rho, "AIRDEN", "Fluid density", f)
        line(params.visc, "VISCO", "Kinematic viscosity", f)
        blankline(f)
        f.write("k. The APPLICATION parameters\n")
        blankline(f)
        line(0, "IAPPLIC", "= 0(no action), 1(velocity profiles in the wakes)", f)
        line(0, "IUEXTER", "= 0(no action), 1(there is an external velocity field)", f)
        blankline(f)
        blankline(f)
        f.write("GIVE THE NAME OF THE DATA FILE FOR THE BODIES OF THE CONFIGURATION\n")
        f.write("            ... FILEGEO\n")
        line(
            "hermes.geo",
            "FILEGEO",
            "the data file for the geometry of the configuration",
            f,
        )
        f.write(
            "                       (See DGEOM-3.frm format file, Subr. INITGEO and\n",
        )
        f.write("                        gnvp-3.txt)\n")
        line(0, "IYNELST", "(1=BEAMDYN,2-ALCYONE,3=GAST)", f)
        f.write("\n")
        f.write("\n")


def geo_file(
    movements: list[list[GNVP_Movement]],
    bodies_dicts: list[GenuSurface],
) -> None:
    """Create Geo file for GNVP3

    Args:
        movements (list[list[Movement]]): List of Movements for each body
        bd_dicts (GenuSurface): List of Bodies in GenuSurface format

    """
    fname = "hermes.geo"
    # with open(fname, "r") as file:
    #     data = file.readlines()
    data: list[str] = []
    data.append("READ THE FLOW AND GEOMETRICAL DATA FOR EVERY SOLID BODY\n")
    data.append("               <blank>\n")

    for i, bod in enumerate(bodies_dicts):
        data.append("               <blank>\n")
        NB: int = bod.NB
        geo_body_header(data, bod, NB)
        data.append(f"{len(movements[i]) + 1}           LEVEL  the level of movement\n")
        data.append("               <blank>\n")
        data.append("Give  data for every level\n")
        # PITCH, ROLL, YAW, Movements to CG with angular velocity
        for j, mov in enumerate(movements[i]):
            geo_body_movements(data, mov, len(movements[i]) - j, NB)

        data.append(
            "-----<end of movement data>----------------------------------------------------\n",
        )
        data.append("               <blank>\n")
        data.append("Cl, Cd data / IYNVCR(.)=0 then Cl=1., Cd=0.\n")
        data.append("1           IYNVCR(1)\n")
        data.append(
            f"{bod.cld_fname}      FLCLCD      file name wherefrom Cl, Cd are read\n",
        )
        data.append("               <blank>\n")
        data.append("Give the file name for the geometrical distributions\n")
        data.append(f"{bod.bld_fname}\n")
    data.append("               <blank>\n")
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def geo_body_header(data: list[str], body: GenuSurface, NB: int) -> None:
    """Crates the header of each body in the geo file.

    Args:
        data (list[str]): List of string to append to and write to file
        body (GenuSurface): Body in GenuSurface format
        NB (int): Body Number

    """
    data.append(f"Body Number   NB = {NB}\n")
    data.append("               <blank>\n")
    data.append("2           NLIFT\n")
    data.append("0           IYNELSTB   \n")
    data.append("0           NBAER2ELST \n")
    data.append(f"{body.NNB}           NNBB\n")
    data.append(f"{body.NCWB}          NCWB\n")
    data.append("2           ISUBSCB\n")
    data.append("2\n")
    data.append("3           NLEVELSB\n")
    data.append("0           IYNTIPS \n")
    data.append("0           IYNLES  \n")
    data.append("0           NELES   \n")
    data.append("0           IYNCONTW\n")
    data.append("2           IDIRMOB  direction for the torque calculation\n")
    data.append("               <blank>\n")


def geo_body_movements(data: list[str], mov: GNVP_Movement, i: int, NB: int) -> None:
    """Add Movement Data to Geo File.

    Args:
        data (list[str]): Data to append to
        mov (Movement): Movement Object
        i (int): Index of Movement
        NB (int): Body Number

    """
    data.append(f"NB={NB}, lev={i}  ( {mov.name} )\n")
    data.append("Rotation\n")
    data.append(f"{int(mov.rotation_type)}           IMOVEAB  type of movement\n")
    data.append(
        f"{int(mov.rotation_axis)}           NAXISA   =1,2,3 axis of rotation\n",
    )
    data.append(f"{ff3(mov.rot_t1)}    TMOVEAB  -1  1st time step\n")
    data.append(f"{ff3(mov.rot_t2)}    TMOVEAB  -2  2nd time step\n")
    data.append("0.          TMOVEAB  -3  3d  time step\n")
    data.append("0.          TMOVEAB  -4  4th time step!---->omega\n")
    data.append(f"{ff3(mov.rot_a1)}    AMOVEAB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.rot_a2)}    AMOVEAB  -2  2nd value of amplitude\n")
    data.append("0.          AMOVEAB  -3  3d  value of amplitude\n")
    data.append("0.          AMOVEAB  -4  4th value of amplitude!---->phase\n")
    data.append("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")
    data.append("Translation\n")
    data.append(f"{int(mov.translation_type)}           IMOVEUB  type of movement\n")
    data.append(
        f"{int(mov.translation_axis)}           NAXISU   =1,2,3 axis of translation\n",
    )
    data.append(f"{ff3(mov.translation_t1)}    TMOVEUB  -1  1st time step\n")
    data.append(f"{ff3(mov.translation_t2)}    TMOVEUB  -2  2nd time step\n")
    data.append("0.          TMOVEUB  -3  3d  time step\n")
    data.append("0.          TMOVEUB  -4  4th time step\n")
    data.append(f"{ff3(mov.translation_a1)}    AMOVEUB  -1  1st value of amplitude\n")
    data.append(f"{ff3(mov.translation_a2)}    AMOVEUB  -2  2nd value of amplitude\n")
    data.append("0.          AMOVEUB  -3  3d  value of amplitude\n")
    data.append("0.          AMOVEUB  -4  4th value of amplitude\n")
    data.append("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")


def cld_files(bodies: list[GenuSurface], params: GenuParameters, solver: str) -> None:
    """Create Polars CL-CD-Cm files for each airfoil

    Args:
        bodies (list[GenuSurface]): list of bodies in GenuSurface format
        solver (str): preferred solver

    """
    for bod in bodies:
        fname: str = f"{bod.cld_fname}"

        # Get the airfoil polar
        reynolds = float(bod.mean_aerodynamic_chord * np.linalg.norm(params.u_freestream) / params.visc)
        try:
            DB = Database.get_instance()
            polars: AirfoilPolars = DB.get_or_compute_airfoil_polars(
                airfoil=DB.get_airfoil(bod.airfoil_name),
                reynolds=reynolds,
                solver_name=solver,
                aoa=np.linspace(-10, 16, 53),
            )
        except PolarsNotFoundError:
            raise ValueError(
                f"Airfoil Polars for {bod.airfoil_name} not found in the database and could not be computed. Please compute the polars first.",
            )

        f_io = StringIO()
        f_io.write(f"------ CL and CD data input file for {bod.airfoil_name}\n")
        f_io.write("------ Mach number dependence included\n")
        blankline(f_io)
        line(2, "NSPAN", "Number of positions for which CL-CD data are given", f_io)
        line(
            len(polars.reynolds_nums),
            "! NMACH",
            "Mach numbers for which CL-CD are given",
            f_io,
        )
        for _ in range(len(polars.reynolds_nums)):
            f_io.write("0.08\n")
        f_io.write("! Reyn numbers for which CL-CD are given\n")
        for reyn in polars.reynolds_keys:
            f_io.write(f"{reyn.zfill(5)}\n")
        blankline(f_io)

        df: DataFrame = polars.df
        angles = polars.angles

        for radpos in [-100.0, 100.0]:
            line(radpos, "RADPOS", "! Radial Position", f_io)
            f_io.write(
                f"{len(angles)}         ! Number of Angles / Airfoil      {bod.airfoil_name}\n",
            )
            f_io.write(
                "   ALPHA   CL(M=0.0)   CD       CM    CL(M=1)   CD       CM \n",
            )
            for i, ang in enumerate(angles):
                string: str = ""
                nums = df.loc[df["AoA"] == ang].to_numpy().squeeze()
                for num in nums:
                    string = string + ff2(num) + "  "
                f_io.write(f"{string}\n")
            f_io.write("\n")
        f_io.write("------ End of CL and CD data input file\n")
        # Write to File
        contents: str = f_io.getvalue().expandtabs(4)
        with open(fname, "w", encoding="utf-8") as file:
            file.write(contents)


def bld_files(bodies: list[GenuSurface], params: GenuParameters) -> None:
    """Create BLD files for each body

    Args:
        bodies (list[GenuSurface]): list of bodies in GenuSurface format
        params (GenuParameters): Genu Parameters object containing all parameters

    """
    for bod in bodies:
        fname: str = bod.bld_fname
        with open(fname, "w", encoding="UTF-8") as f:
            f.write(f"INPUT FILE FOR {bod.name}\n")
            f.write("0\n")
            f.write(
                "IFTYP      INDEXFL    NFILFD      [IFTYP=1-->NACA, =2-->Read data from NFILFD]\n",
            )

            # Check Whether to split a symmetric body into two parts
            if not params.Use_Grid:
                f.write(
                    f"1          {''.join(char for char in bod.airfoil_name if char.isdigit())}\n",
                )
            else:
                f.write(
                    f"0          {''.join(char for char in bod.airfoil_name if char.isdigit())}       {bod.name}.WG\n",
                )
                # WRITE GRID FILE Since Symmetric objects cant be defined parametrically
                # Specify option 0 to read the file
                with open(f"{bod.name}.WG", "w") as f_wg:
                    grid: FloatArray | list[FloatArray] = bod.grid
                    f_wg.write("\n")
                    if isinstance(grid, list):
                        grid = np.array(grid)

                    if len(np.shape(grid)) == 4:
                        for subgrid in grid:
                            for nstrip in subgrid:
                                for point in nstrip:
                                    f_wg.write(f"{point[0]} {point[1]} {point[2]}\n")
                                f_wg.write("\n")
                    elif len(np.shape(grid)) == 3:
                        for n_strip in grid:  # For each strip
                            for m_point in n_strip:  # For each point in the strip
                                # Grid Coordinates
                                f_wg.write(f"{m_point[0]} {m_point[1]} {m_point[2]}\n")
                            f_wg.write("\n")
                    elif len(np.shape(grid)) == 2:
                        for point in grid:
                            f_wg.write(f"{point[0]} {point[1]} {point[2]}\n")
                        f_wg.write("\n")

            blankline(f)
            f.write(
                "IFWRFL     IFWRDS     IFWRWG      [=0, no action, =1, write results]\n",
            )
            f.write("1          1          1\n")
            blankline(f)
            f.write("NFILFL     NFILDS     NFILWG      [corresponding file names]\n")
            f.write(
                f"{file_name(bod.name, 'FL')}{file_name(bod.name, 'DS')}{file_name(bod.name, 'OWG')}\n",
            )
            blankline(f)
            f.write("XOO        YOO        ZOO\n")
            f.write(f"{ff4(bod.x_0)} {ff4(bod.y_0)} {ff4(bod.z_0)}\n")
            blankline(f)
            f.write("PITCH      CONE       WNGANG[azimuthal angle]\n")
            f.write(f"{ff4(bod.pitch)} {ff4(bod.cone)} {ff4(bod.wngang)}\n")
            blankline(f)
            f.write("IEXKS      NFILKS      AKSI(1)    AKSI(NNC)\n")
            f.write("1          0           0.         1.\n")
            blankline(f)

            step: float = round(
                (bod.root_chord - bod.tip_chord) / (bod.y_end - bod.y_0),
                ndigits=5,
            )
            offset: float = round(bod.offset / (bod.y_end - bod.y_0), ndigits=5)

            f.write("IEXRC      NFILRC      RC  (1)    RC  (NNC)\n")
            f.write(f"1          0           0.         {ff5(bod.y_end - bod.y_0, 8)}\n")
            blankline(f)
            f.write(
                "IEXCH      NFILCH      FCCH(1)    FCCH(2)   FCCH(3)    FCCH(4)    FCCH(5)    FCCH(6)\n",
            )
            f.write(
                f"4                      {ff4(bod.root_chord)} {ff4(-step)}     0.         0.         0.         0.\n",
            )
            blankline(f)
            f.write(
                "IEXTW      NFILTW      FCTW(1)    FCTW(2)   FCTW(3)    FCTW(4)    FCTW(5)    FCTW(6)\n",
            )
            f.write(
                "4                      0.        0         0.         0.         0.         0.\n",
            )
            blankline(f)
            f.write(
                "IEXXO      NFILXO      FCXO(1)    FCXO(2)   FCXO(3)    FCXO(4)    FCXO(5)    FCXO(6)\n",
            )
            f.write(
                f"4                      {ff4(0.0)} {ff4(offset)}     0.         0.         0.         0.\n",
            )
            blankline(f)
            f.write(
                "IEXZO      NFILZO      FCZO(1)    FCZO(2)   FCZO(3)    FCZO(4)    FCZO(5)    FCZO(6)\n",
            )
            f.write(
                "4                      0.         0.         0.         0.         0.         0.\n",
            )
            blankline(f)
            f.write(
                "IEXXS      NFILXS      FCXS(1)    FCXS(2)   FCXS(3)    FCXS(4)    FCXS(5)    FCXS(6)\n",
            )
            f.write(
                "4                      0.         0.         0.         0.         0.         0.\n",
            )
            blankline(f)
            f.write(
                "IEXRT      NFILRT      FCRT(1)    FCRT(2)   FCRT(3)    FCRT(4)    FCRT(5)    FCRT(6)\n",
            )
            f.write(
                "4                      0.         0.         0.         0.         0.         0.\n",
            )
            blankline(f)

            f.write(
                "C  INDEXFL    index of the airfoil (e.g. 4412 for NACA 4-digits airfoil)\n",
            )
            f.write(
                "C  NFILxx     xx = KS, RC, CH, TH, XO, ZO, RT input file names for the\n",
            )
            f.write(
                "C                                             corresponding distributions\n",
            )
            f.write("C  IEXxx      parameter defining the type of interpolation\n")
            f.write("C         I)  in s. DISTRB\n")
            f.write(
                "C             xx distributions (xx=KS for AKSI(.), xx=RC for RC(.))\n",
            )
            f.write("C             =1 for linear interpolation\n")
            f.write("C             =2 for sinusoinal interpolation\n")
            f.write("C             =3 for double sinusoidal interpolation\n")
            f.write("C             =5 for reading the disribution of xx from file\n")
            f.write("C                (NNB(xx=KS)/NCW(xx=RC) must be equal to NUPO)\n")
            f.write("C        II)  in s. CATAN\n")
            f.write(
                "C             xx distributions (xx=CH for CH(.), xx=TW for TW(.),\n",
            )
            f.write(
                "C                xx=XO for XO(.), xx=ZO for ZO(.), xx=RT for RT(.))\n",
            )
            f.write("C             =4 for polynomial interpolation\n")
            f.write(
                "C             =5 for reading the disributions of xx and DxxDR from file\n",
            )
            f.write("C                (NCW must be equal to NUPO)\n")
            f.write(
                "C             =6 for reading the disributions of xx and DxxDR from file\n",
            )
            f.write("C                and interpolate both according RC(.)\n")
            f.write(
                "C             =7 for reading the disribution ONLY of xx from file and\n",
            )
            f.write(
                "C                calculate DxxDR using 2nd order finite differences\n",
            )
            f.write("C                schemes. (NCW must be equal to NUPO)\n")
            f.write(
                "C             =8 for reading the disribution ONLY of xx from file,\n",
            )
            f.write(
                "C                interpolate according RC(.) and  calculate DxxDR using\n",
            )
            f.write(
                "C                2nd order finite differences shemes according RC(.)\n",
            )
            f.write("C  NUPOxx     number of data pairs for the distributions:\n")
            f.write("C             AKSI(.), RC(.), CH(.), TW(.), XO(.), ZO(.), RT(.)\n")
            f.write("C             NUPOxx is read from the 2nd line of file  NFILxx\n")
            f.write("C")


def hybrid_wake_file() -> None:
    """Creates the hybrid wake file for GNVP3"""
    fname: str = "hyb.inf"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("Data for the Hybrid wake calculations \n")
        f.write("  0.150  DGRLEN\n")
        f.write("10000    NTIMEHYB\n")
        f.write(" 18      NHYBAV\n")
        f.write("360      NTHYBP\n")


def make_input_files(
    ANGLEDIR: str,
    HOMEDIR: str,
    movements: list[list[GNVP_Movement]],
    bodies: list[GenuSurface],
    params: GenuParameters,
    solver: str,
) -> None:
    os.chdir(ANGLEDIR)
    # Input File
    input_file()
    # Hybrid Wake input file
    hybrid_wake_file()
    # DFILE
    case_file(params)
    # HERMES.GEO
    geo_file(movements, bodies)
    # BLD FILES
    bld_files(bodies, params)
    # CLD FILES
    cld_files(bodies, params, solver)
    # Check if gnvp.out exists and remove it
    if os.path.exists("gnvp.out"):
        os.remove("gnvp.out")
    os.chdir(HOMEDIR)


def remove_results(CASEDIR: str, HOMEDIR: str) -> None:
    """Removes the simulation results from a GNVP3 case

    Args:
        CASEDIR (str): _description_
        HOMEDIR (str): _description_

    """
    os.chdir(CASEDIR)
    os.remove("strip*")
    os.remove("x*")
    os.remove("YOURS*")
    os.remove("refstate*")
    os.chdir(HOMEDIR)
