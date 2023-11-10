"""
GNVP7 input files generation
"""
import os
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

from ICARUS.Airfoils.airfoil_polars import Polars
from ICARUS.Core.formatting import ff2
from ICARUS.Core.formatting import ff3
from ICARUS.Core.formatting import ff4
from ICARUS.Core.formatting import sps
from ICARUS.Core.formatting import tabs
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Input_Output.GenuVP.utils.genu_movement import Movement
from ICARUS.Input_Output.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.Input_Output.GenuVP.utils.genu_surface import GenuSurface


def input_file(maxiter: float, timestep: float) -> None:
    """Create input file for gnvp7"""

    fname: str = "input"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"{maxiter}             !NTIMEM\n")
        f.write("1               !IDT\n")
        f.write(f"{timestep}            !OMEGA_DT\n")
        f.write("1               !INIT_gn\n")


def dfile(params: GenuParameters) -> None:
    """
    Create dfile for gnvp7

    Args:
        params (GenuParameters): An object containing the parameters values
    """
    f_io = StringIO()
    # Header
    f_io.write(f"{tabs(16)}<blank>\n")
    f_io.write(f"** Define the type of application{tabs(8)}<blank>\n")
    f_io.write(f"7{tabs(3)}IEFARMOGH 4=Heli,6=WIND\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # Basic parameters
    nbodt = int(params.nBods)
    nblades = int(params.nBlades)
    f_io.write(f"** Read the BASIC parameters{tabs(9)}<blank>\n")
    f_io.write(f"1{tabs(3)}NSYMF{sps(6)}=1,2,3 (no-symm, axi-symm, Y-symm)\n")
    f_io.write(f"{nbodt}{tabs(3)}NBODT{sps(6)}number of bodies\n")
    f_io.write(f"{nblades}{tabs(3)}SymLevels=old NBLADE{tabs(1)} number of blades\n")
    f_io.write(f"1{tabs(3)}IAXISRF{sps(4)}=1,2,3 gives the axis of rotation if IABSREF=1\n")
    f_io.write(f"{params.NLEVELT}{tabs(3)}NLEVELT{sps(4)}number of movement levels\n")
    f_io.write(f"0.{tabs(3)}OMEGAR{sps(5)}is the rotation speed of the RCS\n")

    # Simulation options
    f_io.write(f"** Read the simulation options{tabs(9)}<blank>\n")
    f_io.write(f"0{tabs(3)}IYNPMESH\n")
    f_io.write(f"0{tabs(3)}IYNSG\n")
    f_io.write(f"0{tabs(3)}IYNCHVOR\n")
    f_io.write(f"10000{tabs(2)}NTIMEHYB{tabs(11)}<blank>\n")
    f_io.write(f"18{tabs(3)}NHYBAV{tabs(12)}<blank>\n")
    f_io.write(f"360{tabs(3)}NTHYBP{tabs(12)}<blank>\n")
    f_io.write(f"0.15{tabs(2)}DGRLEN\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # Time parameters
    f_io.write(f"** Read the TIME parameters{tabs(10)}<blank>\n")
    f_io.write(f"1{tabs(3)}OMEGAT{sps(5)}the rotation speed for the definition of the PERIOD\n")
    f_io.write(f"{params.NMETH}{tabs(3)}NMETHT{sps(5)}=1 for Euler =2 for Adams Bashford time integrat. scheme\n")

    # Tip emission parameters
    f_io.write(f"{params.NEMTIP}{tabs(3)}NEMTIP{sps(5)}=0,1. The latter means that tip-emission takes place\n")
    f_io.write(f"{ff2(params.NTIMET)}{tabs(2)}NTIMET{sps(5)}time step that tip-emission begins\n")
    f_io.write(f"{params.NEMSLE}{tabs(3)}NEMSLE{sps(5)}=0(no action), 1(leading-edge separ. takes place)\n")
    f_io.write(f"{ff2(params.NTIMEL)}{tabs(2)}NTIMEL{sps(5)}time step that leading-edge separation starts\n")

    # Root emission parameters
    f_io.write(f"0{tabs(3)}NEMROOT{sps(4)}=0(no action), 1(root emission takes place)\n")
    f_io.write(f"0{tabs(3)}NTIMERO{sps(4)}time step that root emission starts\n")
    f_io.write(f"0.{tabs(3)}AZIMIN{sps(5)}the initial azimuthal angle\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # Solution parameters
    f_io.write(f"** Read the SOLUTION parameters{tabs(9)}<blank>\n")
    f_io.write(f"0{tabs(3)}IMAT{sps(7)}=0 AS is calculated every timestep, =1 only once\n")
    f_io.write(f"200{tabs(3)}ITERM{sps(6)}maximum number of potential iterations\n")
    f_io.write(f"{params.RELAXS}{tabs(3)}RELAXS{sps(5)}relaxation factor for the singularity distributions\n")
    f_io.write(f"{params.EPSDS}{tabs(2)}EPSDS{sps(6)}convergence tolerance of the potential calculations\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # Inflow parameters
    u_x: float = params.u_freestream[0]
    u_y: float = params.u_freestream[1]
    u_z: float = params.u_freestream[2]

    u_inf: float = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    yaw: float = np.arctan2(u_y, u_x) * 180 / np.pi
    inc: float = np.arctan2(u_z, u_x) * 180 / np.pi

    f_io.write(f"** Read the INFLOW parameters{tabs(9)}<blank>\n")
    f_io.write(f"0.{tabs(3)}UREF{tabs(2)}reference velocity\n")
    f_io.write(f"{ff2(u_inf)}{tabs(3)}AUINF{tabs(2)}wind velocity\n")
    f_io.write(f" {ff2(yaw)}{tabs(2)}YawAngle{tabs(1)}Yaw Angle [deg]  ! in Z always\n")
    f_io.write(f" {ff2(inc)}{tabs(2)}IncAngle{tabs(1)}Inclination Angle [deg]  ! in Y always\n")
    f_io.write(f" 0.0{tabs(2)}VertShear{tabs(1)}Shear effect in the vertical direction wrt hub height (exponential)\n")
    f_io.write(f" 0.0{tabs(2)}HorzShear{tabs(1)}DUMMY at the moment\n")
    f_io.write(f" 0.0{tabs(2)}WindVeer{tabs(1)}slope of yaw angle / m wrt hub height [deg/m]\n")
    f_io.write(
        f" 0{tabs(3)}IWINDC{tabs(2)}wind case 0: uniform wind, 1: defined wind scenario, 2: EOG, 3x: ECD, 4x: EDC, 5x: EWS\n",
    )
    f_io.write(f" 0.0{tabs(2)}TIME_GUST{tabs(1)}Time to initiate the extreme event [2<=IWINDC<=5x]\n")
    f_io.write(f" 0.0{tabs(2)}TIREF_GUST{tabs(1)}TI for gust [see IEC]\n")
    f_io.write(f" 0.0{tabs(2)}VREF_GUST{tabs(1)}Vref for gust [see IEC]\n")
    f_io.write(f" 0{tabs(3)}ITURB{tabs(2)}0: no turb, 1: disk, 2: rectangular\n")
    f_io.write(f" 0.0{tabs(2)}TIME_turb{tabs(1)}Time after which the turbulent wind starts [sec]\n")
    f_io.write(f" 0{tabs(3)}ISHADOW{tabs(2)}Include Tower Shadow: 1: Y, 0: N\n")
    f_io.write(f"  0{tabs(3)}BotTowerR{tabs(1)}R tower bottom{tabs(1)}[m]\n")
    f_io.write(f"  0{tabs(3)}TopTowerR{tabs(1)}R tower top{tabs(2)}[m]\n")
    f_io.write(f"  0{tabs(2)}TowerH{tabs(2)}tower height{tabs(1)}[m]\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # Geometrical parameters
    f_io.write(f"** Read the geometrical parameters{tabs(8)}<blank>\n")
    f_io.write(f"1 {tabs(3)}IAXISUI{tabs(2)}axis of the global system\n")
    f_io.write(
        f"1.{tabs(3)}Refz{tabs(2)}reference Z position for the shear and the turbulent wind [defines Hub height]\n",
    )
    f_io.write(f"1.{tabs(3)}RTIP{tabs(2)}Radius of the blade\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # EMISSION parameters
    f_io.write(f"** Read the EMISSION parameters{tabs(9)}<blank>\n")
    f_io.write(f"{ff2(params.NNEVP0)}{tabs(3)}NNEVP0     per near-wake element of a thin wing\n")
    f_io.write(f"{ff2(params.NNEVP0)}{tabs(3)}NNEVP1     per near-wake element of a thick wing\n")
    f_io.write(f"{ff2(params.RELAXU)}{tabs(3)}RELAXU     relaxation factor for the emission velocity\n")
    f_io.write(f"{ff2(params.NEMIS)}{tabs(2)}NEMISS     =0,1 (See CREATE)\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # DEFORMATION parameters
    # DX: float = float(1.5 * np.linalg.norm(params.u_freestream) * params.timestep)
    # if DX > 0.005:
        # DX = 0.003
    

    f_io.write(f"** Read the DEFORMATION parameters{tabs(8)}<blank>\n")
    f_io.write(f"{ff2(params.EPSFB)}{tabs(2)}EPSFB      Cut-off length for the bound vorticity\n")
    f_io.write(f"{ff2(params.EPSFW)}{tabs(2)}EPSFW      Cut-off length for the near-wake vorticity\n")
    f_io.write(f"{ff2(params.EPSSR)}{tabs(2)}EPSSRC     Cut-off length for source distributions\n")
    f_io.write(f"{ff2(params.EPSDI)}{tabs(2)}EPSDIP     Cut-off length for dipole distributions\n")
    f_io.write(f"{ff2(params.EPSVR)}{tabs(2)}EPSVR      Cut-off length for the free vortex particles (final)\n")
    f_io.write(f"{ff2(params.EPSO)}{tabs(2)}EPSO       Cut-off length for the free vortex particles (init.)\n")
    f_io.write(f"{ff2(params.EPSINT)}{tabs(2)}EPSINTbas  Cut-off length for VORTEX SOLID INTERACTION\n")
    f_io.write(f"{ff2(params.COEF)}{tabs(2)}COEF       Factor for the dissipation of particles\n")
    f_io.write(f"{ff2(params.RMETM)}{tabs(2)}RMETM      Upper bound of the deformation rate\n")
    f_io.write(f"{ff2(params.IDEFW)} {tabs(2)}IDEFW      Parameter for the deformation induced by the near wake\n")
    f_io.write(f"{ff2(params.REFLEN)}{tabs(2)}REFLEN     Length used in VELEF for suppressing far-particle calculations\n")
    f_io.write(f"{ff2(params.IDIVVRP)} {tabs(2)}IDIVVRP    Parameter for the subdivision of particles\n")
    f_io.write(f"{ff2(params.FLENSC)}{tabs(2)}FLENSC     Length scale for the subdivision of particles\n")
    f_io.write(f"{ff2(params.NREWAK)}{tabs(2)}NREWAK     Parameter for merging of particles\n")
    f_io.write(f"{ff2(params.NMER)}{tabs(2)}NMER       Parameter for merging of particles\n")
    f_io.write(f"{ff2(params.XREWAK)}{tabs(2)}XREWAK     X starting distance of merging\n")
    f_io.write(f"{ff2(params.RADMER)}{tabs(2)}RADMER     Radius for merging\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # I/O specifications
    name: str = params.name

    f_io.write(f"** I/O specifications{tabs(11)}<blank>\n")
    f_io.write(f"{name}.TOT{tabs(1)}OFILE\n")
    f_io.write(f"{name}.SAS{tabs(1)}SUPAS\n")
    f_io.write(f"{name}.CHW{tabs(1)}CHWAK\n")
    f_io.write(f"100{tabs(3)}ITERCHW    Check the wake calculations every ... time steps\n")
    f_io.write(f"{name}.WAK{tabs(1)}OWAKE\n")
    f_io.write(f"100{tabs(3)}ITERWAK    Write wake geometry every ... time steps\n")
    f_io.write(f"{name}.PRE{tabs(1)}OPRES\n")
    f_io.write(f"100{tabs(3)}ITERPRE    Write forces every ... time steps\n")
    f_io.write(f"{name}.BAK{tabs(1)}RCALL\n")
    f_io.write(f"100{tabs(3)}ITERREC    Take back-up every ... time steps\n")
    f_io.write(f"{name}.LOA{tabs(1)}LOADS\n")
    f_io.write(f"100{tabs(3)}ITERLOA    Write loads every ... time steps\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    f_io.write(f"1{tabs(3)}ICHEDAT    = 0(no action), 1(check the data)\n")
    f_io.write(f"1{tabs(3)}ICHEDIM    = 0(no action), 1(check the dimensions)\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # FLUID parameters
    f_io.write(f"** Read the FLUID parameters{tabs(9)}<blank>\n")
    f_io.write(f"{params.rho}{tabs(2)}AIRDEN     Fluid density\n")
    f_io.write(f"{params.visc}{tabs(1)}VISCO      Kinematic viscosity\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # APPLICATION parameters
    f_io.write(f"** Read the APPLICATION parameters{tabs(8)}<blank>\n")
    f_io.write(f"0{tabs(3)}IAPPLIC    = 0(no action), 1(------------------------------)\n")
    f_io.write(f"0{tabs(3)}IDEX_APPLIC\n")
    f_io.write(f"{tabs(3)}FILE_APPLIC\n")
    f_io.write(f"{params.IYNELST}{tabs(3)}IYNELAST    aeroelastic coupling\n")
    f_io.write(f"{tabs(16)}<blank>\n")

    # FILES data
    f_io.write(f"** Read the FILES data{tabs(11)}<blank>\n")
    f_io.write(f"{name}.geo{tabs(1)}FILEGEO\n")
    f_io.write(f"{tabs(16)}<blank>\n")
    f_io.write(f"{name}.bcon{tabs(1)}FilBConn\n")
    f_io.write(f"{name}.wcon{tabs(1)}FilWConn\n")

    fname = "dfile"
    contents: str = f_io.getvalue().expandtabs(4)

    with open(fname, "w", encoding="utf-8") as f:
        f.write(contents)


def geofile(
    movements: list[list[Movement]],
    bodies: list[GenuSurface],
    params: GenuParameters,
) -> None:
    """
    Creates the geofile for GNVP7. The geofile contains the information about
    the geometry of the bodies and the movements of the bodies.

    Args:
        movements (list[list[Movement]]): List of movements for every body
        bodies (list[GenuSurface]): List of bodies in dictionary format
        params (GenuParameters): An object containing the parameters values
    """
    f_io = StringIO()
    f_io.write(f"READ THE FLOW AND GEOMETRICAL DATA FOR EVERY SOLID BODY\n")
    f_io.write(f"{tabs(3)}<blank>\n")
    f_io.write(f"{tabs(3)}<blank>\n")

    for i, bod in enumerate(bodies):
        NB: int = bod.NB + 1
        NNB: int = bod.NNB
        NCWB: int = bod.NCWB
        type_bod: int = 2 if bod.type == "thin" else 3
        type_lift: int = 1 if bod.lifting else 0

        f_io.write(f"Body Number{tabs(1)}NB = {NB}\n")
        f_io.write(f"{tabs(3)}<blank>\n")
        f_io.write(f"{bod.name}{tabs(2)}BodyName\n")
        f_io.write(f"{NB}{tabs(3)}IndBodyB\n")
        f_io.write(f"{type_bod}{tabs(3)}TypBodyB   1 2 (thin) 3 (thick) 4 (tip) 5 (spoiler)\n")
        f_io.write(f"{type_lift}{tabs(3)}YNLiftB    0 1        1         0/1     1\n")
        f_io.write(f"1{tabs(3)}YNxyTerm\n")
        f_io.write(f"{NNB}{tabs(3)}NBB\n")
        f_io.write(f"{NCWB}{tabs(3)}NCWB\n")
        f_io.write(f"1{tabs(3)}NodeHalf\n")
        f_io.write(f"4{tabs(3)}NEL_C4\n")
        f_io.write(f"{tabs(3)}ISUBSCB\n")
        f_io.write(f"{tabs(3)}JSUBSCB\n")
        f_io.write(f"{tabs(3)}NLEVELSB\n")
        f_io.write(f"3{tabs(3)}IDIRMOB direction for the torque calculation\n")
        f_io.write(f"{tabs(16)}<blank>\n")

        # Write the movements
        f_io.write(f"{len(movements[i])}{tabs(3)}LEVEL   the level of movement\n")
        f_io.write(f"{bod.move_fname}{tabs(2)}FILMOV\n")
        body_movements(movements=movements[i], NB=NB, fname=bod.move_fname)
        f_io.write(f"{tabs(16)}<blank>\n")
        f_io.write(f"{tabs(16)}<blank>\n")

        # Write the load parameters
        f_io.write(f"1 1 0{tabs(2)}LoadViscCor  0= No, 1[1,2]=a_ci[a_ci,a_f] 2[1,2]=a_f[a_ci,a_f]\n")
        f_io.write(f"{bod.cld_fname}{tabs(2)}FLCLCDB     file name wherefrom Cl, Cd are read\n")
        f_io.write(f"{tabs(16)}<blank>\n")
        f_io.write(f"{tabs(16)}<blank>\n")

        # Write the grid parameters
        f_io.write(f"{bod.grid_fname}{tabs(2)}FilGridB\n")
        grid_file(bod)
        f_io.write(f"{bod.topo_fname}{tabs(2)}FilTopoB\n")
        f_io.write(f"{bod.wake_fname}{tabs(2)}FilWakeB\n")
        f_io.write(f"{tabs(16)}<blank>\n")
        f_io.write(f"{tabs(16)}<blank>\n")

        # Elastic properties
        f_io.write(f"0{tabs(3)} ElastBody\n")
        f_io.write(f"{tabs(16)}<blank>\n")

        # The Rotation Center and the Axis represent the axis and position around which Genu Calculates the Torque around which
        # GNVP7 calculates the torque. The rotation center is the point around which the torque is calculated. The axis is the
        # axis around which the torque is calculated. Should be noted that we first define them around the CG of the body. Should,
        # one body move each new rot_center will be updated to reflect the movement, but the new CG will NOT!
        cg: list[float] = params.CG.tolist()
        # f_io.write(f"{ff4(cg[0])} {ff4(cg[1])} {ff4(cg[2])}{tabs(2)}ROTOR Center !for WT (0.,0.,HubHeigth)\n")
        f_io.write(f"{ff4(0)} {ff4(0)} {ff4(0)}{tabs(2)}ROTOR Center !for WT (0.,0.,HubHeigth)\n")
        # f_io.write(f"{u_y/u_inf} {u_z/u_inf} {u_x/u_inf}{tabs(2)}ROTOR AXIS\n")
        f_io.write(f"1 0 0{tabs(2)}ROTOR AXIS\n")
        f_io.write("\n")
        f_io.write("------<end of body data>-----------------------------------\n")
    fname: str = f"{params.name}.geo"
    contents: str = f_io.getvalue().expandtabs(4)

    with open(fname, "w", encoding="utf-8") as f:
        f.write(contents)


def grid_file(body_dict: GenuSurface) -> None:
    """
    Generates the grid file for a body.

    Args:
        body_dict (GenuSurface): Dictionary Containing the information about
            the body in GenuSurface format.
    """
    with open(f"{body_dict.grid_fname}", "w") as file:
        grid: FloatArray = body_dict.Grid
        file.write("\n")
        for n_strip in grid:  # For each strip
            for m_point in n_strip:  # For each point in the strip
                # Grid Coordinates
                file.write(f"{m_point[0]} {m_point[1]} {m_point[2]}\n")
            file.write("\n")


def topology_files(
    bodies_dicts: list[GenuSurface],
) -> None:
    """
    # ! TODO: Make the description of this function more clear and accurate
    Specifies the topology files for each body. This is the file that contains
    the information about the connectivity of the grid points. If two bodies are
    connected, then calculations change. Specifically emmision parameters for the
    vortex particles are changed.

    Args:
        bodies_dicts (list[GenuSurface]): List of bodies in GenuSurface format.
    """
    for bod in bodies_dicts:
        f_io = StringIO()
        f_io.write("I_readNE (2=means no exection of panels)\n")
        f_io.write("2\n")
        f_io.write(" ... read bodies in contact ...\n")
        f_io.write("NumbS1  NumbS2  NumbS3  NumbS4 num of bodies per side\n")
        f_io.write("0       0       0       0\n")
        f_io.write(" ... read bod\n")
        f_io.write("  side1\n")
        f_io.write("  side2\n")
        f_io.write("  side3\n")
        f_io.write("  side4\n")
        f_io.write("---< end >------------------------------\n")

        fname: str = bod.topo_fname
        contents: str = f_io.getvalue().expandtabs(4)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(contents)


def wake_files(
    bodies_dicts: list[GenuSurface],
) -> None:
    """
    # ! TODO: Make the description of this function more clear and accurate
    Creates the wake files for each body. The wake files are the files that
    specify the wake parameters for each body. Specifically, the wake files
    specify where the emmision of particles will occur, the direction of the
    emmission and whether separation takes place

    Args:
        bodies_dicts (list[GenuSurface]): List of bodies in GenuSurface format.
    """
    for bod in bodies_dicts:
        f_io = StringIO()
        f_io.write("! Emission parameters in FilWakeB\n")
        f_io.write("\n")
        f_io.write("1          IYNTRED\n")
        f_io.write("0          IYNTIPS\n")
        f_io.write("0          NEWTIP\n")
        f_io.write("0          IYNROOT\n")
        f_io.write("0          NEWROOT\n")
        f_io.write("0          IYNLES\n")
        f_io.write("0          NELES\n")
        f_io.write("\n")
        f_io.write("# Emission velocity directions\n")
        f_io.write("1   2   3   4      side\n")
        f_io.write("0.  0.  0.  0.     Angl_velG=0\n")
        f_io.write("0.1 0.1 0.1 0.1    PARVEC_G\n")
        f_io.write("\n")
        f_io.write("LE separation is valid for thin bodies only\n")

        fname: str = bod.wake_fname
        contents: str = f_io.getvalue().expandtabs(4)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(contents)


def body_movements(
    movements: list[Movement],
    NB: int,
    fname: str,
) -> None:
    """
    Generate the body movement file for a given body.

    Args:
        movements (list[Movement]): List of movements to be applied to the body.
        NB (int): Number of levels of the body.
        fname (str): Name of the file to be save the movements.
    """
    f_io = StringIO()
    f_io.write("Give Flap data\n")
    f_io.write("0           IYNFlap\n")
    f_io.write("            SpanFl1  SpanFl2\n")
    f_io.write("72.016   82.304    ! =0.7*RTIP, 0.8*RTIP, RTIP=102.88m\n")
    f_io.write("            AKSIo  AFlap  Freq  Phase\n")
    f_io.write("0.9  10. 6.0318576  0.     ! AKSIo[]  AFLap[deg]  Freq[rad/s]  Phase[deg]\n")
    f_io.write("-------------------------------------------------------------------\n")
    f_io.write("Give  data for every level\n")

    for j, mov in enumerate(movements):
        f_io.write(f"NB={NB}, lev={NB-j}  ({mov.translation_axis} axis {mov.name} rotation)\n")
        f_io.write("Rotation\n")
        f_io.write(f"{int(mov.rotation_type)}           IMOVEAB  type of movement\n")
        f_io.write(f"{int(mov.rotation_axis)}           NAXISA   =1,2,3 axis of rotation\n")
        f_io.write(f"{ff4(mov.rot_t1)}    TMOVEAB  -1  1st time step\n")
        f_io.write(f"{ff4(mov.rot_t2)}    TMOVEAB  -2  2nd time step\n")
        f_io.write("0.          TMOVEAB  -3  3d  time step\n")
        f_io.write("0.          TMOVEAB  -4  4th time step!---->omega\n")
        f_io.write(f"{ff4(mov.rot_a1)}    AMOVEAB  -1  1st value of amplitude\n")
        f_io.write(f"{ff4(mov.rot_a2)}    AMOVEAB  -2  2nd value of amplitude\n")
        f_io.write("0.          AMOVEAB  -3  3d  value of amplitude\n")
        f_io.write("0.          AMOVEAB  -4  4th value of amplitude!---->phase\n")
        f_io.write("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")
        f_io.write("Translation\n")
        f_io.write(f"{int(mov.translation_type)}           IMOVEUB  type of movement\n")
        f_io.write(f"{int(mov.translation_axis)}           NAXISU   =1,2,3 axis of translation\n")
        f_io.write(f"{ff4(mov.translation_t1)}    TMOVEUB  -1  1st time step\n")
        f_io.write(f"{ff4(mov.translation_t2)}    TMOVEUB  -2  2nd time step\n")
        f_io.write("0.          TMOVEUB  -3  3d  time step\n")
        f_io.write("0.          TMOVEUB  -4  4th time step\n")
        f_io.write(f"{ff4(mov.translation_a1)}    AMOVEUB  -1  1st value of amplitude\n")
        f_io.write(f"{ff4(mov.translation_a2)}    AMOVEUB  -2  2nd value of amplitude\n")
        f_io.write("0.          AMOVEUB  -3  3d  value of amplitude\n")
        f_io.write("0.          AMOVEUB  -4  4th value of amplitude\n")
        f_io.write("            FILTMSA  file name for TIME SERIES [IMOVEB=6]\n")

    contents: str = f_io.getvalue().expandtabs(4)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(contents)


def pm_file() -> None:
    """
    Write the pm.input file used for the vortex particle parallelization.
    """
    fname = "pm.input"
    with open(fname, "w", encoding="utf-8") as file:
        file.write(f"8.0{tabs(2)}8.0{tabs(1)}8.0{tabs(3)}! Dpm[X,Y,Z)\n")
        file.write(f"4{tabs(6)}! projection fun\n")
        file.write(f"2{tabs(6)}! boundary cond type\n")
        file.write(f"1{tabs(6)}! Variable/Constant Volume(0,1)\n")
        file.write(f"0{tabs(6)}! EPSVOL\n")
        file.write(f"8{tabs(6)}! ncoarse\n")
        file.write(f"4  1  1{tabs(3)}! Number of Blocks[i,j,k) == np\n")
        file.write(f"1  1{tabs(4)}! remesh(0,1)/Number of Particels per cell\n")
        file.write(f"1  3{tabs(4)}! tree(0,1),Number of levels\n")
        file.write(f"12{tabs(6)}! Number of threads       ==max(available threads)\n")
        file.write(f"0\n")
        file.write(f"0\n")
        file.write(f"0{tabs(6)}! IPMWRITE  number of pm time series, max value 10\n")

        # UKNOWN
        file.write(f"649  72\n")
        file.write(f"1008  72\n")
        file.write(f"900 90\n")
        file.write(f"1200 90\n")
        file.write(f"4410 90\n")


def cld_files(
    foil_dat: Struct,
    bodies: list[GenuSurface],
    solver: str,
) -> None:
    """
    Write the .cld files for the airfoils. These files contain the CL-CD-CM polars

    Args:
        foil_dat (Struct): Foil Database Data containing all airfoils, solver and Reynolds
        bodies (list[GenuSurface]): list of bodies in GenuSurface format
        solver (str): preferred solver
    """
    for bod in bodies:
        fname: str = f"{bod.cld_fname}"
        try:
            polars: dict[str, DataFrame] = foil_dat[bod.airfoil_name][solver]

        except KeyError:
            try:
                polars = foil_dat[f"NACA{bod.airfoil_name}"][solver]
            except KeyError:
                raise KeyError(f"Airfoil {bod.airfoil_name} not found in database")

        f_io = StringIO()
        f_io.write(f"CL-CD POLARS by {solver}\n")

        # WRITE MACH AND REYNOLDS NUMBERS
        f_io.write(f"{len(polars)}  ! Mach/Reynolds combs for which CL-CD are given\n")
        for i, _ in enumerate(polars.keys()):
            f_io.write(f"0.08000{tabs(1)}")
        f_io.write("MACH\n")

        for reyn in polars.keys():
            f_io.write(f"{reyn.zfill(4)}{tabs(1)}")
        f_io.write("Reynolds\n")

        f_io.write(f"2{tabs(1)}! stations\n")
        f_io.write("\n")

        # GET ALL 2D Airfoil POLARS IN ONE TABLE
        polar_obj = Polars(polars)
        df: DataFrame = polar_obj.df
        # Get Angles
        angles: FloatArray = polar_obj.angles
        # FILL FILE
        for radpos in [-100.0, 100.0]:
            f_io.write(f"!profile: {radpos}\n")
            f_io.write(f"{radpos}{tabs(1)}{0.25}{tabs(1)}{1}{tabs(7)}Span AerCentr NumFlap\n")

            anglenum: int = len(angles)
            flap_angle: float = polar_obj.flap_angle  # Flap Angle
            a_zero_pot: float = polar_obj.a_zero_pot  # Potential Zero Lift Angle
            cm_pot: float = polar_obj.cm_pot  # Potential Cm at Zero Lift Angle
            a_zero: float = polar_obj.a_zero_visc  # Viscous Zero Lift Angle
            cl_slope: float = polar_obj.cl_slope_visc  # Slope of Cl vs Alpha (viscous)

            f_io.write(f"{anglenum} ")
            for item in [flap_angle, a_zero_pot, cm_pot, a_zero, cl_slope]:
                f_io.write(f"{ff4(item)} ")
            f_io.write(f"{tabs(3)}NumAlpha FlapAng AZERpot CMPot AZERO CLslope\n")

            for i, ang in enumerate(angles):
                string: str = ""
                nums = df.loc[df["AoA"] == ang].to_numpy().squeeze()
                for num in nums:
                    string = string + ff2(num) + "  "
                f_io.write(f"{string}\n")
            f_io.write("\n")
        f_io.write("\n")
        # Write to File
        contents: str = f_io.getvalue().expandtabs(4)
        with open(fname, "w", encoding="utf-8") as file:
            file.write(contents)


def body_connections(NBs: int, name: str) -> None:
    """
    Write the .bcon file for the body connections
    ! TODO: Implement

    Args:
        name (str): _description_
    """
    f_io = StringIO()
    f_io.write(f"{NBs}{tabs(4)}! Number of Body Connections\n")
    f_io.write(f"{tabs(4)}<blank>\n")
    for i in range(NBs):
        f_io.write(f"<{i} connection>\n")
        f_io.write(f"2{tabs(4)}! Type of connection single btype=3\n")
        f_io.write(f"1{tabs(4)}! Number of bodies\n")
        f_io.write(f"{i+1} {1} {1}{tabs(3)}! Index of connected Body\n")
        f_io.write(f"<end of connection>\n")
    with open(f"{name}.bcon", "w", encoding="utf-8") as file:
        file.write(f_io.getvalue().expandtabs(4))


def wake_connections(name: str) -> None:
    """
    Write the .wcon file for the wake connections
    ! TODO: Implement

    Args:
        name (str): _description_
    """
    f_io = StringIO()
    f_io.write(f"0{tabs(4)}!Number of Wake Connections\n")
    f_io.write(f"\n")
    with open(f"{name}.wcon", "w", encoding="utf-8") as file:
        file.write(f_io.getvalue().expandtabs(4))


def angles_inp(
    foil_dat: Struct,
    airfoils: list[str],
    solver: str,
) -> None:
    angles: list[float] = []
    # Find all distinct angles in foil_dat.
    for airf in airfoils:
        polars: dict[str, DataFrame] = foil_dat[airf][solver]
        for reyn in polars.keys():
            angles.extend(polars[reyn]["AoA"].to_list())
    angles = [angle for angle in angles if np.isnan(angle) == False]
    angles = list(set(angles))
    angles.sort()

    f_io = StringIO()
    f_io.write(f"{len(angles)}{tabs(4)}! Number of Angle of Attack\n")
    for ang in angles:
        f_io.write(f"{ff2(ang)}{tabs(4)}\n")

    f_io.write(f"\n")

    with open("angles.inp", "w", encoding="utf-8") as file:
        file.write(f_io.getvalue().expandtabs(4))


def make_input_files(
    ANGLEDIR: str,
    HOMEDIR: str,
    movements: list[list[Movement]],
    bodies_dicts: list[GenuSurface],
    params: GenuParameters,
    airfoils: list[str],
    solver: str,
) -> None:
    os.chdir(ANGLEDIR)

    # Input File
    input_file(params.maxiter, params.timestep)
    # PM File
    pm_file()
    # DFILE
    dfile(params)
    # GEO
    geofile(movements, bodies_dicts, params)
    # TOPOLOGY Files
    topology_files(bodies_dicts)
    # BODY CONNECTIONS
    body_connections(len(bodies_dicts), params.name)
    # Wake Connections
    wake_connections(params.name)
    # WAKE Files
    wake_files(bodies_dicts)
    # ANGLES File
    angles_inp(DB.foils_db.data, airfoils, solver)
    # CLD FILES
    cld_files(DB.foils_db.data, bodies_dicts, solver)

    if "gnvp7" not in next(os.walk("."))[2]:
        src: str = os.path.join(HOMEDIR, "ICARUS", "gnvp7")
        dst: str = os.path.join(ANGLEDIR, "gnvp7")
        os.symlink(src, dst)
    os.chdir(HOMEDIR)


def remove_results(CASEDIR: str, HOMEDIR: str) -> None:
    """Removes the simulation results from a GNVP3 case

    Args:
        CASEDIR (str): _description_
        HOMEDIR (str): _description_
    """
    os.chdir(CASEDIR)
    os.remove("*dat")
    os.remove("*err")
    os.remove("*out")
    os.remove(".bak")
    os.remove(".BAK")
    os.remove("fort*")

    os.remove("LOA*")
    os.remove("*.TOT")
    os.remove("CLD_res*")
    os.remove("INIT_check")
    os.remove("GWIN*")
    os.remove("NWAK*")
    os.remove("nwake")

    os.remove("cpu*")
    os.chdir(HOMEDIR)
