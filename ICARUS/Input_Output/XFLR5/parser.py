import os
import re

import numpy as np
from xmltodict import parse

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.types import FloatArray
from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing_segment import define_linear_chord
from ICARUS.Vehicle.wing_segment import define_linear_span
from ICARUS.Vehicle.wing_segment import Wing_Segment


def parse_xfl_project(filename: str) -> Airplane:
    """
    Function to parse the xflr5 project file as exported in xml format.
    Shortcomings:
    - No airfoil morphing. XFLR can interpolate in between airfoils, but this is not implemented.

    Args:
        filename (str): Path to the xflr5 project file

    Returns:
        Airplane: Airplane Object Equivalent to XFLR5 design
    """
    with open(filename) as f:
        my_xml: str = f.read()

    dict = parse(
        my_xml,
        encoding="utf-8",
        process_namespaces=False,
        namespace_separator=":",
    )["explane"]

    # version = dict["@version"]
    # units = dict["Units"]
    plane = dict["Plane"]

    point_masses: list[tuple[float, FloatArray]] = []
    if plane["Inertia"] is not None:
        for pmass in plane["Inertia"]["Point_Mass"]:
            m = float(pmass["Mass"])
            coor: FloatArray = np.array(
                pmass["coordinates"].replace(",", "").split(),
                dtype=float,
            )
            point_masses.append((m, coor))

    plane_name = plane["Name"]
    # description = plane["Description"]

    # Check if there are multiple wings
    if isinstance(plane["wing"], list):
        wings_xflr = plane["wing"]
    else:
        wings_xflr = [plane["wing"]]

    lifting_surfaces: list[Wing_Segment] = []
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    for wing in wings_xflr:
        wing_position: FloatArray = np.array(
            wing["Position"].replace(",", "").split(),
            dtype=float,
        )
        is_symmetric: bool = wing["Symetric"] == "true"
        mass = float(wing["Inertia"]["Volume_Mass"])
        tilt_angle = float(wing["Tilt_angle"])

        if wing["Name"] == "Main Wing":
            name = "wing"
            wing_orientation = np.array(
                (
                    tilt_angle,
                    0.0,
                    0.0,
                ),
            )
        elif wing["Name"] == "Elevator":
            name = "tail"
            wing_orientation = np.array(
                (
                    tilt_angle,
                    0.0,
                    0.0,
                ),
            )
        elif wing["Name"] == "Fin":
            name = "rudder"
            wing_orientation = np.array(
                (
                    tilt_angle,
                    0.0,
                    90.0,
                ),
            )
            is_symmetric = False
        else:
            raise ValueError("Unknown wing name")

        airfoil_prev: AirfoilD | None = None
        y_pos_prev: float = 0
        offset_prev: float = 0
        chord_prev: float = 0
        N_prev: int = 0
        M_prev: int = 0
        span: float = 0
        section_position: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
        dihedral_prev: float = 0

        for i, section in enumerate(wing["Sections"]["Section"]):
            # Get Section Details
            dihedral: float = float(section["Dihedral"])

            # Define the regex pattern
            pattern = r"\([^)]*\)|[^0-9a-zA-Z]+"

            # Use re.sub to remove text inside parentheses and non-alphanumeric characters
            foil_name = section["Left_Side_FoilName"]
            cleaned_string = re.sub(pattern, " ", foil_name)

            # Split the cleaned string into numeric and text parts
            foil = "".join(filter(str.isdigit, cleaned_string))
            text_part = "".join(filter(str.isalpha, cleaned_string))

            if text_part.find("flap") != -1:
                # We should try to load the flap airfoil from the xflr5 data
                # check if there is a flap airfoil folder in XFLR5/
                # if not, raise an error

                # list the folders in the XFLR5DB
                folders: list[str] = os.listdir(XFLRDB)
                if foil_name in folders:
                    # list the files in the flap folder
                    flap_files: list[str] = os.listdir(os.path.join(XFLRDB, foil_name))
                    # check if the airfoil is in the flap folder
                    if foil_name + ".dat" in flap_files:
                        # load the airfoil from the flap folder
                        filename = os.path.join(XFLRDB, foil_name, foil_name + ".dat")
                        airfoil: AirfoilD = AirfoilD.load_from_file(filename)
                        airfoil.name = f"{foil + 'fl'}"
                    else:
                        raise FileNotFoundError(f"Couldnt Find Airfoil {foil_name} in XFLR5DB")
                else:
                    raise FileNotFoundError(f"Couldnt Find Airfoil {foil_name} in XFLR5DB")

            elif "naca" in text_part.lower():
                if len(foil) == 4:
                    airfoil: AirfoilD = AirfoilD.naca(foil)
                    airfoil.name = foil
                elif len(foil) == 5:
                    airfoil = AirfoilD.naca(foil)
                    airfoil.name = foil
                else:
                    raise ValueError(f"Unknown NACA {foil}")
            else:
                raise ValueError(f"Unknown Airfoil {foil}")

            # twist: float = float(section["Twist"])  # ! TODO: IMPLEMENT TWIST
            y_pos: float = float(section["y_position"])
            offset: float = float(section["xOffset"])
            N: int = int(section["x_number_of_panels"])
            M: int = int(section["y_number_of_panels"])
            chord: float = float(section["Chord"])

            if airfoil_prev is not None:
                # if i  is in [2,3]:
                # XFLR Positions c/4 points where as I position LE.
                span = 2 * (y_pos - y_pos_prev) if is_symmetric else (y_pos - y_pos_prev)
                pos = origin + wing_position + section_position  # - np.array((chord_prev / 4, 0, 0))
                surf = Wing_Segment(
                    name=f"{name}_{i}",
                    airfoil=airfoil_prev,  # Should interpolate. RN there is only taking the prev airfoil
                    origin=pos,
                    orientation=wing_orientation,
                    is_symmetric=is_symmetric,
                    span=span,
                    sweep_offset=offset - offset_prev,
                    dih_angle=dihedral_prev,
                    chord_fun=define_linear_chord,
                    chord=np.array((chord_prev, chord), dtype=float),
                    span_fun=define_linear_span,
                    N=N_prev,
                    M=M_prev,
                    mass=mass,
                )
                lifting_surfaces.append(surf)
                if is_symmetric:
                    section_position += np.array([0, span / 2, np.sin(dihedral_prev * np.pi / 180) * span / 2])
                else:
                    section_position += np.array(
                        [0, np.cos(dihedral * np.pi / 180) * span, np.sin(dihedral * np.pi / 180) * span],
                    )
                section_position[0] = offset

            airfoil_prev = airfoil
            y_pos_prev = y_pos
            offset_prev = offset
            chord_prev = chord
            dihedral_prev = dihedral
            N_prev = N
            M_prev = M
    airplane: Airplane = Airplane(
        name=plane_name,
        surfaces=lifting_surfaces,
    )
    airplane.add_point_masses(point_masses)
    # airplane.visualize()
    return airplane
