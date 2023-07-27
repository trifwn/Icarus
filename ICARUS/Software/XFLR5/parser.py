from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from xmltodict import parse

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import define_linear_chord
from ICARUS.Vehicle.wing import define_linear_span
from ICARUS.Vehicle.wing import Wing


def parse_xfl_project(filename: str) -> Airplane:
    """
    Function to parse the xflr5 project file as exported in xml format.
    Shortcomings:
        - No airfoil morphing. XFLR can interpolate in between airfoils, but
        this is not implemented.
        - No Multiple Sections. Right now only linear interpolation beetween
        root and tip chord.

    Args:
        filename (str): Path to the xflr5 project file

    Returns:
        Wing: _description_
    """
    with open(filename) as f:
        my_xml: str = f.read()

    dict = parse(
        my_xml,
        encoding="utf-8",
        process_namespaces=False,
        namespace_separator=":",
    )["explane"]

    version = dict["@version"]
    units = dict["Units"]
    plane = dict["Plane"]

    point_masses: list[tuple[float, FloatArray]] = []
    for pmass in plane["Inertia"]["Point_Mass"]:
        m = float(pmass["Mass"])
        coor: FloatArray = np.array(
            pmass["coordinates"].replace(",", "").split(),
            dtype=float,
        )
        point_masses.append((m, coor))

    plane_name = plane["Name"]
    description = plane["Description"]

    wings_xflr = plane["wing"]
    lifting_surfaces: list[Wing] = []
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=floating)

    for wing in wings_xflr:
        # pprint.pprint(wing)
        wing_position: FloatArray = np.array(
            wing["Position"].replace(",", "").split(),
            dtype=float,
        )
        is_symmetric = wing["Symetric"] == "true"
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
        else:
            raise ValueError("Unknown wing name")

        for i, section in enumerate(wing["Sections"]["Section"]):
            if i % 2 == 0:
                dihedral = float(section["Dihedral"])
                foil = section["Right_Side_FoilName"][-4:]
                airfoil = AirfoilD.naca(foil)

                twist = float(section["Twist"])
                start_offset = float(section["xOffset"])
                y_start_position = float(section["y_position"])

                N = int(section["x_number_of_panels"])
                M = int(section["x_number_of_panels"])
                start_chord: float = float(section["Chord"])

            if i % 2 == 1:
                end_chord = float(section["Chord"])
                end_offset = float(section["xOffset"])
                y_end_position = float(section["y_position"])
                pos = origin + wing_position - np.array((start_chord / 4, 0, 0))

                if name == "rudder":
                    pos += np.array((start_chord / 4, 0, 0))

                surf = Wing(
                    name=name,
                    airfoil=airfoil,
                    origin=pos,
                    orientation=wing_orientation,
                    is_symmetric=is_symmetric,
                    span=2 * (y_end_position - y_start_position),
                    sweep_offset=-(start_offset - end_offset),
                    dih_angle=dihedral,
                    chord_fun=define_linear_chord,
                    chord=np.array((start_chord, end_chord), dtype=float),
                    span_fun=define_linear_span,
                    N=N,
                    M=M,
                    mass=mass,
                )
                lifting_surfaces.append(surf)

    airplane: Airplane = Airplane(
        name=plane_name,
        surfaces=lifting_surfaces,
    )

    airplane.add_point_masses(point_masses)
    return airplane
