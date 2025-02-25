import numpy as np
from .hermes import hermes


def get_wing_attr(
    aspect_ratio,
    surface_area,
    taper_ratio,
):
    "Should return root chord, tip chord and span"
    S = surface_area
    half_span = np.sqrt(aspect_ratio * S)
    tip_chord = 2 * (S / (half_span)) * (1 / (1 + taper_ratio))
    root_chord = tip_chord * taper_ratio
    return root_chord, tip_chord, half_span


def pfa(x, name):
    plane = hermes(name)
    payload_position_x = x[0]

    elevator_position_x = x[1]
    elevator_aspect_ratio = x[2]
    elevator_surface_area = x[3]
    elevator_taper_ratio = x[4]

    rudder_position_x = x[5]
    rudder_aspect_ratio = x[6]
    rudder_surface_area = x[7]
    rudder_taper_ratio = x[8]
    wing_dihedral = x[9]

    plane.set_property("payload_position_x", payload_position_x)
    plane.set_property("elevator_position_x", elevator_position_x)
    plane.set_property("rudder_position_x", rudder_position_x)

    elevator_root_chord, elevator_tip_chord, elevator_span = get_wing_attr(
        elevator_aspect_ratio,
        elevator_surface_area,
        elevator_taper_ratio,
    )
    plane.set_property("elevator_root_chord", elevator_root_chord)
    plane.set_property("elevator_tip_chord", elevator_tip_chord)
    plane.set_property("elevator_span", 2 * elevator_span)

    rudder_root_chord, rudder_tip_chord, rudder_span = get_wing_attr(
        rudder_aspect_ratio,
        rudder_surface_area,
        rudder_taper_ratio,
    )
    plane.set_property("rudder_root_chord", rudder_root_chord)
    plane.set_property("rudder_tip_chord", rudder_tip_chord)
    plane.set_property("rudder_span", rudder_span)
    plane.set_property("wing_root_dihedral_angle", wing_dihedral)
    plane.set_property("wing_tip_dihedral_angle", wing_dihedral)
    return plane


def get_goat(name):
    array_goat = np.array([0.1, 0.525, 4.825, 0.04204, 4.93333333, 0.5, 0.76666667, 0.009, 2.6, 4])
    the_goat = pfa(array_goat, name)
    return the_goat


if __name__ == "__main__":
    plane = get_goat("the_goat")
    plane.visualize()
