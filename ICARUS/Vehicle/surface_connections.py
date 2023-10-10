"""
Class to Define the Surface Connections of the Vehicle.
Surface Connections specify wing_segments and fuselage_segments that are
practically connected to each other. This is important to define where we
have the formation of vortexes and where we don't. Also it specifies how
the wakes of the different surfaces interact with each other (The are
the same in the connection).
"""
# ! TODO: This class is a prototype.
# ! In the future, this class will be used to define the connections between
# ! different surfaces, and how they interact with each other in a much more
# ! robust manner. Right now, it is just a placeholder to make GNVP3-7 work
# ! when we have multiple wing segments.


class Surface_Connection:
    """
    _summary_:
        Class to define the connection between two wing segments.
    """

    def __init__(self) -> None:
        pass
