def geom():
    print("Testing Geometry...")

    from Data.Planes.simple_wing import Simplewing

    return (
        Simplewing.S,
        Simplewing.mean_aerodynamic_chord,
        Simplewing.Area,
        Simplewing.CG,
        Simplewing.inertia,
    )
