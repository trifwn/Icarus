def geom():
    print("Testing Geometry...")

    from Data.Planes.simple_wing import Simplewing

    return (
        Simplewing.S,
        Simplewing.MAC,
        Simplewing.Area,
        Simplewing.CG,
        Simplewing.INERTIA,
    )
