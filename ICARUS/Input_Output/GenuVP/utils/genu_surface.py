"""
Genu Surface Class
A class to define lifting surfaces for the GenuVP solvers.
"""
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.wing_segment import Wing_Segment


class GenuSurface:
    surf_names: dict[str, int] = {}
    airfoil_names: dict[str, int] = {}

    def __init__(self, surf: Wing_Segment, idx: int) -> None:
        """
        Converts a Wing Object to an object that can be used
        for making the input files of GNVP3 and GNVP7

        Args:
            surf (Wing): Wing Object
            idx (int): IND of the surface to be assigned

        Returns:
            dict[str, Any]: Dict with the surface parameters
        """
        if surf.is_symmetric:
            N: int = 2 * surf.N - 1
            M: int = surf.M
        else:
            N = surf.N
            M = surf.M

        # We have to make sure that the surface names and airfoil names
        # are unique for each surface and airfoil and also give them names
        # that are valid for GenuVP. This means a total of 12 characters

        # First we check if the surface name is unique
        if surf.name[:8] not in GenuSurface.surf_names.keys():
            GenuSurface.surf_names[surf.name[:8]] = 0
        else:
            GenuSurface.surf_names[surf.name[:8]] += 1
        # Then we check if the airfoil name is unique
        if surf.airfoil.name not in GenuSurface.airfoil_names.keys():
            GenuSurface.airfoil_names[surf.airfoil.name[:8]] = 0
        else:
            GenuSurface.airfoil_names[surf.airfoil.name[:8]] += 1

        name: str = f"{GenuSurface.surf_names[surf.name[:8]]}_{surf.name}"
        airfoil_name: str = f"{GenuSurface.airfoil_names[surf.airfoil.name[:8]]}_{surf.airfoil.name}"

        # Make the names valid for GenuVP and make sure they are unique
        # by adding a number to the end of the name
        name = name[:8]
        airfoil_name = airfoil_name[:8]

        # Now we can start assigning the parameters
        self.type: str = "thin"
        self.lifting: str = "yes"
        self.NB: int = idx
        self.NACA: str = surf.airfoil.name
        self.name: str = surf.name
        self.airfoil_name: str = surf.airfoil.name
        self.bld_fname: str = f"{name}.bld"
        self.topo_fname: str = f"{name}.topo"
        self.wake_fname: str = f"{name}.wake"
        self.cld_fname: str = f"{airfoil_name}.cld"
        self.move_fname: str = f"{name}.mov"
        self.grid_fname: str = f"{name}.grid"
        self.NNB: int = M
        self.NCWB: int = N
        self.x_0: float = surf.origin[0]
        self.y_0: float = surf.origin[1]
        self.z_0: float = surf.origin[2]
        self.pitch: float = surf.orientation[0]
        self.cone: float = surf.orientation[1]
        self.wngang: float = surf.orientation[2]
        self.x_end: float = surf.origin[0] + surf._offset_dist[-1]
        self.y_end: float = surf.origin[1] + surf.span
        self.z_end: float = surf.origin[2] + surf._dihedral_dist[-1]
        self.Root_chord: float = surf.chord[0]
        self.Tip_chord: float = surf.chord[-1]
        self.Offset: float = surf._offset_dist[-1]
        self.Grid: FloatArray = surf.getGrid()
