"""
Genu Surface Class
A class to define lifting surfaces for the GenuVP solvers.
"""
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.wing import Wing


class GenuSurface:
    def __init__(self, surf: Wing, idx: int) -> None:
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

        self.type: str = 'thin'
        self.lifting: str = "yes"
        self.NB: int = idx
        self.NACA: str = surf.airfoil.name
        self.name: str = surf.name
        self.bld_fname: str = f"{surf.name}.bld"
        self.topo_fname: str = f"{surf.name}.topo"
        self.wake_fname: str = f"{surf.name}.wake"
        self.cld_fname: str = f"{surf.airfoil.name}.cld"
        self.move_fname: str = f"{surf.name}.mov"
        self.grid_fname: str = f"{surf.name}.grid"
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
